"""Drawing the map: terrain, coverage, routes, incidents, safety layer and drones."""
from __future__ import annotations

import math

import numpy as np
import pygame

from dashboard.drawing import mix, smooth_noise, blend_circle, dashed_line, arrow
from dashboard.style import (CELL, GRID_ORIGIN, TRAIL_LEN, FLASH_FRAMES, PIN_FRAMES, MUTED, GROUND,
                             GROUND_DRY, CANOPY, COVER, FOG, VORONOI, AMBER, DEAD, HOME, PLAN_C, TRACK_C,
                             RADIO, SAFETY, FIRE_C, INTRUDER_C, INCIDENT_COLORS, RETURN, DOCKED, TRACK,
                             UAV_COLORS, TYPE_COLORS)


class MapView:
    """Map drawing for Dashboard."""

    def _cell_rect(self, r, c):
        return pygame.Rect(c * CELL, r * CELL, CELL, CELL)

    def _terrain_surface(self):
        """Forest floor with soft colour variation, and tree crowns that merge into canopy."""
        size = self.grid_n * CELL
        rng = np.random.default_rng(int(self.meta["seed"]) + 1000)     # decoration only
        shade = 0.6 * smooth_noise(rng, 7, size) + 0.4 * smooth_noise(rng, 22, size)
        dry = np.clip((smooth_noise(rng, 5, size) - 0.55) * 2.5, 0, 1)[..., None] * 0.6
        rgb = np.array(GROUND, np.float32) * (1 - dry) + np.array(GROUND_DRY, np.float32) * dry
        rgb = rgb * (0.78 + 0.44 * shade[..., None]) + rng.normal(0, 3.0, (size, size, 1))
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        s = pygame.surfarray.make_surface(np.ascontiguousarray(rgb.transpose(1, 0, 2)))
        cells = list(zip(*np.where(self.ep["obstacle_grid"])))
        jit = rng.uniform(-2.0, 2.0, (len(cells), 4))
        rad = rng.uniform(0.55, 0.72, (len(cells), 2)) * CELL
        tone = rng.integers(0, len(CANOPY), (len(cells), 2))
        centres = [((c + 0.5) * CELL, (r + 0.5) * CELL) for r, c in cells]
        shadow = pygame.Surface((size, size), pygame.SRCALPHA)
        for (x, y), j, rd in zip(centres, jit, rad):
            pygame.draw.circle(shadow, (4, 14, 8, 150), (x + 3 + j[0], y + 3 + j[1]), rd[0] + 1)
        s.blit(shadow, (0, 0))
        for (x, y), j, rd, tn in zip(centres, jit, rad, tone):
            pygame.draw.circle(s, CANOPY[tn[0]], (x + j[0], y + j[1]), rd[0])
            pygame.draw.circle(s, CANOPY[tn[1]], (x + j[2], y + j[3]), rd[1] * 0.8)
        light = pygame.Surface((size, size), pygame.SRCALPHA)
        for (x, y), j, rd in zip(centres, jit, rad):
            pygame.draw.circle(light, (120, 190, 110, 55), (x + j[0] - 2, y + j[1] - 2), rd[0] * 0.45)
        s.blit(light, (0, 0))
        return s

    def _voronoi_surface(self):
        s = pygame.Surface((self.grid_n * CELL, self.grid_n * CELL), pygame.SRCALPHA)
        region = self.ep["region_masks"].argmax(axis=0)
        n = self.grid_n
        for r in range(n):
            for c in range(n):
                if (r + c) % 2:
                    continue
                if c + 1 < n and region[r, c] != region[r, c + 1]:
                    x = (c + 1) * CELL
                    pygame.draw.line(s, VORONOI, (x, r * CELL), (x, (r + 1) * CELL), 2)
                if r + 1 < n and region[r, c] != region[r + 1, c]:
                    y = (r + 1) * CELL
                    pygame.draw.line(s, VORONOI, (c * CELL, y), ((c + 1) * CELL, y), 2)
        for i, (rr, cc) in enumerate(self.ep["region_centers"]):
            x, y = float((cc + 0.5) * CELL), float((rr + 0.5) * CELL)
            col = (*UAV_COLORS[i], 170)
            pygame.draw.line(s, col, (x - 6, y), (x + 6, y), 2)
            pygame.draw.line(s, col, (x, y - 6), (x, y + 6), 2)
            s.blit(self.font.render(f"R{i}", True, col), (x + 6, y - 18))
        return s

    def _footprint_surface(self, color):
        r = int(self.meta["obs_radius"])
        d = np.arange(-r, r + 1)
        dx, dy = np.meshgrid(d, d, indexing="ij")
        mask = dx ** 2 + dy ** 2 <= r ** 2
        s = pygame.Surface(((2 * r + 1) * CELL, (2 * r + 1) * CELL), pygame.SRCALPHA)
        for i, j in zip(*np.where(mask)):
            s.fill((*color, 30), self._cell_rect(i, j))
        centre = ((r + 0.5) * CELL, (r + 0.5) * CELL)
        pygame.draw.circle(s, (*color, 120), centre, (r + 0.5) * CELL, 1)
        return s

    @staticmethod
    def _intruder_surface(alpha):
        s = pygame.Surface((20, 24), pygame.SRCALPHA)
        pygame.draw.polygon(s, INTRUDER_C, [(4, 22), (16, 22), (13, 11), (7, 11)])
        pygame.draw.circle(s, INTRUDER_C, (10, 6), 4)
        pygame.draw.circle(s, (255, 255, 255), (10, 6), 4, 1)
        pygame.draw.polygon(s, (255, 255, 255), [(4, 22), (16, 22), (13, 11), (7, 11)], 1)
        s = s.convert_alpha()
        s.set_alpha(int(255 * alpha))
        return s

    def _draw_map(self):
        f, ep = self.frame, self.ep
        size = self.grid_n * CELL
        self.anim = pygame.time.get_ticks() / 1000.0
        surf = self.screen.subsurface(pygame.Rect(GRID_ORIGIN, (size, size)))    # draw straight onto the screen
        surf.blit(self._ground(f), (0, 0))
        pos = self._lerp(ep["positions"])
        flying = self._flying(f)
        if self.show["footprint"]:
            r = int(self.meta["obs_radius"])
            for u in np.where(flying)[0]:
                gr, gc = int(pos[u, 0]), int(pos[u, 1])
                surf.blit(self.fp_surf[u], ((gc - r) * CELL, (gr - r) * CELL))

        over = self.overlay
        over.fill((0, 0, 0, 0))
        self._radio_graph(over, pos, flying)
        if self.show["trails"]:
            lo = max(0, f - TRAIL_LEN)
            for u in range(self.n_uav):
                pts = [self._pos_px(p) for p in ep["positions"][lo:f + 1, u]] + [self._pos_px(pos[u])]
                for k in range(1, len(pts)):
                    a = int(40 + 180 * k / len(pts))
                    pygame.draw.line(over, (*UAV_COLORS[u], a), pts[k - 1], pts[k], 2)
        if self.show["routes"] and self.has_plans:
            self._draw_routes(over, f, pos)
        self._draw_targets(over, f)
        surf.blit(over, (0, 0))

        epos = self._lerp(ep["event_pos"]) if self.has_incidents else None
        if self.has_incidents and self.show["events"]:
            self._draw_incidents(surf, f, epos, pos)
        if self.show["safety"] and self.safety:
            self._draw_safety(surf, f, pos, flying)
        self._draw_uavs(surf, f, pos)
        if self.has_incidents and self.show["events"]:
            self._draw_pins(surf, f, epos)
        pygame.draw.rect(self.screen, (60, 72, 84), (*GRID_ORIGIN, size, size), 1)

    def _ground(self, f):
        """Terrain with the coverage, fog, Voronoi and base layers; rebuilt only when the frame or a layer changes."""
        key = (f, self.show["coverage"], self.show["fog"], self.show["voronoi"])
        if self._ground_cache[0] == key:
            return self._ground_cache[1]
        size = self.grid_n * CELL
        surf = self.terrain.copy()
        if self.show["coverage"]:
            rgba = np.zeros((self.grid_n, self.grid_n, 4), np.uint8)
            rgba[self.ep["coverage_map"][f]] = COVER
            small = pygame.image.frombuffer(rgba.tobytes(), (self.grid_n, self.grid_n), "RGBA").convert_alpha()
            surf.blit(pygame.transform.scale(small, (size, size)), (0, 0))
        if self.show["fog"]:
            surf.blit(self._fog(f), (0, 0))
        if self.show["voronoi"]:
            surf.blit(self.voronoi, (0, 0))
        self._draw_base(surf)
        self._ground_cache = (key, surf)
        return surf

    def _fog(self, f):
        """Fog of war: clear where a sensor looked recently, thickening with time since; dense where never seen."""
        ls = self.last_seen[f].astype(np.int32)
        alpha = np.clip((f - ls - 10) / (3.0 * self.window), 0, 1) * 150
        alpha[ls < 0] = 215
        rgba = np.zeros((self.grid_n, self.grid_n, 4), np.uint8)
        rgba[..., :3] = FOG
        rgba[..., 3] = alpha.astype(np.uint8)
        small = pygame.image.frombuffer(rgba.tobytes(), (self.grid_n, self.grid_n), "RGBA").convert_alpha()
        size = self.grid_n * CELL
        return pygame.transform.smoothscale(small, (size, size))

    def _draw_base(self, s):
        if not self.pads:
            bx, by = self._cell_px(1, 1)
            pygame.draw.rect(s, (235, 235, 235), (bx - 6, by - 2, 12, 9))
            pygame.draw.polygon(s, (235, 235, 235), [(bx - 8, by - 1), (bx, by - 9), (bx + 8, by - 1)])
            return
        apron = pygame.Surface((7 * CELL, 7 * CELL), pygame.SRCALPHA)
        pygame.draw.rect(apron, (190, 190, 175, 46), (CELL // 2, CELL // 2, 6 * CELL, 2 * CELL), border_radius=6)
        pygame.draw.rect(apron, (190, 190, 175, 46), (CELL // 2, CELL // 2, 2 * CELL, 6 * CELL), border_radius=6)
        s.blit(apron, (0, 0))
        for r, c in self.pads:
            x, y = self._cell_px(r, c)
            blend_circle(s, (18, 24, 32, 210), (x, y), 6.5)
            pygame.draw.circle(s, PLAN_C, (x, y), 6.5, 1)
            h = self.font_xs.render("H", True, (225, 230, 235))
            s.blit(h, h.get_rect(center=(x, y)))

    def _radio_graph(self, s, pos, flying):
        """Links between flying drones within radio range, and which drones reach the base by relay."""
        R = float(self.meta["comm_radius"])
        idx = list(np.where(flying)[0])
        nodes = [np.array([1.5, 1.5], dtype=np.float32)] + [pos[u] for u in idx]
        adj = {i: [] for i in range(len(nodes))}
        links = 0
        for a in range(len(nodes)):
            for b in range(a + 1, len(nodes)):
                d = float(np.linalg.norm(nodes[a] - nodes[b]))
                if d > R:
                    continue
                adj[a].append(b)
                adj[b].append(a)
                links += a > 0
                if self.show["radio"]:
                    alpha = int(50 + 150 * (1 - d / R))
                    pygame.draw.line(s, (*RADIO, alpha), self._pos_px(nodes[a]), self._pos_px(nodes[b]), 1)
        seen, stack = {0}, [0]
        while stack:
            for b in adj[stack.pop()]:
                if b not in seen:
                    seen.add(b)
                    stack.append(b)
        self.radio = (links, len(seen) - 1, len(idx))

    def _draw_routes(self, s, f, pos):
        ep = self.ep
        phase = (self.anim * 14) % 11
        for u in range(self.n_uav):
            goal_cell = ep["targets"][f, u]
            if goal_cell[0] < 0 or not ep["active"][f, u] or ep["mode"][f, u] == DOCKED:
                continue
            col = UAV_COLORS[u]
            cells = ep["routes"][f, u]
            cells = cells[cells[:, 0] >= 0]
            pts = [self._pos_px(pos[u])] + [self._cell_px(r, c) for r, c in cells[1:]]
            goal = self._cell_px(*goal_cell)
            if len(pts) > 1:
                dashed_line(s, (*col, 230), pts, 2, 7, 4, phase)
            if len(cells) == 0 or tuple(cells[-1]) != tuple(goal_cell):    # route longer than stored
                dashed_line(s, (*col, 110), [pts[-1], goal], 1, 3, 5, phase)
            gx, gy = goal
            pygame.draw.circle(s, (*col, 235), (gx, gy), 6, 2)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                pygame.draw.line(s, (*col, 235), (gx + 4 * dx, gy + 4 * dy), (gx + 9 * dx, gy + 9 * dy), 2)

    def _draw_targets(self, s, f):
        t = self.anim
        for i, kind in enumerate(self.types):
            x, y = self._cell_px(*self.ep["target_positions"][f, i])
            found = bool(self.ep["detected_mask"][f, i])
            col = TYPE_COLORS[kind]
            pulse = 0.5 + 0.5 * math.sin(t * 4 + i)
            if kind == "animal":
                if found:
                    pygame.draw.circle(s, col, (x, y), 6)
                else:
                    pygame.draw.circle(s, (*col, 90), (x, y), 7 + 3 * pulse, 1)
                    pygame.draw.circle(s, col, (x, y), 6, 2)
            elif kind == "fire":
                flame = [(x, y - 8), (x - 6, y + 6), (x + 6, y + 6)]
                if found:
                    pygame.draw.polygon(s, col, flame)
                    pygame.draw.circle(s, col, (x, y), 10, 1)
                else:
                    pygame.draw.circle(s, (*col, int(60 + 80 * pulse)), (x, y), 9 + 2 * pulse)
                    pygame.draw.polygon(s, (255, 200, 60), [(x, y - 6 - 2 * pulse), (x - 4, y + 5), (x + 4, y + 5)])
            else:
                dia = [(x, y - 7), (x + 7, y), (x, y + 7), (x - 7, y)]
                if found:
                    pygame.draw.polygon(s, col, dia)
                else:
                    pygame.draw.polygon(s, col, dia, 2)
            if found:
                pygame.draw.lines(s, (255, 255, 255), False, [(x - 3, y), (x - 1, y + 3), (x + 4, y - 3)], 2)

    def _draw_incidents(self, s, f, epos, pos):
        ep = self.ep
        watched = set(int(k) for k in ep["track_event"][f] if k >= 0) if "track_event" in ep else set()
        for k in np.where((self.inc_spawn >= 0) & (self.inc_spawn <= f))[0]:
            if np.isnan(epos[k, 0]):
                continue
            x, y = self._pos_px(epos[k])
            seen = 0 <= self.inc_det[k] <= f
            ghost = 1.0 if seen else 0.45
            if 0 <= self.inc_conf[k] <= f and k not in watched:
                self._draw_reported(s, k, x, y, float(ep["event_radius"][f, k]))
                continue
            if self.inc_kind[k] == "fire":
                self._draw_fire(s, k, x, y, float(ep["event_radius"][f, k]), ghost)
            else:
                lo = max(int(self.inc_spawn[k]), f - 12)
                trail = ep["event_pos"][lo:f + 1, k]
                for j, p in enumerate(trail[:-1]):
                    blend_circle(s, (*INTRUDER_C, int((30 + 90 * j / len(trail)) * ghost)), self._pos_px(p), 1.6)
                s.blit(self.intruder_icon[seen], (x - 10, y - 14))
            if not seen:                             # the swarm does not know about it yet
                q = self.font_xs.render("?", True, (230, 230, 230))
                blend_circle(s, (10, 14, 20, 150), (x + 9, y - 11), 6)
                s.blit(q, q.get_rect(center=(x + 9, y - 11)))
        if "track_event" not in ep:
            return
        for u in range(self.n_uav):                  # tracking: a ring in the tracker's colour
            k = int(ep["track_event"][f, u])
            if k < 0 or np.isnan(epos[k, 0]):
                continue
            x, y = self._pos_px(epos[k])
            R = float(ep["event_radius"][f, k]) * CELL + 11
            rect = pygame.Rect(x - R, y - R, 2 * R, 2 * R)
            rot = self.anim * 1.5
            for j in range(8):
                a0 = rot + j * math.pi / 4
                pygame.draw.arc(s, UAV_COLORS[u], rect, a0, a0 + 0.45, 2)
            if not 0 <= self.inc_conf[k] <= f:
                dashed_line(s, UAV_COLORS[u], [self._pos_px(pos[u]), (x, y)], 1, 4, 4, self.anim * 10)

    def _draw_fire(self, s, k, x, y, radius, ghost):
        R = max(radius * CELL, 4.0)
        t = self.anim
        flick = 0.5 + 0.5 * math.sin(t * 11 + k * 1.7)
        blend_circle(s, (40, 18, 10, int(150 * ghost)), (x, y), R + 2)                  # scorched ground
        blend_circle(s, (255, 96, 24, int((70 + 50 * flick) * ghost)), (x, y), R * 0.85 + 1)
        blend_circle(s, (255, 150, 50, int((120 + 80 * flick) * ghost)), (x, y), R + 1, 2)   # burning front
        flames = pygame.Surface((int(2 * R) + 24, int(2 * R) + 24), pygame.SRCALPHA)
        cx, cy = flames.get_width() / 2, flames.get_height() / 2
        for j, (ang, rho, ph) in enumerate(self.flames[k][:2 + int(radius * 2)]):
            fx, fy = cx + math.cos(ang) * rho * R * 0.75, cy + math.sin(ang) * rho * R * 0.75
            h = 5 + 3 * math.sin(t * 9 + ph)
            col = (255, 214, 90) if j % 2 else (255, 130, 40)
            pygame.draw.polygon(flames, col, [(fx, fy - h), (fx - 3, fy + 2), (fx + 3, fy + 2)])
        flames.set_alpha(int(255 * ghost))
        s.blit(flames, (x - cx, y - cy))
        grow = 1 + radius / 4
        for j in range(3):                           # smoke drifting with the wind
            ph = (t * 0.35 + j / 3 + k * 0.13) % 1.0
            blend_circle(s, (150, 150, 150, int(90 * (1 - ph) * ghost)),
                         (x + (6 + 18 * ph) * grow, y - (6 + 26 * ph) * grow), (4 + 8 * ph) * grow)

    def _draw_reported(self, s, k, x, y, radius):
        """An incident already confirmed and handed on: drawn faded so live ones stand out."""
        if self.inc_kind[k] == "fire":
            R = max(radius * CELL, 4.0)
            blend_circle(s, (40, 18, 10, 70), (x, y), R)
            blend_circle(s, (*FIRE_C, 110), (x, y), R, 1)
            pygame.draw.polygon(s, mix(FIRE_C, (60, 60, 60), 0.4), [(x, y - 5), (x - 4, y + 4), (x + 4, y + 4)])
        else:
            s.blit(self.intruder_icon[None], (x - 10, y - 14))

    def _draw_pins(self, s, f, epos):
        ep = self.ep
        tracked = set(int(k) for k in ep["track_event"][f] if k >= 0) if "track_event" in ep else set()
        for k in range(len(self.inc_kind)):
            det = int(self.inc_det[k])
            if not 0 <= det <= f or np.isnan(epos[k, 0]):
                continue
            age = f - det
            if age > PIN_FRAMES and k not in tracked:
                continue
            x, y = self._pos_px(epos[k])
            col = INCIDENT_COLORS[self.inc_kind[k]]
            top = y - 27
            pygame.draw.polygon(s, col, [(x - 5, top + 5), (x + 5, top + 5), (x, y - 9)])
            pygame.draw.circle(s, col, (x, top), 8)
            pygame.draw.circle(s, (255, 255, 255), (x, top), 8, 1)
            bang = self.font_xs.render("!", True, (20, 20, 20))
            s.blit(bang, bang.get_rect(center=(x, top)))
            if age <= PIN_FRAMES:
                conf = 0 <= self.inc_conf[k] <= f
                r, c = int(epos[k, 0]), int(epos[k, 1])
                txt = f"{self.inc_kind[k].upper()} #{k}  t={det}  ({r},{c})" + ("  confirmed" if conf else "")
                self._label(s, txt, x + 11, top - 8, col)

    def _draw_safety(self, s, f, pos, flying):
        ep = self.ep
        ids = list(np.where(flying)[0])
        for u in ids:                                # keep-out ring: other drones stay outside it
            near = any(np.linalg.norm(pos[u] - pos[v]) < 2 * self.separation for v in ids if v != u)
            col = (*AMBER, 150) if near else (255, 255, 255, 40)
            blend_circle(s, col, self._pos_px(pos[u]), self.separation * CELL, 1)
        if "corrected" not in ep:
            return
        labelled = False
        for u in range(self.n_uav):
            g = int(self.last_fix[f, u])
            if g < 1 or f - g >= FLASH_FRAMES:
                continue
            k = 1 - (f - g) / FLASH_FRAMES
            x, y = self._pos_px(pos[u])
            blend_circle(s, (*SAFETY, int(220 * k)), (x, y), 11 + 12 * (1 - k), 2)
            start = self._pos_px(ep["positions"][g - 1, u])     # where the corrected move began
            arrow(s, (255, 80, 80), start, ep["proposed"][g, u], 2.5 * CELL, 2, dashed=True)
            arrow(s, SAFETY, start, ep["actions"][g, u], 2.5 * CELL, 2)
            if k > 0.4 and not labelled:
                self._label(s, "safety fix: red = asked, yellow = flown", x + 14, y + 8, SAFETY)
                labelled = True

    def _draw_drone(self, s, x, y, a, col, outline=(255, 255, 255), width=1, scale=1.0, dim=False):
        """Quadcopter seen from above, in the drone's colour, nose pointing along its heading."""
        col = mix(col, (90, 96, 104), 0.5) if dim else col
        arm = 7.5 * scale
        rotor = mix(col, (255, 255, 255), 0.3)
        for j in range(4):
            ang = a + math.pi / 4 + j * math.pi / 2
            rx, ry = x + arm * math.cos(ang), y + arm * math.sin(ang)
            pygame.draw.line(s, (20, 24, 30), (x, y), (rx, ry), 3)
            pygame.draw.circle(s, rotor, (rx, ry), 3.6 * scale)
            pygame.draw.circle(s, (20, 24, 30), (rx, ry), 3.6 * scale, 1)
        body = 5.2 * scale
        pygame.draw.circle(s, col, (x, y), body)
        pygame.draw.circle(s, outline, (x, y), body, width)
        tip = (x + (body + 5) * math.cos(a), y + (body + 5) * math.sin(a))
        pygame.draw.polygon(s, outline, [tip, (x + body * math.cos(a + 0.6), y + body * math.sin(a + 0.6)),
                                         (x + body * math.cos(a - 0.6), y + body * math.sin(a - 0.6))])

    def _draw_uavs(self, s, f, pos):
        ep = self.ep
        mode = ep["mode"][f] if self.mission else None
        for u in range(self.n_uav):
            x, y = self._pos_px(pos[u])
            m = int(mode[u]) if mode is not None else None
            if m == DOCKED:
                if self.pads:                        # docked on its landing pad
                    self._draw_drone(s, x, y, -math.pi / 2, UAV_COLORS[u], scale=0.75, dim=True)
                    if ep["battery_fraction"][f, u] < 0.999:
                        pygame.draw.polygon(s, PLAN_C, [(x + 7, y - 11), (x + 3, y - 5), (x + 6, y - 5),
                                                        (x + 4, y + 1), (x + 9, y - 7), (x + 6, y - 7)])
                else:                                # docked UAVs wait in a row beside the base station
                    bx, by = self._cell_px(1, 1)
                    pygame.draw.rect(s, UAV_COLORS[u], (bx + 12 + 9 * u, by - 3, 7, 7))
                continue
            if not ep["active"][f, u]:
                pygame.draw.circle(s, DEAD, (x, y), 6)
                pygame.draw.line(s, (30, 30, 30), (x - 4, y - 4), (x + 4, y + 4), 2)
                pygame.draw.line(s, (30, 30, 30), (x - 4, y + 4), (x + 4, y - 4), 2)
                continue
            if ep["collided"][f, u] and f > 0:
                outline, width = AMBER, 2
            elif m == RETURN:
                outline, width = HOME, 2
            elif m == TRACK:
                outline, width = TRACK_C, 2
            else:
                outline, width = (255, 255, 255), 1
            self._draw_drone(s, x, y, self._heading(u), UAV_COLORS[u], outline, width)
            if self.show["badges"] and self.mission:
                short, col = self._badge_style(f, u)
                if short:
                    self._badge(s, short, x, y - 16, col)

    def _badge_style(self, f, u):
        if "flown_by" in self.ep:
            _, short, col = self.flown_style.get(self._in_control(f, u), ("", "", MUTED))
            return short, col
        return ("RTH", HOME) if self.ep["mode"][f, u] == RETURN else ("", MUTED)

    def _badge(self, s, txt, cx, cy, color):
        img = self.font_xs.render(txt, True, (16, 20, 26))
        w, h = img.get_size()
        rect = pygame.Rect(0, 0, w + 6, h)
        rect.center = (cx, cy)
        pygame.draw.rect(s, color, rect, border_radius=h // 2)
        s.blit(img, (rect.left + 3, rect.top))

    def _label(self, s, txt, x, y, color):
        img = self.font_s.render(txt, True, color)
        w, h = img.get_size()
        size = self.grid_n * CELL
        if x + w + 8 > size:
            x = x - w - 30
        x, y = max(2, x), max(2, min(y, size - h - 4))
        bg = pygame.Surface((w + 8, h + 2), pygame.SRCALPHA)
        bg.fill((8, 12, 18, 205))
        s.blit(bg, (x, y))
        s.blit(img, (x + 4, y + 1))
