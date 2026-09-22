"""Interactive replay of an episode recorded by visualization/record_episode.py.

Usage:
    python visualization/pygame_dashboard.py --input <episode.npz> [--speed FPS]

Replays the .npz only. It never imports or runs the environment or MADDPG.
Frame 0 is the reset state; frame t is the state after step t.

Controls
    Space          play / pause
    Right / Left   step one frame forward / back (pauses)
    Shift+arrow    step 10 frames
    Up / Down, + - replay speed
    R / Home       restart from frame 0
    End            jump to last frame
    C F V T        toggle coverage / sensor footprints / Voronoi prior / trails
    Click the progress bar to seek.   Esc or Q quits.
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import pygame

WINDOW = (1280, 760)
CELL = 14
GRID_ORIGIN = (10, 44)
PANEL_X = 730
TRAIL_LEN = 60
SPEEDS = [1, 2, 5, 10, 15, 30, 60, 120]
OBSTACLE = 1

BG = (13, 19, 26)
PANEL_BG = (19, 27, 36)
TEXT = (230, 237, 243)
MUTED = (139, 152, 165)
FREE = (22, 48, 31)
BARK = (61, 43, 29)
LEAF = (34, 78, 44)
COVER = (95, 208, 138, 88)
VORONOI = (255, 255, 255, 80)
AMBER = (255, 191, 0)
DEAD = (110, 116, 124)
UAV_COLORS = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189)]
TYPE_COLORS = {"animal": (74, 163, 255), "fire": (228, 87, 46), "poi": (175, 183, 191)}
TYPE_LABELS = {"animal": "Animals", "fire": "Fire", "poi": "Points of interest"}


def load_episode(path):
    with np.load(path, allow_pickle=False) as z:
        ep = {k: z[k] for k in z.files}
    ep["meta"] = json.loads(str(ep.pop("metadata")))
    return ep


def build_events(ep):
    """Human-readable event log derived from the recorded arrays."""
    ev = [(0, "Episode start: 5 UAVs at base (1,1)", TEXT)]
    types = [str(t) for t in ep["target_types"]]
    det = ep["detected_mask"]
    for i in np.where(det.any(axis=0))[0]:
        f = int(det[:, i].argmax())
        ev.append((f, f"{types[i].upper()} #{i} detected", TYPE_COLORS[types[i]]))
    hits = ep["collided"] & ep["active"]
    hits[0] = False
    for f, u in zip(*np.where(hits)):
        ev.append((int(f), f"UAV{u} collision (blocked by obstacle)", AMBER))
    for u in range(ep["active"].shape[1]):
        low = np.where(ep["battery_fraction"][:, u] < 0.30)[0]
        if len(low):
            ev.append((int(low[0]), f"UAV{u} battery low (<30%)", AMBER))
        off = np.where(~ep["active"][:, u])[0]
        if len(off):
            ev.append((int(off[0]), f"UAV{u} inactive (battery depleted)", MUTED))
    cov = ep["coverage_rate"]
    for m in (0.25, 0.50, 0.75, 0.90, 0.95):
        hit = np.where(cov >= m)[0]
        if len(hit):
            ev.append((int(hit[0]), f"Coverage reached {int(m * 100)}%", (95, 208, 138)))
    last = len(ep["timestep"]) - 1
    ev.append((last, f"Episode end at step {last}", TEXT))
    ev.sort(key=lambda e: e[0])
    return ev


class Dashboard:
    def __init__(self, path, speed=15, screen=None):
        self.ep = load_episode(path)
        self.meta = self.ep["meta"]
        self.n_frames = len(self.ep["timestep"])
        self.grid_n = int(self.meta["grid_size"])
        self.n_uav = int(self.meta["n_agents"])
        self.types = [str(t) for t in self.ep["target_types"]]

        pygame.init()
        self.screen = screen or pygame.display.set_mode(WINDOW)
        pygame.display.set_caption(f"UAV Swarm Replay - seed {self.meta['seed']}")
        self.font = pygame.font.SysFont("segoeui,arial", 14)
        self.font_b = pygame.font.SysFont("segoeui,arial", 15, bold=True)
        self.font_big = pygame.font.SysFont("segoeui,arial", 30, bold=True)
        self.clock = pygame.time.Clock()

        self.frame = 0
        self.playing = True
        self.speed_idx = SPEEDS.index(speed) if speed in SPEEDS else SPEEDS.index(15)
        self.acc = 0.0
        self.show = {"coverage": True, "footprint": True, "voronoi": True, "trails": True}

        self.events = build_events(self.ep)
        hits = self.ep["collided"] & self.ep["active"]
        hits[0] = False
        self.coll_so_far = np.cumsum(hits, axis=0)
        self.headings = self._headings()
        self.terrain = self._terrain_surface()
        self.voronoi = self._voronoi_surface()
        self.fp_surf = [self._footprint_surface(c) for c in UAV_COLORS]
        self.progress_rect = pygame.Rect(PANEL_X + 10, 58, 520, 10)

    # ------------------------------------------------------------ precompute
    def _headings(self):
        vel = self.ep["velocities"]
        h = np.zeros(vel.shape[:2])
        last = np.full(vel.shape[1], math.pi / 4)
        for f in range(vel.shape[0]):
            moving = np.linalg.norm(vel[f], axis=1) > 1e-6
            last[moving] = np.arctan2(vel[f, moving, 0], vel[f, moving, 1])
            h[f] = last
        return h

    def _cell_rect(self, r, c):
        return pygame.Rect(c * CELL, r * CELL, CELL, CELL)

    def _terrain_surface(self):
        s = pygame.Surface((self.grid_n * CELL, self.grid_n * CELL))
        s.fill(FREE)
        obst = self.ep["obstacle_grid"]
        for r, c in zip(*np.where(obst)):
            rect = self._cell_rect(r, c)
            pygame.draw.rect(s, BARK, rect)
            cx, cy = rect.center
            pygame.draw.polygon(s, LEAF, [(cx, rect.top + 2), (rect.left + 2, rect.bottom - 3),
                                          (rect.right - 2, rect.bottom - 3)])
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
            s.fill((*color, 34), self._cell_rect(i, j))
        centre = ((r + 0.5) * CELL, (r + 0.5) * CELL)
        pygame.draw.circle(s, (*color, 120), centre, (r + 0.5) * CELL, 1)
        return s

    # -------------------------------------------------------------- controls
    def set_frame(self, f):
        self.frame = int(np.clip(f, 0, self.n_frames - 1))
        self.acc = 0.0

    def handle_event(self, e):
        if e.type == pygame.QUIT:
            return False
        if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1 and self.progress_rect.inflate(0, 12).collidepoint(e.pos):
            frac = (e.pos[0] - self.progress_rect.left) / self.progress_rect.width
            self.set_frame(round(frac * (self.n_frames - 1)))
        if e.type != pygame.KEYDOWN:
            return True
        step = 10 if e.mod & pygame.KMOD_SHIFT else 1
        k = e.key
        if k in (pygame.K_ESCAPE, pygame.K_q):
            return False
        if k == pygame.K_SPACE:
            if self.frame == self.n_frames - 1 and not self.playing:
                self.set_frame(0)
            self.playing = not self.playing
        elif k in (pygame.K_RIGHT, pygame.K_PERIOD):
            self.playing = False
            self.set_frame(self.frame + step)
        elif k in (pygame.K_LEFT, pygame.K_COMMA):
            self.playing = False
            self.set_frame(self.frame - step)
        elif k in (pygame.K_r, pygame.K_HOME):
            self.set_frame(0)
            self.playing = True
        elif k == pygame.K_END:
            self.playing = False
            self.set_frame(self.n_frames - 1)
        elif k in (pygame.K_UP, pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
            self.speed_idx = min(self.speed_idx + 1, len(SPEEDS) - 1)
        elif k in (pygame.K_DOWN, pygame.K_MINUS, pygame.K_KP_MINUS):
            self.speed_idx = max(self.speed_idx - 1, 0)
        elif k == pygame.K_c:
            self.show["coverage"] = not self.show["coverage"]
        elif k == pygame.K_f:
            self.show["footprint"] = not self.show["footprint"]
        elif k == pygame.K_v:
            self.show["voronoi"] = not self.show["voronoi"]
        elif k == pygame.K_t:
            self.show["trails"] = not self.show["trails"]
        return True

    def update(self, dt):
        if not self.playing:
            return
        self.acc += dt * SPEEDS[self.speed_idx]
        adv = int(self.acc)
        if adv:
            self.acc -= adv
            self.frame = min(self.frame + adv, self.n_frames - 1)
        if self.frame == self.n_frames - 1:
            self.playing = False

    # -------------------------------------------------------------- drawing
    @staticmethod
    def _to_px(pos):
        """(row, col) continuous position -> pixel inside the grid surface."""
        return (float((pos[1] + 0.5) * CELL), float((pos[0] + 0.5) * CELL))

    def _draw_map(self):
        f = self.frame
        ep = self.ep
        size = self.grid_n * CELL
        surf = self.terrain.copy()

        if self.show["coverage"]:
            cov = ep["coverage_map"][f]
            rgba = np.zeros((self.grid_n, self.grid_n, 4), np.uint8)
            rgba[cov] = COVER
            small = pygame.image.frombuffer(rgba.tobytes(), (self.grid_n, self.grid_n), "RGBA")
            surf.blit(pygame.transform.scale(small, (size, size)), (0, 0))

        over = pygame.Surface((size, size), pygame.SRCALPHA)
        if self.show["voronoi"]:
            over.blit(self.voronoi, (0, 0))

        # base station
        bx, by = self._to_px((1, 1))
        pygame.draw.rect(over, (235, 235, 235), (bx - 6, by - 2, 12, 9))
        pygame.draw.polygon(over, (235, 235, 235), [(bx - 8, by - 1), (bx, by - 9), (bx + 8, by - 1)])

        pos = ep["positions"][f]
        active = ep["active"][f]
        if self.show["footprint"]:
            r = int(self.meta["obs_radius"])
            for u in range(self.n_uav):
                if active[u]:
                    gr, gc = int(pos[u, 0]), int(pos[u, 1])
                    over.blit(self.fp_surf[u], ((gc - r) * CELL, (gr - r) * CELL))

        if self.show["trails"]:
            lo = max(0, f - TRAIL_LEN)
            for u in range(self.n_uav):
                pts = [self._to_px(p) for p in ep["positions"][lo:f + 1, u]]
                for k in range(1, len(pts)):
                    a = int(40 + 180 * k / len(pts))
                    pygame.draw.line(over, (*UAV_COLORS[u], a), pts[k - 1], pts[k], 2)

        self._draw_targets(over, f)
        self._draw_uavs(over, f)
        surf.blit(over, (0, 0))
        self.screen.blit(surf, GRID_ORIGIN)
        pygame.draw.rect(self.screen, (60, 72, 84), (*GRID_ORIGIN, size, size), 1)

    def _draw_targets(self, s, f):
        t = pygame.time.get_ticks() / 1000.0
        for i, kind in enumerate(self.types):
            x, y = self._to_px(self.ep["target_positions"][f, i])
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

    def _draw_uavs(self, s, f):
        ep = self.ep
        for u in range(self.n_uav):
            x, y = self._to_px(ep["positions"][f, u])
            if not ep["active"][f, u]:
                pygame.draw.circle(s, DEAD, (x, y), 6)
                pygame.draw.line(s, (30, 30, 30), (x - 4, y - 4), (x + 4, y + 4), 2)
                pygame.draw.line(s, (30, 30, 30), (x - 4, y + 4), (x + 4, y - 4), 2)
                continue
            a = self.headings[f, u]
            tip = (x + 10 * math.cos(a), y + 10 * math.sin(a))
            left = (x + 7 * math.cos(a + 2.5), y + 7 * math.sin(a + 2.5))
            right = (x + 7 * math.cos(a - 2.5), y + 7 * math.sin(a - 2.5))
            pygame.draw.polygon(s, UAV_COLORS[u], [tip, left, right])
            outline = AMBER if (ep["collided"][f, u] and f > 0) else (255, 255, 255)
            pygame.draw.polygon(s, outline, [tip, left, right], 2 if outline == AMBER else 1)

    def _text(self, txt, pos, font=None, color=TEXT):
        self.screen.blit((font or self.font).render(txt, True, color), pos)

    def _type_icon(self, kind, x, y):
        col = TYPE_COLORS[kind]
        if kind == "animal":
            pygame.draw.circle(self.screen, col, (x, y), 6)
        elif kind == "fire":
            pygame.draw.polygon(self.screen, col, [(x, y - 7), (x - 6, y + 6), (x + 6, y + 6)])
        else:
            pygame.draw.polygon(self.screen, col, [(x, y - 7), (x + 7, y), (x, y + 7), (x - 7, y)])

    def _draw_panel(self):
        f = self.frame
        ep = self.ep
        x0 = PANEL_X
        pygame.draw.rect(self.screen, PANEL_BG, (x0, 10, 540, 740), border_radius=6)
        state = "PLAYING" if self.playing else "PAUSED"
        self._text(f"Step {f} / {self.n_frames - 1}", (x0 + 10, 20), self.font_b)
        self._text(f"{state}  |  {SPEEDS[self.speed_idx]} steps/s", (x0 + 330, 20), self.font,
                   (95, 208, 138) if self.playing else AMBER)
        pr = self.progress_rect
        pygame.draw.rect(self.screen, (45, 56, 68), pr, border_radius=4)
        done = pr.copy(); done.width = int(pr.width * f / max(1, self.n_frames - 1))
        pygame.draw.rect(self.screen, (95, 208, 138), done, border_radius=4)
        for ev_f, _, col in self.events:
            if col in TYPE_COLORS.values():
                ex = pr.left + pr.width * ev_f / max(1, self.n_frames - 1)
                pygame.draw.line(self.screen, col, (ex, pr.top - 3), (ex, pr.bottom + 3), 2)

        # coverage
        cov = float(ep["coverage_rate"][f])
        y = 84
        self._text("COVERAGE", (x0 + 10, y), self.font_b, MUTED)
        centre = (x0 + 60, y + 58)
        pygame.draw.circle(self.screen, (45, 56, 68), centre, 38, 6)
        if cov > 0:
            pygame.draw.arc(self.screen, (95, 208, 138), (centre[0] - 38, centre[1] - 38, 76, 76),
                            math.pi / 2 - 2 * math.pi * cov, math.pi / 2, 6)
        txt = self.font_big.render(f"{100 * cov:.1f}%", True, TEXT)
        self.screen.blit(txt, txt.get_rect(center=centre))
        spark = pygame.Rect(x0 + 120, y + 24, 400, 70)
        pygame.draw.rect(self.screen, (27, 37, 48), spark)
        series = ep["coverage_rate"][: f + 1]
        if len(series) > 1:
            pts = [(spark.left + spark.width * k / (self.n_frames - 1), spark.bottom - spark.height * float(v))
                   for k, v in enumerate(series)]
            pygame.draw.lines(self.screen, (95, 208, 138), False, pts, 2)
        self._text("coverage vs step (full episode width)", (spark.left, spark.bottom + 2), self.font, MUTED)

        # detection
        y = 196
        nd = int(ep["n_detected"][f])
        self._text(f"DETECTED  {nd}/{len(self.types)}  ({100 * nd / len(self.types):.0f}%)",
                   (x0 + 10, y), self.font_b, MUTED)
        self._text("types are display labels only", (x0 + 330, y), self.font, MUTED)
        for row, kind in enumerate(("animal", "fire", "poi")):
            idx = [i for i, k in enumerate(self.types) if k == kind]
            got = int(ep["detected_mask"][f, idx].sum())
            yy = y + 26 + row * 22
            self._type_icon(kind, x0 + 22, yy + 8)
            self._text(f"{TYPE_LABELS[kind]}", (x0 + 38, yy))
            self._text(f"{got}/{len(idx)}", (x0 + 190, yy), self.font_b)
            for k, i in enumerate(idx):
                c = TYPE_COLORS[kind]
                cx = x0 + 240 + k * 20
                if ep["detected_mask"][f, i]:
                    pygame.draw.circle(self.screen, c, (cx, yy + 8), 6)
                else:
                    pygame.draw.circle(self.screen, c, (cx, yy + 8), 6, 1)

        # drones
        y = 300
        self._text("DRONES", (x0 + 10, y), self.font_b, MUTED)
        self._text("battery", (x0 + 110, y), self.font, MUTED)
        self._text("status", (x0 + 330, y), self.font, MUTED)
        self._text("collisions", (x0 + 440, y), self.font, MUTED)
        for u in range(self.n_uav):
            yy = y + 24 + u * 26
            pygame.draw.rect(self.screen, UAV_COLORS[u], (x0 + 12, yy + 3, 12, 12))
            self._text(f"UAV{u}", (x0 + 32, yy))
            frac = float(ep["battery_fraction"][f, u])
            bar = pygame.Rect(x0 + 110, yy + 3, 170, 12)
            pygame.draw.rect(self.screen, (45, 56, 68), bar)
            bcol = (80, 200, 120) if frac > 0.5 else AMBER if frac > 0.3 else (228, 87, 46)
            pygame.draw.rect(self.screen, bcol, (bar.left, bar.top, int(bar.width * frac), bar.height))
            self._text(f"{100 * frac:3.0f}%", (x0 + 286, yy))
            if not ep["active"][f, u]:
                st, sc = "INACTIVE", MUTED
            elif ep["collided"][f, u] and f > 0:
                st, sc = "COLLISION", AMBER
            elif frac < 0.3:
                st, sc = "LOW BATTERY", AMBER
            else:
                st, sc = "ACTIVE", (95, 208, 138)
            self._text(st, (x0 + 330, yy), self.font_b, sc)
            self._text(str(int(self.coll_so_far[f, u])), (x0 + 470, yy))

        # events
        y = 470
        self._text("EVENTS", (x0 + 10, y), self.font_b, MUTED)
        recent = [e for e in self.events if e[0] <= f][-8:]
        for k, (ef, txt, col) in enumerate(reversed(recent)):
            self._text(f"t={ef:<4d} {txt}", (x0 + 14, y + 22 + k * 19), self.font, col)

        # layers and help
        y = 650
        on = lambda k: "on" if self.show[k] else "off"
        self._text(f"Layers:  [C]overage {on('coverage')}   [F]ootprints {on('footprint')}   "
                   f"[V]oronoi {on('voronoi')}   [T]rails {on('trails')}", (x0 + 10, y), self.font, MUTED)
        self._text("Dashed lines + R0-R4: Voronoi regions, the fixed planning prior", (x0 + 10, y + 20), self.font, MUTED)
        self._text("(shown for reference; the policy is not constrained to them)", (x0 + 10, y + 38), self.font, MUTED)
        self._text("Space play/pause   <- -> step (Shift x10)   Up/Down speed   R restart   End   Esc",
                   (x0 + 10, y + 66), self.font, MUTED)

    def render(self):
        self.screen.fill(BG)
        m = self.meta
        self._text(f"UAV swarm surveillance replay  |  seed {m['seed']}  |  "
                   f"{os.path.basename(m['checkpoint'])}  |  {m['episode_length']} steps",
                   (GRID_ORIGIN[0], 14), self.font_b)
        self._draw_map()
        self._draw_panel()

    def tick(self, events=None):
        """One loop iteration. Returns False when the user quits."""
        dt = self.clock.tick(60) / 1000.0
        for e in (pygame.event.get() if events is None else events):
            if not self.handle_event(e):
                return False
        self.update(dt)
        self.render()
        pygame.display.flip()
        return True

    def run(self):
        while self.tick():
            pass
        pygame.quit()


def main():
    parser = argparse.ArgumentParser(description="Replay a recorded UAV swarm episode.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--speed", type=int, default=15, help=f"steps per second, one of {SPEEDS}")
    args = parser.parse_args()
    Dashboard(args.input, speed=args.speed).run()


if __name__ == "__main__":
    main()
