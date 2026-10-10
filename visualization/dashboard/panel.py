"""Drawing the side panel: progress, coverage, incidents, drone table and event log."""
from __future__ import annotations

import bisect
import math

import numpy as np
import pygame

from dashboard.drawing import mix, battery_colour
from dashboard.style import (PANEL_X, PANEL_W, SPEEDS, LANES, PANEL_BG, SLOT, TEXT, MUTED, GREEN, AMBER,
                             HOME, LOST, PLAN_C, TRACK_C, RADIO, INCIDENT_COLORS, RETURN, DOCKED, STRANDED,
                             TRACK, LANDED, UAV_COLORS, TYPE_COLORS, TYPE_LABELS)


class PanelView:
    """Side-panel drawing for Dashboard."""

    def _text(self, txt, pos, font=None, color=TEXT):
        self.screen.blit((font or self.font).render(txt, True, color), pos)

    def _pill(self, txt, x, y, color):
        img = self.font_s.render(txt, True, (16, 20, 26))
        w, h = img.get_size()
        pygame.draw.rect(self.screen, color, (x, y, w + 10, h), border_radius=h // 2)
        self.screen.blit(img, (x + 5, y))

    def _type_icon(self, kind, x, y):
        col = TYPE_COLORS.get(kind) or INCIDENT_COLORS[kind]
        if kind == "animal":
            pygame.draw.circle(self.screen, col, (x, y), 6)
        elif kind == "fire":
            pygame.draw.polygon(self.screen, col, [(x, y - 7), (x - 6, y + 6), (x + 6, y + 6)])
        elif kind == "intruder":
            self.screen.blit(self.intruder_icon[True], (x - 10, y - 12))
        else:
            pygame.draw.polygon(self.screen, col, [(x, y - 7), (x + 7, y), (x, y + 7), (x - 7, y)])

    def _draw_progress(self, f):
        pr = self.progress_rect
        span = max(1, self.n_frames - 1)
        pygame.draw.rect(self.screen, SLOT, pr, border_radius=4)
        done = pr.copy()
        done.width = int(pr.width * f / span)
        pygame.draw.rect(self.screen, GREEN, done, border_radius=4)
        for ev_f, _, col, mark in self.events:
            if mark:
                ex = pr.left + pr.width * ev_f / span
                pygame.draw.line(self.screen, col, (ex, pr.top - 3), (ex, pr.bottom + 3), 2)

    def _draw_timeline(self, f):
        """Each incident as a bar: dim while undetected, bright once detected, dot when confirmed."""
        r = self.timeline_rect
        span = max(1, self.n_frames - 1)
        X = lambda fr: r.left + r.width * fr / span
        pygame.draw.rect(self.screen, (27, 37, 48), r)
        lane_h = r.height / LANES
        for k, lane in enumerate(self.lanes):
            s, d, c = int(self.inc_spawn[k]), int(self.inc_det[k]), int(self.inc_conf[k])
            if s > f:
                continue
            y = r.top + lane * lane_h + lane_h / 2
            col = INCIDENT_COLORS[self.inc_kind[k]]
            seen = 0 <= d <= f
            end = d if seen else f
            pygame.draw.line(self.screen, mix(col, PANEL_BG, 0.55), (X(s), y), (max(X(end), X(s) + 1), y), 3)
            if seen:
                end2 = c if 0 <= c <= f else f
                pygame.draw.line(self.screen, col, (X(d), y), (max(X(end2), X(d) + 1), y), 3)
                if 0 <= c <= f:
                    pygame.draw.circle(self.screen, (255, 255, 255), (X(c), y), 2)
        pygame.draw.line(self.screen, TEXT, (X(f), r.top - 2), (X(f), r.bottom + 2), 1)

    def _draw_rings(self, f, y):
        x0, ep = PANEL_X, self.ep
        cov = float(ep["coverage_rate"][f])
        if self.patrol:
            val, col = float(self.fresh_series[f]), RADIO
            self._text(f"FOREST SEEN IN THE LAST {self.window} STEPS", (x0 + 10, y), self.font_b, MUTED)
        else:
            val, col = cov, GREEN
            self._text("COVERAGE", (x0 + 10, y), self.font_b, MUTED)
        centre = (x0 + 60, y + 60)
        pygame.draw.circle(self.screen, SLOT, centre, 38, 6)
        if val > 0:
            pygame.draw.arc(self.screen, col, (centre[0] - 38, centre[1] - 38, 76, 76),
                            math.pi / 2 - 2 * math.pi * val, math.pi / 2, 6)
        txt = self.font_ring.render(f"{100 * val:.1f}%" if val < 0.9995 else "100%", True, TEXT)
        self.screen.blit(txt, txt.get_rect(center=centre))
        spark = pygame.Rect(x0 + 120, y + 26, 400, 64)
        pygame.draw.rect(self.screen, (27, 37, 48), spark)
        plot = spark.inflate(0, -8)
        span = max(1, self.n_frames - 1)
        series = [(ep["coverage_rate"], GREEN)] + ([(self.fresh_series, RADIO)] if self.patrol else [])
        for data, c in series:
            if f > 0:
                xs = plot.left + plot.width * np.arange(f + 1) / span
                ys = plot.bottom - plot.height * np.asarray(data[: f + 1], dtype=np.float64)
                pygame.draw.lines(self.screen, c, False, np.column_stack([xs, ys]).tolist(), 2)
        if self.patrol:
            self._text(f"green: ever seen ({100 * cov:.0f}%)    blue: seen in the last {self.window} steps",
                       (spark.left, spark.bottom + 2), self.font_s, MUTED)
        else:
            self._text("coverage vs step (full episode width)", (spark.left, spark.bottom + 2), self.font_s, MUTED)
        return y + 112

    def _draw_target_rows(self, f, y):
        x0, ep = PANEL_X, self.ep
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
                pygame.draw.circle(self.screen, c, (cx, yy + 8), 6, 0 if ep["detected_mask"][f, i] else 1)
        return y + 98

    def _draw_target_line(self, f, y):
        x0, ep = PANEL_X, self.ep
        nd = int(ep["n_detected"][f])
        self._text(f"FIXED TARGETS  {nd}/{len(self.types)}", (x0 + 10, y), self.font_b, MUTED)
        x = x0 + 185
        for kind in ("animal", "fire", "poi"):
            idx = [i for i, k in enumerate(self.types) if k == kind]
            got = int(ep["detected_mask"][f, idx].sum())
            self._type_icon(kind, x, y + 9)
            self._text(f"{TYPE_LABELS[kind].split()[0].lower()} {got}/{len(idx)}", (x + 12, y + 1))
            x += 112
        return y + 26

    def _incident_rows(self, f):
        """Incidents worth listing now: being handled, still hidden, then most recently resolved."""
        ep = self.ep
        live = [k for k in range(len(self.inc_kind)) if 0 <= self.inc_spawn[k] <= f]
        tracked = {int(k): u for u, k in enumerate(ep["track_event"][f]) if k >= 0}
        hidden = [k for k in live if not 0 <= self.inc_det[k] <= f]
        done = [k for k in live if k not in tracked and k not in hidden]
        done.sort(key=lambda k: -max(int(self.inc_det[k]), int(self.inc_conf[k]) if self.inc_conf[k] <= f else -1))
        return [(k, tracked[k]) for k in sorted(tracked)] + [(k, None) for k in hidden] + [(k, None) for k in done]

    def _draw_incident_list(self, f, y):
        x0 = PANEL_X
        started = int(((self.inc_spawn >= 0) & (self.inc_spawn <= f)).sum())
        found = [k for k in range(len(self.inc_kind)) if 0 <= self.inc_det[k] <= f]
        self._text("INCIDENTS", (x0 + 10, y), self.font_b, MUTED)
        if found:
            delays = [int(self.inc_det[k] - self.inc_spawn[k]) for k in found]
            summary = (f"{started} started · {len(found)} detected · time to detect: "
                       f"mean {np.mean(delays):.0f}, worst {max(delays)} steps")
        else:
            summary = f"{started} started · none detected yet"
        self._text(summary, (x0 + 100, y + 1), self.font_s, MUTED)
        rows = self._incident_rows(f)[:3]
        if not rows:
            self._text("No fires or intruders yet", (x0 + 14, y + 22), self.font, MUTED)
        for j, (k, u) in enumerate(rows):
            yy = y + 22 + j * 19
            kind = self.inc_kind[k]
            col = INCIDENT_COLORS[kind]
            p = self.ep["event_pos"][f, k]
            self._type_icon(kind, x0 + 22, yy + 8)
            self._text(f"#{k} {kind.upper()} ({int(p[0])},{int(p[1])})", (x0 + 38, yy), self.font, col)
            s, d, c = int(self.inc_spawn[k]), int(self.inc_det[k]), int(self.inc_conf[k])
            if not 0 <= d <= f:
                st, sc = f"undetected for {f - s} steps (swarm unaware)", LOST
            elif u is not None and not 0 <= c <= f:
                st, sc = f"found after {d - s} steps · UAV{u} on the way to confirm", TRACK_C
            elif u is not None:
                st, sc = f"confirmed by UAV{u} · watching it", TRACK_C
            else:
                st, sc = f"found after {d - s} steps · confirmed and reported", MUTED
            self._text(st, (x0 + 200, yy + 1), self.font_s, sc)
        return y + 22 + 3 * 19 + 6

    def _flown_label(self, f, u):
        ep = self.ep
        if "flown_by" in ep:
            label, _, col = self.flown_style.get(self._in_control(f, u), ("?", "", MUTED))
            return label, col
        if not self.mission:
            return "POLICY", GREEN
        return {RETURN: ("RETURN", HOME), DOCKED: ("DOCKED", MUTED),
                STRANDED: ("LOST", LOST)}.get(int(ep["mode"][f, u]), ("EXPLORE", MUTED))

    def _status(self, f, u):
        ep = self.ep
        frac = float(ep["battery_fraction"][f, u])
        m = int(ep["mode"][f, u]) if self.mission else None
        if m == STRANDED:
            return "LOST", LOST
        if m == LANDED:
            return "LANDED OUT", AMBER
        if m == DOCKED:
            if frac >= 0.999:
                return "DOCKED", MUTED
            return ("SWAPPING" if "packs" in ep else "CHARGING"), HOME
        if m == RETURN:
            return "RETURNING", HOME
        if m == TRACK:
            k = int(ep["track_event"][f, u])
            return (f"WATCHING #{k}" if 0 <= self.inc_conf[k] <= f else f"CONFIRMING #{k}"), TRACK_C
        if not ep["active"][f, u]:
            return "INACTIVE", MUTED
        if ep["collided"][f, u] and f > 0:
            return "COLLISION", AMBER
        nxt = min(f + 1, self.n_frames - 1)           # the plan being flown now, as in _in_control
        if self.has_plans and ep["targets"][nxt, u, 0] >= 0:
            r, c = ep["targets"][nxt, u]
            return f"TO ({r},{c})", PLAN_C
        if frac < 0.3 and not self.mission:
            return "LOW BATTERY", AMBER
        return ("PATROLLING" if self.patrol else "EXPLORING" if self.mission else "ACTIVE"), GREEN

    def _draw_drone_table(self, f, y):
        x0, ep = PANEL_X, self.ep
        self._text("DRONES", (x0 + 10, y), self.font_b, MUTED)
        self._text("battery", (x0 + 80, y + 2), self.font_s, MUTED)
        self._text("flown by", (x0 + 236, y + 2), self.font_s, MUTED)
        self._text("status", (x0 + 318, y + 2), self.font_s, MUTED)
        self._text("safety fixes" if self.has_fixes else "collisions", (x0 + 455, y + 2), self.font_s, MUTED)
        for u in range(self.n_uav):
            yy = y + 22 + u * 24
            pygame.draw.rect(self.screen, UAV_COLORS[u], (x0 + 12, yy + 3, 12, 12), border_radius=2)
            self._text(f"UAV{u}", (x0 + 30, yy))
            frac = float(ep["battery_fraction"][f, u])
            bar = pygame.Rect(x0 + 80, yy + 4, 110, 11)
            pygame.draw.rect(self.screen, SLOT, bar)
            pygame.draw.rect(self.screen, battery_colour(frac), (bar.left, bar.top, int(bar.width * frac), bar.height))
            self._text(f"{100 * frac:3.0f}%", (x0 + 196, yy))
            label, col = self._flown_label(f, u)
            self._pill(label, x0 + 236, yy + 2, col)
            st, sc = self._status(f, u)
            self._text(st, (x0 + 318, yy), self.font_b, sc)
            n = self.fixes_so_far[f, u] if self.has_fixes else self.coll_so_far[f, u]
            self._text(str(int(n)), (x0 + 480, yy))
        return y + 22 + self.n_uav * 24 + 4

    def _draw_fleet_line(self, f, y):
        x0 = PANEL_X
        x = x0 + 10
        if "packs" in self.ep:
            self._text("Spare batteries", (x, y), self.font_s, MUTED)
            for j, p in enumerate(self.ep["packs"][f]):
                frac = float(p) / self.max_battery
                bx = x0 + 100 + j * 36
                pygame.draw.rect(self.screen, MUTED, (bx, y + 3, 26, 12), 1, border_radius=2)
                pygame.draw.rect(self.screen, MUTED, (bx + 26, y + 6, 2, 6))
                pygame.draw.rect(self.screen, battery_colour(frac), (bx + 2, y + 5, int(22 * frac), 8))
            x = x0 + 100 + len(self.ep["packs"][f]) * 36 + 14
        links, reach, airborne = self.radio
        R = int(self.meta["comm_radius"])
        txt = (f"Radio ({R} cells): {links} drone link{'s' if links != 1 else ''} · "
               f"{reach}/{airborne} flying reach the base" if airborne else "Radio: no drones in the air")
        self._text(txt, (x, y + 1), self.font_s, RADIO if airborne and reach == airborne else MUTED)
        return y + 24

    def _draw_log(self, f, y, bottom):
        self._text("EVENT LOG", (PANEL_X + 10, y), self.font_b, MUTED)
        n = max(0, (bottom - y - 22) // 18)
        upto = bisect.bisect_right(self.event_frames, f)
        recent = self.events[max(0, upto - n):upto]
        for k, (ef, txt, col, _) in enumerate(reversed(recent)):
            self._text(f"t={ef:<4d} {txt}", (PANEL_X + 14, y + 22 + k * 18), self.font, col)

    def _draw_footer(self, y):
        def on(k):
            return "n/a" if not self.available.get(k, True) else "on" if self.show[k] else "off"
        lines = [
            f"[C]overage {on('coverage')}   [G] fog {on('fog')}   [F]ootprints {on('footprint')}   "
            f"[V]oronoi {on('voronoi')}   [T]rails {on('trails')}",
            f"[P] routes {on('routes')}   [L] radio {on('radio')}   [S]afety {on('safety')}   "
            f"[B]adges {on('badges')}   [E]vents {on('events')}",
            "Badges: AI = trained policy   PL = planner   RTH = return home   TRK = confirming an event",
            "Space play/pause   <- -> step (Shift x10)   Up/Down speed   R restart   F12 screenshot",
        ]
        for k, txt in enumerate(lines):
            self._text(txt, (PANEL_X + 10, y + k * 19), self.font_s, MUTED)

    def _draw_panel(self):
        f = self.frame
        x0 = PANEL_X
        pygame.draw.rect(self.screen, PANEL_BG, (x0, 10, PANEL_W, 740), border_radius=6)
        state = "PLAYING" if self.playing else "PAUSED"
        self._text(f"Step {f} / {self.n_frames - 1}", (x0 + 10, 18), self.font_b)
        self._text(f"{state}  |  {SPEEDS[self.speed_idx]} steps/s", (x0 + 330, 18), self.font,
                   GREEN if self.playing else AMBER)
        self._draw_progress(f)
        y = 64
        if self.has_incidents:
            self._draw_timeline(f)
            self._text("timeline: dim = undetected, bright = detected, dot = confirmed  ·  map: reported ones fade",
                       (x0 + 10, 87), self.font_s, MUTED)
            y = 108
        y = self._draw_rings(f, y)
        if self.has_incidents:
            y = self._draw_target_line(f, y)
            y = self._draw_incident_list(f, y)
        else:
            y = self._draw_target_rows(f, y)
        y = self._draw_drone_table(f, y)
        y = self._draw_fleet_line(f, y)
        self._draw_log(f, y, 662)
        self._draw_footer(668)
