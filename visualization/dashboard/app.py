"""The replay window: state, controls and the render loop."""
from __future__ import annotations

import math
import os

import numpy as np
import pygame

from dashboard.episode import build_events, last_seen_history, load_episode
from dashboard.map_view import MapView
from dashboard.panel import PanelView
from dashboard.style import (BG, CELL, DOCKED, FLOWN_STYLE, GRID_ORIGIN, LANES, PANEL_X, SPEEDS, UAV_COLORS,
                             WINDOW)


class Dashboard(MapView, PanelView):
    """Replays a recorded episode: map on the left, status panel on the right."""

    def __init__(self, path, speed=15, screen=None):
        self.path = path
        self._load(path)
        self._open_window(screen)
        self._reset_controls(speed)
        self._precompute()
        self._build_surfaces()

    # ----------------------------------------------------------------- setup
    def _load(self, path):
        self.ep = ep = load_episode(path)
        self.meta = self.ep["meta"]
        self.n_frames = len(ep["timestep"])
        self.grid_n = int(self.meta["grid_size"])
        self.n_uav = int(self.meta["n_agents"])
        self.types = [str(t) for t in ep["target_types"]]
        self.max_battery = float(self.meta["max_battery"])
        self.mission = "mode" in ep
        self.mcfg = self.meta.get("mission", {})
        self.patrol = bool(self.mcfg.get("patrol"))
        self.safety = bool(self.mcfg.get("safety"))
        self.has_plans = "routes" in ep
        self.has_fixes = self.safety and "corrected" in ep         # older recordings lack it
        self.has_incidents = "event_kind" in ep and len(ep["event_kind"]) > 0
        self.pads = [tuple(p) for p in self.meta.get("pads", [])]
        self.window = int(self.meta.get("fresh_window", 100))
        codes = self.meta.get("flown_by_codes", {})
        self.flown_style = {code: FLOWN_STYLE[name] for name, code in codes.items() if name in FLOWN_STYLE}

    def _open_window(self, screen):
        pygame.init()
        self.screen = screen or pygame.display.set_mode(WINDOW)
        pygame.display.set_caption(f"UAV Swarm Replay - seed {self.meta['seed']}")
        self.font = pygame.font.SysFont("segoeui,arial", 14)
        self.font_s = pygame.font.SysFont("segoeui,arial", 12)
        self.font_xs = pygame.font.SysFont("segoeui,arial", 10, bold=True)
        self.font_b = pygame.font.SysFont("segoeui,arial", 15, bold=True)
        self.font_big = pygame.font.SysFont("segoeui,arial", 30, bold=True)
        self.font_ring = pygame.font.SysFont("segoeui,arial", 22, bold=True)
        self.clock = pygame.time.Clock()

    def _reset_controls(self, speed):
        self.frame = 0
        self.playing = True
        self.speed_idx = SPEEDS.index(speed) if speed in SPEEDS else SPEEDS.index(15)
        self.acc = 0.0
        self.anim = 0.0
        self.show = {"coverage": not self.patrol, "fog": self.patrol, "footprint": True,
                     "voronoi": not self.mission, "trails": True, "routes": True, "radio": True,
                     "safety": True, "badges": True, "events": True}
        self.available = {"routes": self.has_plans, "safety": self.safety, "events": self.has_incidents,
                          "badges": self.mission}

    def _precompute(self):
        """Per-frame data derived once from the recording."""
        ep = self.ep
        self.events = build_events(ep)
        self.event_frames = [e[0] for e in self.events]
        hits = ep["collided"] & ep["active"]
        hits[0] = False
        self.coll_so_far = np.cumsum(hits, axis=0)
        if "corrected" in ep:
            c = ep["corrected"]
            self.fixes_so_far = np.cumsum(c, axis=0)
            self.last_fix = np.maximum.accumulate(np.where(c, np.arange(len(c))[:, None], -1), axis=0)
        noise = float(self.mcfg.get("position_noise", 0.0))
        self.separation = float(self.mcfg.get("separation", 1.0)) + 4.0 * noise   # as SafetySupervisor
        self.headings = self._headings()

        r = int(self.meta["obs_radius"])
        self.last_seen = ep["last_seen"] if "last_seen" in ep else last_seen_history(ep, self.grid_n, r)
        t = np.arange(self.n_frames)[:, None, None]
        fresh = (self.last_seen >= 0) & (t - self.last_seen <= self.window)
        self.fresh_series = fresh[:, ~ep["obstacle_grid"]].mean(axis=1)
        self._ground_cache = (None, None)
        self.radio = (0, 0, 0)

        if self.has_incidents:
            self.inc_kind = [str(k) for k in ep["event_kind"]]
            self.inc_spawn = ep["event_spawn"]
            self.inc_det = ep["event_detected"]
            self.inc_conf = ep["event_confirmed"]
            self.lanes = self._incident_lanes()
            rng = np.random.default_rng(1)
            self.flames = [[(rng.uniform(0, 2 * math.pi), math.sqrt(rng.random()), rng.uniform(0, 6))
                            for _ in range(10)] for _ in self.inc_kind]

    def _build_surfaces(self):
        # intruder icons: detected / not yet detected / already reported
        self.intruder_icon = {True: self._intruder_surface(1.0), False: self._intruder_surface(0.45),
                              None: self._intruder_surface(0.3)}

        # converted to the display's pixel format, which makes blits several times faster
        self.terrain = self._terrain_surface().convert()
        self.voronoi = self._voronoi_surface().convert_alpha()
        self.fp_surf = [self._footprint_surface(c).convert_alpha() for c in UAV_COLORS]
        size = self.grid_n * CELL
        self.overlay = pygame.Surface((size, size), pygame.SRCALPHA).convert_alpha()
        self.panel_area = pygame.Rect(PANEL_X, 0, WINDOW[0] - PANEL_X, WINDOW[1])
        self._panel_key, self._panel_cache = None, None
        self.progress_rect = pygame.Rect(PANEL_X + 10, 44, 520, 10)
        self.timeline_rect = pygame.Rect(PANEL_X + 10, 60, 520, 24)

    def _headings(self):
        vel = self.ep["velocities"]
        h = np.zeros(vel.shape[:2])
        last = np.full(vel.shape[1], math.pi / 4)
        for f in range(vel.shape[0]):
            moving = np.linalg.norm(vel[f], axis=1) > 1e-6
            last[moving] = np.arctan2(vel[f, moving, 0], vel[f, moving, 1])
            h[f] = last
        return h

    def _incident_lanes(self):
        """Spread the incidents over a few timeline rows so their bars rarely overlap."""
        last = self.n_frames - 1
        ends, lanes = [-10 ** 9] * LANES, []
        for k in range(len(self.inc_kind)):
            s = int(self.inc_spawn[k])
            e = max(int(self.inc_conf[k]), int(self.inc_det[k]))
            e = e if e >= 0 else last
            free = [ln for ln in range(LANES) if ends[ln] < s - 8]
            ln = free[0] if free else int(np.argmin(ends))
            ends[ln] = max(ends[ln], e)
            lanes.append(ln)
        return lanes

    # -------------------------------------------------------------- controls
    def set_frame(self, f):
        self.frame = int(np.clip(f, 0, self.n_frames - 1))
        self.acc = 0.0

    def _seek(self, x, rect):
        frac = (x - rect.left) / rect.width
        self.set_frame(round(frac * (self.n_frames - 1)))

    def handle_event(self, e):
        if e.type == pygame.QUIT:
            return False
        if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            for rect in (self.progress_rect, self.timeline_rect if self.has_incidents else None):
                if rect is not None and rect.inflate(0, 12).collidepoint(e.pos):
                    self._seek(e.pos[0], rect)
        if e.type != pygame.KEYDOWN:
            return True
        step = 10 if e.mod & pygame.KMOD_SHIFT else 1
        k = e.key
        layers = {pygame.K_c: "coverage", pygame.K_g: "fog", pygame.K_f: "footprint", pygame.K_v: "voronoi",
                  pygame.K_t: "trails", pygame.K_p: "routes", pygame.K_l: "radio", pygame.K_s: "safety",
                  pygame.K_b: "badges", pygame.K_e: "events"}
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
        elif k in layers:
            self.show[layers[k]] = not self.show[layers[k]]
        elif k == pygame.K_F12:
            out = f"{os.path.splitext(self.path)[0]}_frame{self.frame}.png"
            pygame.image.save(self.screen, out)
            print(f"Saved {out}")
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
            self.acc = 0.0

    # --------------------------------------------------------- interpolation
    def _frac(self):
        """How far the replay is between this frame and the next (0 when paused)."""
        return min(self.acc, 0.999) if self.playing and self.frame < self.n_frames - 1 else 0.0

    def _lerp(self, arr):
        f, a = self.frame, self._frac()
        if a <= 0:
            return arr[f]
        return arr[f] + (arr[f + 1] - arr[f]) * a

    def _heading(self, u):
        f, a = self.frame, self._frac()
        h0 = self.headings[f, u]
        if a <= 0:
            return h0
        d = (self.headings[f + 1, u] - h0 + math.pi) % (2 * math.pi) - math.pi
        return h0 + d * min(1.0, 3 * a)

    @staticmethod
    def _pos_px(p):
        """Continuous (row, col) position -> pixel; cell a spans [a, a + 1)."""
        return (float(p[1]) * CELL, float(p[0]) * CELL)

    @staticmethod
    def _cell_px(r, c):
        """Centre of grid cell (r, c) in pixels."""
        return ((float(c) + 0.5) * CELL, (float(r) + 0.5) * CELL)

    def _flying(self, f):
        flying = self.ep["active"][f].copy()
        if self.mission:
            flying &= self.ep["mode"][f] != DOCKED
        return flying

    def _in_control(self, f, u):
        """Who flies the drone's next move. flown_by[t] records who flew the move into frame t,
        so the next frame's entry is who is in control at frame f."""
        return int(self.ep["flown_by"][min(f + 1, self.n_frames - 1), u])

    # -------------------------------------------------------------- render
    def _title(self):
        m = self.meta
        if not self.mission:
            kind = "trained policy only"
        elif self.patrol:
            kind = "persistent patrol" + (" + fires & intruders" if self.has_incidents else "")
        else:
            kind = "mission controller" + (" + safety layer" if self.safety else "") + \
                   (" + smart planner" if self.mcfg.get("gain_targets") else "")
        return f"UAV swarm replay  |  seed {m['seed']}  |  {m['episode_length']} steps  |  {kind}"

    def render(self):
        self.screen.fill(BG, (0, 0, PANEL_X, WINDOW[1]))
        self._text(self._title(), (GRID_ORIGIN[0], 14), self.font_b)
        self._draw_map()
        # the panel only changes with the frame, so it is redrawn once per step, not every display frame
        key = (self.frame, self.playing, self.speed_idx, tuple(self.show.values()), self.radio)
        if key != self._panel_key:
            self.screen.fill(BG, self.panel_area)
            self._draw_panel()
            self._panel_cache = self.screen.subsurface(self.panel_area).copy()
            self._panel_key = key
        else:
            self.screen.blit(self._panel_cache, self.panel_area)

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
