"""Interactive replay of an episode recorded by visualization/record_episode.py.

Usage:
    python visualization/pygame_dashboard.py --input <episode.npz> [--speed FPS] [--frame N]
    python visualization/pygame_dashboard.py --input <episode.npz> --screenshot out.png --frame N

Replays the .npz only. It never imports or runs the environment or MADDPG.
Frame 0 is the reset state; frame t is the state after step t. Between two
recorded frames the drones are drawn part-way along their move, so playback
stays smooth at any speed.

Mission recordings (--mission, --patrol) also show what the mission layer was
thinking: who flew each drone (the trained policy, the planner, return-to-home
or event tracking), its planned route and target, the safety layer's
corrections, the radio links between drones, how long ago every part of the
forest was last seen, and the fires and intruders the swarm responded to.
Older recordings replay with the layers they have data for.

Controls
    Space          play / pause
    Right / Left   step one frame forward / back (pauses)
    Shift+arrow    step 10 frames
    Up / Down, + - replay speed
    R / Home       restart from frame 0
    End            jump to last frame
    C G F V T      coverage / fog of war (time since seen) / sensor footprints / Voronoi prior / trails
    P L S B E      planned routes / radio links / safety layer / flown-by badges / fires and intruders
    F12            save a screenshot next to the recording
    Click the progress bar or the incident timeline to seek.   Esc or Q quits.
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import os

import numpy as np
import pygame

WINDOW = (1280, 760)
CELL = 14
GRID_ORIGIN = (10, 44)
PANEL_X = 730
PANEL_W = 540
TRAIL_LEN = 60
SPEEDS = [1, 2, 5, 10, 15, 30, 60, 120]
FLASH_FRAMES = 12          # a safety correction stays highlighted this many frames
PIN_FRAMES = 80            # a detection alert keeps its label this many frames
LANES = 4                  # rows of the incident timeline

BG = (13, 19, 26)
PANEL_BG = (19, 27, 36)
SLOT = (45, 56, 68)
TEXT = (230, 237, 243)
MUTED = (139, 152, 165)
GROUND = (26, 54, 34)
GROUND_DRY = (58, 70, 38)
CANOPY = [(30, 78, 42), (36, 90, 48), (42, 100, 52), (27, 70, 39)]
COVER = (95, 208, 138, 80)
FOG = (6, 10, 16)
VORONOI = (255, 255, 255, 80)
GREEN = (95, 208, 138)
AMBER = (255, 191, 0)
DEAD = (110, 116, 124)
HOME = (120, 200, 255)
LOST = (228, 87, 46)
PLAN_C = (255, 214, 64)
TRACK_C = (255, 92, 138)
RADIO = (120, 220, 255)
SAFETY = (255, 214, 64)
FIRE_C = (255, 120, 40)
INTRUDER_C = (200, 110, 255)
INCIDENT_COLORS = {"fire": FIRE_C, "intruder": INTRUDER_C}
# mission mode codes written by record_episode.py
EXPLORE, RETURN, DOCKED, STRANDED, TRACK = 0, 1, 2, 3, 4
# who flew a drone, by the names record_episode.py stores in metadata["flown_by_codes"]
FLOWN_STYLE = {"policy": ("POLICY", "AI", GREEN), "planner": ("PLANNER", "PL", PLAN_C),
               "return": ("RETURN", "RTH", HOME), "docked": ("DOCKED", "", MUTED),
               "stranded": ("LOST", "", LOST), "track": ("TRACKING", "TRK", TRACK_C)}
UAV_COLORS = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189)]
TYPE_COLORS = {"animal": (74, 163, 255), "fire": (228, 87, 46), "poi": (175, 183, 191)}
TYPE_LABELS = {"animal": "Animals", "fire": "Fire", "poi": "Points of interest"}


def load_episode(path):
    with np.load(path, allow_pickle=False) as z:
        ep = {k: z[k] for k in z.files}
    ep["meta"] = json.loads(str(ep.pop("metadata")))
    return ep


def footprint_offsets(r):
    d = np.arange(-r, r + 1)
    dx, dy = np.meshgrid(d, d, indexing="ij")
    m = dx ** 2 + dy ** 2 <= r ** 2
    return dx[m], dy[m]


def last_seen_history(ep, n, r):
    """Step at which every cell was last inside a flying UAV's sensor footprint (-1: never).

    Patrol recordings store this; for older recordings it is rebuilt from the
    positions with the same circular footprint around the UAV's cell as the environment.
    """
    T = len(ep["timestep"])
    flying = ep["active"].copy()
    if "mode" in ep:
        flying &= ep["mode"] != DOCKED
    dx, dy = footprint_offsets(r)
    cur = np.full((n, n), -1, dtype=np.int16)
    out = np.empty((T, n, n), dtype=np.int16)
    for t in range(T):
        for u in np.where(flying[t])[0]:
            gx, gy = np.clip(ep["positions"][t, u], 0, n - 1).astype(int)
            xs, ys = gx + dx, gy + dy
            ok = (xs >= 0) & (xs < n) & (ys >= 0) & (ys < n)
            cur[xs[ok], ys[ok]] = t
        out[t] = cur
    return out


def build_events(ep):
    """Event log derived from the recorded arrays: (frame, text, colour, mark on the progress bar)."""
    n_uav = ep["active"].shape[1]
    ev = [(0, f"Start: {n_uav} UAVs at base (1,1)", TEXT, False)]
    types = [str(t) for t in ep["target_types"]]
    det = ep["detected_mask"]
    for i in np.where(det.any(axis=0))[0]:
        f = int(det[:, i].argmax())
        ev.append((f, f"{types[i].upper()} #{i} detected", TYPE_COLORS[types[i]], True))
    hits = ep["collided"] & ep["active"]
    hits[0] = False
    for f, u in zip(*np.where(hits)):
        ev.append((int(f), f"UAV{u} collision (blocked by obstacle)", AMBER, False))
    kinds = [str(k) for k in ep.get("event_kind", [])]
    if "mode" in ep:                                 # mission recording: log every mode change
        mode = ep["mode"]
        swaps = "packs" in ep
        for f in range(1, len(mode)):
            for u in np.where(mode[f] != mode[f - 1])[0]:
                new, old = int(mode[f, u]), int(mode[f - 1, u])
                if new == EXPLORE:
                    txt, col = ("launched", GREEN) if old == DOCKED else ("back on patrol", GREEN)
                elif new == RETURN:
                    txt, col = "returning to base (battery)", HOME
                elif new == DOCKED:
                    txt, col = ("landed for a battery swap", MUTED) if swaps else ("docked, recharging", MUTED)
                elif new == STRANDED:
                    txt, col = "LOST (battery empty in the field)", LOST
                else:
                    k = int(ep["track_event"][f, u])
                    txt, col = f"sent to confirm {kinds[k].upper()} #{k}", TRACK_C
                ev.append((f, f"UAV{u} {txt}", col, False))
    else:
        for u in range(n_uav):
            low = np.where(ep["battery_fraction"][:, u] < 0.30)[0]
            if len(low):
                ev.append((int(low[0]), f"UAV{u} battery low (<30%)", AMBER, False))
            off = np.where(~ep["active"][:, u])[0]
            if len(off):
                ev.append((int(off[0]), f"UAV{u} inactive (battery depleted)", MUTED, False))
    if "corrected" in ep and ep["meta"].get("mission", {}).get("safety"):
        for f, u in zip(*np.where(ep["corrected"])):
            ev.append((int(f), f"Safety layer changed UAV{u}'s move", SAFETY, False))
    for k, kind in enumerate(kinds):
        s, d, c = (int(ep[key][k]) for key in ("event_spawn", "event_detected", "event_confirmed"))
        col = INCIDENT_COLORS.get(kind, TEXT)
        if d >= 0:
            r, cc = ep["event_pos"][d, k].astype(int)
            ev.append((d, f"{kind.upper()} #{k} detected at ({r},{cc}), {d - s} steps after it began", col, True))
        if c >= 0:
            who = int(ep["event_tracker"][c, k])
            by = f" by UAV{who}" if who >= 0 else ""
            ev.append((c, f"{kind.upper()} #{k} confirmed{by}, {c - d} steps after detection", col, False))
    cov = ep["coverage_rate"]
    for m in (0.25, 0.50, 0.75, 0.90, 0.95, 1.0):
        hit = np.where(cov >= m - 1e-6)[0]
        if len(hit):
            ev.append((int(hit[0]), f"Coverage reached {int(m * 100)}%", GREEN, False))
    last = len(ep["timestep"]) - 1
    ev.append((last, f"Episode end at step {last}", TEXT, False))
    ev.sort(key=lambda e: e[0])
    return ev


# ------------------------------------------------------------------ drawing helpers
def mix(c1, c2, t):
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))


def battery_colour(frac):
    return (80, 200, 120) if frac > 0.5 else AMBER if frac > 0.3 else LOST


def smooth_noise(rng, coarse, size):
    """Value noise: a coarse random lattice smoothly interpolated up to size x size."""
    g = rng.random((coarse + 1, coarse + 1)).astype(np.float32)
    x = np.linspace(0, coarse, size, endpoint=False)
    i = x.astype(int)
    t = x - i
    t = t * t * (3 - 2 * t)
    tr, tc = t[:, None], t[None, :]
    a, b = g[i][:, i], g[i + 1][:, i]
    c, d = g[i][:, i + 1], g[i + 1][:, i + 1]
    return (a * (1 - tr) + b * tr) * (1 - tc) + (c * (1 - tr) + d * tr) * tc


def blend_circle(dst, rgba, center, radius, width=0):
    """Translucent circle blended onto dst (pygame.draw writes alpha instead of blending)."""
    if radius <= 0 or rgba[3] <= 0:
        return
    r = int(math.ceil(radius)) + 2
    tmp = pygame.Surface((2 * r, 2 * r), pygame.SRCALPHA)
    pygame.draw.circle(tmp, rgba, (r, r), radius, width)
    dst.blit(tmp, (center[0] - r, center[1] - r))


def dashed_line(surf, color, pts, width=2, dash=7.0, gap=4.0, phase=0.0):
    """Polyline drawn as dashes; a growing `phase` makes the dashes march towards the end."""
    period = dash + gap
    walked = -phase
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        seg = math.hypot(x1 - x0, y1 - y0)
        if seg < 1e-6:
            continue
        t = 0.0
        while t < seg - 1e-3:
            at = (walked + t) % period
            if at < dash:
                run = min(dash - at, seg - t)
                a, b = t / seg, (t + run) / seg
                pygame.draw.line(surf, color, (x0 + (x1 - x0) * a, y0 + (y1 - y0) * a),
                                 (x0 + (x1 - x0) * b, y0 + (y1 - y0) * b), width)
            else:
                run = period - at
            t += max(run, 1e-2)                      # always advance (float rounding can give ~0)
        walked += seg


def arrow(surf, color, start, vec, scale, width=2, dashed=False):
    """Arrow from start along a (row, col) vector, scaled to pixels."""
    dx, dy = float(vec[1]) * scale, float(vec[0]) * scale
    if math.hypot(dx, dy) < 3:
        return
    end = (start[0] + dx, start[1] + dy)
    if dashed:
        dashed_line(surf, color, [start, end], width, 4, 3)
    else:
        pygame.draw.line(surf, color, start, end, width)
    a = math.atan2(dy, dx)
    pygame.draw.polygon(surf, color, [end, (end[0] - 7 * math.cos(a - 0.45), end[1] - 7 * math.sin(a - 0.45)),
                                      (end[0] - 7 * math.cos(a + 0.45), end[1] - 7 * math.sin(a + 0.45))])


class Dashboard:
    def __init__(self, path, speed=15, screen=None):
        self.path = path
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
        # detected / not yet detected / already reported
        self.intruder_icon = {True: self._intruder_surface(1.0), False: self._intruder_surface(0.45),
                              None: self._intruder_surface(0.3)}

        # converted to the display's pixel format: blits of unconverted surfaces are several times slower
        self.terrain = self._terrain_surface().convert()
        self.voronoi = self._voronoi_surface().convert_alpha()
        self.fp_surf = [self._footprint_surface(c).convert_alpha() for c in UAV_COLORS]
        size = self.grid_n * CELL
        self.overlay = pygame.Surface((size, size), pygame.SRCALPHA).convert_alpha()
        self.panel_area = pygame.Rect(PANEL_X, 0, WINDOW[0] - PANEL_X, WINDOW[1])
        self._panel_key, self._panel_cache = None, None
        self.progress_rect = pygame.Rect(PANEL_X + 10, 44, 520, 10)
        self.timeline_rect = pygame.Rect(PANEL_X + 10, 60, 520, 24)

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

    # ---------------------------------------------------------- interpolation
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

    # -------------------------------------------------------------- map
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

    def _in_control(self, f, u):
        """Who flies the drone's next move. flown_by[t] records who flew the move into frame t,
        so the next frame's entry is who is in control at frame f."""
        return int(self.ep["flown_by"][min(f + 1, self.n_frames - 1), u])

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

    # -------------------------------------------------------------- panel
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


def main():
    parser = argparse.ArgumentParser(description="Replay a recorded UAV swarm episode.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--speed", type=int, default=15, help=f"steps per second, one of {SPEEDS}")
    parser.add_argument("--frame", type=int, default=0, help="frame to start from (or to capture)")
    parser.add_argument("--screenshot", help="save this frame as a PNG and exit, without opening a window")
    args = parser.parse_args()
    if args.screenshot:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    dash = Dashboard(args.input, speed=args.speed)
    dash.set_frame(args.frame)
    if args.screenshot:
        dash.playing = False
        dash.render()
        pygame.image.save(dash.screen, args.screenshot)
        print(f"Saved {args.screenshot}")
        pygame.quit()
        return
    dash.run()


if __name__ == "__main__":
    main()
