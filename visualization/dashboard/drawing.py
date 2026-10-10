"""Small pygame drawing helpers."""
from __future__ import annotations

import math

import numpy as np
import pygame

from dashboard.style import AMBER, LOST


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
