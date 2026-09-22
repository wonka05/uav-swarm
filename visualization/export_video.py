"""Export a recorded episode (.npz from record_episode.py) as a video.

Usage:
    python visualization/export_video.py --input <episode.npz> --output <video.avi>
        [--fps 15] [--start 0] [--end LAST] [--hold 1.5]

Every frame is drawn headlessly with the same renderer the interactive
dashboard uses (pygame_dashboard.Dashboard), so the video matches the
dashboard exactly. Only the recorded .npz is read: the environment, the
agents and the planner are never imported or run.

Output formats (chosen by file extension), using only installed packages:
    .avi   Motion-JPEG AVI, full colour, plays in VLC / Windows Media Player
    .gif   animated GIF (larger, 256 colours)
One video frame is one recorded timestep, so --fps is also steps per second.
"""
from __future__ import annotations

import argparse
import io
import os
import struct
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")   # headless rendering
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pygame_dashboard import WINDOW, SPEEDS, Dashboard

JPEG_QUALITY = 90


def _chunk(fourcc, data):
    pad = b"\0" if len(data) % 2 else b""
    return fourcc + struct.pack("<I", len(data)) + data + pad


def _list(kind, payload):
    return b"LIST" + struct.pack("<I", len(payload) + 4) + kind + payload


def write_mjpeg_avi(path, jpeg_frames, width, height, fps):
    """Minimal RIFF AVI writer: one Motion-JPEG video stream plus an idx1 index."""
    n = len(jpeg_frames)
    max_frame = max(len(f) for f in jpeg_frames)
    avih = struct.pack("<IIIIIIIIII4I", int(1_000_000 / fps), max_frame * fps, 0, 0x10, n, 0, 1,
                       max_frame, width, height, 0, 0, 0, 0)
    strh = struct.pack("<4s4sIHHIIIIIIIIhhhh", b"vids", b"MJPG", 0, 0, 0, 0, 1, fps, 0, n,
                       max_frame, 0xFFFFFFFF, 0, 0, 0, width, height)
    strf = struct.pack("<IiiHH4sIiiII", 40, width, height, 1, 24, b"MJPG", width * height * 3, 0, 0, 0, 0)
    hdrl = _list(b"hdrl", _chunk(b"avih", avih) + _list(b"strl", _chunk(b"strh", strh) + _chunk(b"strf", strf)))

    movi_body, index, offset = bytearray(), bytearray(), 4   # offsets count from the 'movi' fourcc
    for f in jpeg_frames:
        c = _chunk(b"00dc", f)
        index += b"00dc" + struct.pack("<III", 0x10, offset, len(f))
        movi_body += c
        offset += len(c)
    body = b"AVI " + hdrl + _list(b"movi", bytes(movi_body)) + _chunk(b"idx1", bytes(index))
    with open(path, "wb") as fh:
        fh.write(b"RIFF" + struct.pack("<I", len(body)) + body)


def render_frames(npz, start, end, fps):
    """Yield (frame_index, PIL.Image) for every recorded frame in [start, end]."""
    screen = pygame.Surface(WINDOW)
    dash = Dashboard(npz, speed=min(SPEEDS, key=lambda s: abs(s - fps)), screen=screen)
    last = dash.n_frames - 1
    end = last if end is None else min(end, last)
    start = max(0, min(start, end))

    clock_ms = [0]
    pygame.time.get_ticks = lambda: clock_ms[0]   # target pulse/flicker follows video time, not wall time
    for f in range(start, end + 1):
        clock_ms[0] = int((f - start) * 1000 / fps)
        dash.set_frame(f)
        dash.playing = f < last
        dash.render()
        yield f, Image.frombytes("RGB", WINDOW, pygame.image.tobytes(dash.screen, "RGB"))


def main():
    parser = argparse.ArgumentParser(
        description="Export a recorded UAV swarm episode (.npz) as a video, rendered like the dashboard.",
        epilog="Example: python visualization/export_video.py --input episode_seed42.npz "
               "--output episode_seed42.avi --fps 15")
    parser.add_argument("--input", required=True, help="episode .npz written by record_episode.py")
    parser.add_argument("--output", required=True, help="output video path ending in .avi (recommended) or .gif")
    parser.add_argument("--fps", type=int, default=15, help="frames (= recorded steps) per second, default 15")
    parser.add_argument("--start", type=int, default=0, help="first recorded frame to export, default 0")
    parser.add_argument("--end", type=int, default=None, help="last recorded frame to export, default: episode end")
    parser.add_argument("--hold", type=float, default=1.5, help="seconds to hold the final frame, default 1.5")
    args = parser.parse_args()

    ext = os.path.splitext(args.output)[1].lower()
    if ext not in (".avi", ".gif"):
        parser.error("output must end in .avi or .gif (no MP4 encoder is installed)")
    if args.fps < 1:
        parser.error("--fps must be at least 1")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    frames = []
    first = last = None
    for f, img in render_frames(args.input, args.start, args.end, args.fps):
        first = f if first is None else first
        last = f
        if ext == ".avi":
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=JPEG_QUALITY)
            frames.append(buf.getvalue())
        else:
            frames.append(img.quantize(colors=256, method=Image.Quantize.MEDIANCUT))
        if (f - first) % 50 == 0:
            print(f"  rendered frame {f}", flush=True)
    frames += [frames[-1]] * int(round(args.hold * args.fps))

    if ext == ".avi":
        write_mjpeg_avi(args.output, frames, WINDOW[0], WINDOW[1], args.fps)
    else:
        frames[0].save(args.output, save_all=True, append_images=frames[1:],
                       duration=int(1000 / args.fps), loop=0, optimize=False)
    pygame.quit()

    size = os.path.getsize(args.output)
    print(f"Exported frames {first}-{last} ({last - first + 1} steps + {len(frames) - (last - first + 1)} hold) "
          f"at {args.fps} fps = {len(frames) / args.fps:.1f} s")
    print(f"Output: {os.path.abspath(args.output)}  ({size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
