"""Build the interactive results report from evaluate_mission.py --output files.

Usage:
    python -m evaluation.build_report                                  # every evaluation/results/*.json
    python -m evaluation.build_report --inputs a.json b.json --output report.html

Each input is one evaluate_mission.py run. Arms are keyed by what they are,
not by their number: arms 5 and 6 run with --position-noise > 0 become
"safety_noise" and "smart_noise", and patrol arms (7, 8) go in their own
section because a patrol has no coverage target. When two inputs hold the
same arm at the same target, the later one wins.

The report is a single HTML file with the results embedded; it needs no server.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "report_template.html")
RESULTS = os.path.join(HERE, "results")
ARM_IDS = {1: "policy", 2: "override", 3: "rth", 4: "staggered", 5: "safety", 6: "smart",
           7: "patrol_spares", 8: "patrol_no_spares"}
PATROL = {"patrol_spares", "patrol_no_spares"}


def arm_id(number, noise):
    base = ARM_IDS[number]
    return f"{base}_noise" if noise and base in ("safety", "smart") else base


def columns(results):
    """List of per-map dicts -> {field: [value per map]}; missing values become None."""
    fields = sorted({k for r in results for k in r})
    return {k: [r.get(k) for r in results] for k in fields}


def build(paths):
    data = {"generated": dt.date.today().isoformat(), "targets": {}, "patrol": None, "sources": []}
    for path in paths:
        with open(path) as f:
            run = json.load(f)
        target = f"{round(100 * run['coverage_target'])}"
        noise = float(run.get("settings", {}).get("position_noise") or 0.0)
        notes = run.get("notes", {})
        data["checkpoint"] = run.get("checkpoint")
        data["sources"].append(os.path.basename(path))
        for number, arm in run["arms"].items():
            key = arm_id(int(number), noise)
            entry = {"name": arm["name"], "maps": len(arm["results"]), "note": notes.get(number),
                     "fields": columns(arm["results"])}
            if key in PATROL:
                settings = run.get("settings", {})
                data["patrol"] = data["patrol"] or {"steps": settings.get("patrol_steps"),
                                                    "spare_packs": settings.get("spare_packs"), "arms": {}}
                data["patrol"]["arms"][key] = entry
            else:
                data["targets"].setdefault(target, {"maps": run["maps"], "arms": {}})["arms"][key] = entry
    return data


def main():
    p = argparse.ArgumentParser(description="Build the interactive results report.")
    p.add_argument("--inputs", nargs="*", help="evaluate_mission.py --output files (default: evaluation/results/*.json)")
    p.add_argument("--output", default=os.path.join(RESULTS, "report.html"))
    args = p.parse_args()
    paths = args.inputs or sorted(glob.glob(os.path.join(RESULTS, "*.json")))
    if not paths:
        p.error("no result files: run python -m evaluation.evaluate_mission --output evaluation/results/<name>.json")
    data = build(paths)
    with open(TEMPLATE, encoding="utf-8") as f:
        html = f.read()
    html = html.replace("/*REPORT_DATA*/null", json.dumps(data, separators=(",", ":")))
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html)
    targets = ", ".join(f"{t}%: {len(v['arms'])} arms" for t, v in sorted(data["targets"].items()))
    patrol = f", patrol: {len(data['patrol']['arms'])} arms" if data["patrol"] else ""
    print(f"Wrote {args.output} ({os.path.getsize(args.output) / 1024:.0f} KB) from {len(paths)} files: {targets}{patrol}")


if __name__ == "__main__":
    main()
