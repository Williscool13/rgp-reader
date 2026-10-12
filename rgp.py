#!/usr/bin/env python3
"""
Reads Radeon GPU Profiler captures (.rgp) without RGP: event timings, user marker regions, barriers, idle gaps and
pipeline resource usage. Numbers match RGP's Event timing pane (event = first wave start to last wave end).

    python -I rgp.py [capture|latest] [summary]          # default: newest capture, summary
    python -I rgp.py latest events [--filter ReSTIR]     # every event
    python -I rgp.py latest regions [--depth 2]          # user marker (pass label) tree
    python -I rgp.py latest barriers                     # barriers and idle gaps
    python -I rgp.py A.rgp diff B.rgp                    # per-region delta, B minus A
    python -I rgp.py latest json [out.json]              # everything as JSON

Captures resolve against $RGP_CAPTURE_DIR (default ~/Documents/rgp_profiles, searched recursively) when the path does
not exist; "latest" picks the newest. Every command also writes its output to output/<capture>_<command>.txt next to
this script; json goes to output/ unless a path is given.
Durations are microseconds unless marked ms. Graphics events carry "~": their waves are attributed by CP order, not
launch order (see analysis.py).
"""
import argparse
import bisect
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import analysis

CAPTURE_DIR = os.environ.get("RGP_CAPTURE_DIR", os.path.join(os.path.expanduser("~"), "Documents", "rgp_profiles"))
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


class Tee:
    def __init__(self, path):
        self.file = open(path, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, s):
        self.stdout.write(s)
        self.file.write(s)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def resolve(arg):
    files = glob.glob(os.path.join(CAPTURE_DIR, "**", "*.rgp"), recursive=True)
    if arg == "latest":
        if not files:
            sys.exit("no captures under %s" % CAPTURE_DIR)
        return max(files, key=os.path.getmtime)
    if os.path.exists(arg):
        return arg
    for f in sorted(files, key=os.path.getmtime, reverse=True):
        name = os.path.basename(f)
        if name == arg or name == arg + ".rgp":
            return f
    sys.exit("capture not found: %s" % arg)


def label_str(e):
    return "/".join(analysis.label_of(e)) or "-"


def pipe_str(cap, e):
    p = e.pipeline
    if p is None:
        return ""
    parts = []
    for name, s in p.stages.items():
        parts.append("%s v%d s%d%s w%d occ%d" % (name, s["vgpr"], s["sgpr"], " lds%d" % s["lds"] if s["lds"] else "", s["wave"],
                                                 analysis.theoretical_occupancy(cap.asic, s)))
    return " ".join(parts)


def print_header(cap):
    a = cap.asic
    print("%s  [%s]" % (os.path.basename(cap.path), cap.container))
    print("gpu %s (0x%x), %d SE, sclk %.0f MHz" % (a.name or "?", a.device_id, cap.shader_engines, a.sclk_hz / 1e6))
    busy = sum(min(e, cap.t1) - max(s, cap.t0) for s, e in cap.busy if e > cap.t0 and s < cap.t1)
    span = cap.t1 - cap.t0
    print("frame span %.3f ms, GPU busy %.3f ms (%.1f%%), events %d, queues %s" % (span / 1e3, busy / 1e3, 100 * busy / span if span else 0,
                                                                                 len(cap.events), cap.queues))
    checks = ", ".join("%s %d" % (k, v) for k, v in sorted(cap.checks.items()) if v)
    print("checks: %s" % (checks or "clean"))


def cmd_summary(cap, args):
    print_header(cap)
    print()
    print("top-level regions:")
    print("  span(ms)  sum(ms)  events  region")
    reg = analysis.regions(cap)
    rows = [(k, v) for k, v in reg.items() if len(k) == 1]
    rows.sort(key=lambda kv: -kv[1][3])
    for k, v in rows[:args.top]:
        print("  %8.3f %8.3f  %4d ev  %s" % ((v[1] - v[0]) / 1e3, v[3] / 1e3, v[2], k[0]))
    print()
    print("top events by duration (us):")
    print("   idx  dur(us)  start(us) api                    label                                                      waves  pipeline (stage vgpr sgpr lds wave theoretical-occupancy/16)")
    evs = sorted((e for e in cap.events if e.ws is not None), key=lambda e: -e.duration_us)
    for e in evs[:args.top]:
        print_event(cap, e)


def print_event(cap, e):
    print("  %4d %9.1f %+10.1f %-22s %-56s %7d w %s%s" % (e.index, e.duration_us, (e.ws or cap.t0) - cap.t0, e.api[:22],
                                                         label_str(e)[:56], e.waves, "~ " if e.approx else "", pipe_str(cap, e)))


def cmd_events(cap, args):
    print_header(cap)
    print("   idx  dur(us)  start(us) api                    label                                                      waves  pipeline (stage vgpr sgpr lds wave theoretical-occupancy/16)")
    for e in cap.events:
        if args.filter and args.filter.lower() not in (label_str(e) + " " + e.api).lower():
            continue
        if not args.all and e.ws is None:
            continue
        print_event(cap, e)


def cmd_regions(cap, args):
    print_header(cap)
    print("  span(ms)  sum(ms)  events  waves  region")
    for k, v in analysis.regions(cap).items():
        if len(k) > args.depth:
            continue
        name = "/".join(k)
        if args.filter and args.filter.lower() not in name.lower():
            continue
        print("  %8.3f %8.3f %6d %8d  %s%s" % ((v[1] - v[0]) / 1e3, v[3] / 1e3, v[2], v[4], "  " * (len(k) - 1), k[-1]))


def cmd_barriers(cap, args):
    print_header(cap)
    print()
    print("barriers (CP time, us):")
    for b in sorted(cap.barriers, key=lambda b: -(b["t1"] - b["t0"]))[:args.top]:
        print("  %8.1f at %+10.1f  reason 0x%x  %s" % (b["t1"] - b["t0"], b["t0"] - cap.t0, b["reason"] or 0, "/".join(analysis.normalize_label(b["label"])) or "-"))
    print()
    print("GPU idle gaps >= %.1f us (no wave running on any queue):" % args.min_gap)
    evs = sorted((e for e in cap.events if e.ws is not None), key=lambda e: e.we)
    ends = [e.we for e in evs]
    starts = sorted((e for e in cap.events if e.ws is not None), key=lambda e: e.ws)
    stimes = [e.ws for e in starts]
    gaps = analysis.idle_gaps(cap, args.min_gap)
    total = sum(b - a for a, b in gaps)
    print("  %d gaps, %.3f ms total" % (len(gaps), total / 1e3))
    for a, b in sorted(gaps, key=lambda g: g[0] - g[1])[:args.top]:
        i = bisect.bisect_right(ends, a + 0.01) - 1
        j = bisect.bisect_left(stimes, b - 0.01)
        before = label_str(evs[i]) if i >= 0 else "-"
        after = label_str(starts[j]) if j < len(starts) else "-"
        print("  %8.1f at %+10.1f  after %s  ->  before %s" % (b - a, a - cap.t0, before[:50], after[:50]))


def region_map(cap):
    return {"/".join(k): v for k, v in analysis.regions(cap).items()}


def cmd_diff(cap, args):
    other = analysis.load(resolve(args.other))
    a = region_map(cap)
    b = region_map(other)
    print("A %s  span %.3f ms" % (os.path.basename(cap.path), (cap.t1 - cap.t0) / 1e3))
    print("B %s  span %.3f ms" % (os.path.basename(other.path), (other.t1 - other.t0) / 1e3))
    print("  A sum(ms)  B sum(ms)   delta  region")
    keys = [k for k in dict.fromkeys(list(a) + list(b)) if k.count("/") < args.depth]
    rows = []
    for k in keys:
        va = a[k][3] / 1e3 if k in a else 0.0
        vb = b[k][3] / 1e3 if k in b else 0.0
        rows.append((vb - va, va, vb, k))
    rows.sort(key=lambda r: -abs(r[0]))
    for d, va, vb, k in rows[:args.top]:
        if abs(d) < args.min_delta:
            continue
        print("  %8.3f  %8.3f  %+7.3f  %s" % (va, vb, d, k))


def cmd_json(cap, args):
    out = {
        "capture": cap.path, "container": cap.container, "frame_span_us": cap.t1 - cap.t0, "checks": dict(cap.checks),
        "asic": vars(cap.asic),
        "events": [{
            "index": e.index, "queue": list(e.queue), "api": e.api, "label": list(e.label), "start_us": (e.ws - cap.t0) if e.ws is not None else None,
            "duration_us": e.duration_us, "waves": e.waves, "groups": e.groups, "dispatches": e.dispatches, "approx": e.approx,
            "mean_wave_us": e.life / e.waves if e.waves else 0, "stages": dict(e.stages), "cp_us": e.t_cp - cap.t0,
            "pipeline": {"hash": e.pipeline.hash, "stages": e.pipeline.stages} if e.pipeline else None,
        } for e in cap.events],
        "regions": [{"label": list(k), "span_us": v[1] - v[0], "sum_us": v[3], "events": v[2], "waves": v[4]} for k, v in analysis.regions(cap).items()],
        "barriers": [dict(b, t0=b["t0"] - cap.t0, t1=b["t1"] - cap.t0, label=list(b["label"])) for b in cap.barriers],
        "idle_gaps_us": [(a - cap.t0, b - a) for a, b in analysis.idle_gaps(cap, 1.0)],
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1, default=str)
    print("wrote %s" % args.out)


def main():
    commands = {"summary": cmd_summary, "events": cmd_events, "regions": cmd_regions, "barriers": cmd_barriers, "diff": cmd_diff, "json": cmd_json}
    argv = sys.argv[1:]
    if not argv or argv[0] in commands:
        argv = ["latest"] + argv
    if len(argv) == 1:
        argv.append("summary")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("capture")
    p.add_argument("command", choices=sorted(commands))
    p.add_argument("other", nargs="?", help="second capture (diff) or output path (json)")
    p.add_argument("--filter", default="")
    p.add_argument("--depth", type=int, default=99)
    p.add_argument("--top", type=int, default=30)
    p.add_argument("--all", action="store_true", help="events: include events without waves")
    p.add_argument("--min-gap", type=float, default=5.0)
    p.add_argument("--min-delta", type=float, default=0.005)
    args = p.parse_args(argv)
    if args.command == "diff" and not args.other:
        sys.exit("diff needs a second capture")
    path = resolve(args.capture)
    stem = os.path.splitext(os.path.basename(path))[0]
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if args.command == "json":
        args.out = args.other or os.path.join(OUTPUT_DIR, stem + ".json")
    else:
        suffix = "_" + os.path.splitext(os.path.basename(args.other))[0] if args.command == "diff" else ""
        sys.stdout = Tee(os.path.join(OUTPUT_DIR, "%s%s_%s.txt" % (stem, suffix, args.command)))
    cap = analysis.load(path)
    commands[args.command](cap, args)


if __name__ == "__main__":
    main()
