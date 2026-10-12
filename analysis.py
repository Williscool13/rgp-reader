"""
Turns an RGP capture into events with GPU timings, the way RGP's Event timing pane does: an event lasts from its first
wave start to its last wave end.

Wave attribution:
- Compute: per shader engine, the SPI logs the dispatch initiator as a REG token with the CS bit set, in launch order.
  Waves started after the k-th such token belong to the k-th dispatch of that queue. Exact; verified by comparing the
  wave count against groups * waves-per-group (check "dispatch wave mismatches").
- Graphics: no launch-order draw token is known, so a graphics wave belongs to the last non-dispatch event whose CP
  marker precedes it. Fine when a barrier sits between draws and their predecessors, approximate otherwise.

Dispatches map to events by CP order: the dispatch initiator REG_INIT belongs to the latest event marker before it.
"""
import bisect
import collections
import re
import struct

import numpy as np

import codeobj
import markers
import rgp_file
import sqtt

REG = sqtt.TOKEN["REG"]
REG_INIT = sqtt.TOKEN["REG_INIT"]
WAVE_START = sqtt.TOKEN["WAVE_START"]
WAVE_END = sqtt.TOKEN["WAVE_END"]

COMPUTE_DIM_X = 0x2e01
COMPUTE_NUM_THREAD_X = 0x2e07
COMPUTE_PGM_LO = 0x2e0c
COMPUTE_PGM_HI = 0x2e0d
CS_W32_EN = 0x8000
STAGE_CS = 6
STAGE_NAMES = {0: "PS", 1: "VS", 2: "GS", 3: "ES", 4: "HS", 5: "LS", 6: "CS", 7: "PS", 8: "PS", 9: "PS"}
NON_WORK_EVENTS = ("PipelineBarrier", "ResetQueryPool", "CopyQueryPoolResults", "WaitEvents")


class Event:
    __slots__ = ("index", "queue", "t_cp", "api", "cmd", "dims", "label", "dispatches", "groups", "waves", "ws", "we", "life",
                 "stages", "pso", "pipeline", "expected_waves", "approx")

    def __init__(self, index, queue, t_cp, f, label):
        self.index = index
        self.queue = queue
        self.t_cp = t_cp
        self.api = f.get("api")
        self.cmd = f.get("cmd")
        self.dims = f.get("dims")
        self.label = label
        self.dispatches = 0
        self.groups = 0
        self.waves = 0
        self.ws = None
        self.we = None
        self.life = 0.0
        self.stages = collections.Counter()
        self.pso = None
        self.pipeline = None
        self.expected_waves = 0
        self.approx = False

    @property
    def duration_us(self):
        return (self.we - self.ws) if self.ws is not None else 0.0


class Pipeline:
    __slots__ = ("hash", "kind", "stages")

    def __init__(self, h, kind, stages):
        self.hash = h
        self.kind = kind
        self.stages = stages


class Capture:
    pass


def _clock_map(realtime, ts_hz):
    xs = np.array([a for a, b in realtime], dtype=np.float64)
    ys = np.array([b for a, b in realtime], dtype=np.float64) * (1e6 / ts_hz)
    return xs, ys


def _to_us(xs, ys, t):
    return np.interp(np.asarray(t, dtype=np.float64), xs, ys)


def _code_objects(chunks):
    out = {}
    for c in chunks:
        if c.id == "CodeObject" and len(c.hdr) >= 24:
            out[bytes(c.hdr[8:24])] = c
    return out


def _pipeline_from_elf(h, elf):
    try:
        meta, syms = codeobj.pipeline_meta(elf)
    except (ValueError, IndexError, struct.error):
        return None
    if not meta:
        return None
    p = meta.get("amdpal.pipelines", [{}])[0]
    stages = {}
    for name, s in p.get(".hardware_stages", {}).items():
        stages[name.strip(".")] = {
            "vgpr": s.get(".vgpr_count", 0), "sgpr": s.get(".sgpr_count", 0), "lds": s.get(".lds_size", 0),
            "scratch": s.get(".scratch_memory_size", 0), "wave": s.get(".wavefront_size", 64),
            "tg": tuple(s.get(".threadgroup_dimensions", ())),
        }
    return Pipeline(h, p.get(".type", ""), stages)


def theoretical_occupancy(asic, stage):
    wave = stage["wave"]
    alloc = max(asic.min_vgpr_alloc, -(-max(stage["vgpr"], 1) // asic.vgpr_granularity) * asic.vgpr_granularity)
    occ = min(asic.waves_per_simd, asic.vgprs_per_simd // (alloc * (2 if wave == 64 else 1)))
    if stage["lds"] and stage["tg"]:
        threads = stage["tg"][0] * stage["tg"][1] * stage["tg"][2]
        groups_per_cu = asic.lds_bytes // stage["lds"]
        occ = min(occ, groups_per_cu * -(-threads // wave) // asic.simd_per_cu)
    return occ


def load(path):
    container, chunks = rgp_file.read_chunks(path)
    asic = rgp_file.parse_asic(container, chunks)
    sq = [c for c in chunks if c.id == "SqttData"]
    if not sq:
        raise ValueError("capture has no SqttData")
    cap = Capture()
    cap.path = path
    cap.container = container
    cap.asic = asic
    cap.shader_engines = len(sq)
    cap.checks = collections.Counter()

    per_se = []
    cap.dropped_se = []
    for si, c in enumerate(sq):
        toks, realtime, ok = sqtt.tokenize(c.data)
        if not ok:
            cap.dropped_se.append(si)
            continue
        per_se.append((toks, realtime))
    if not per_se:
        raise ValueError("no shader engine stream decoded cleanly")

    marker_se = max(range(len(per_se)), key=lambda s: sum(1 for ty, t, w in per_se[s][0] if ty == REG and (w >> 16) & 0xfffe == 0xc342))
    toks0, rt0 = per_se[marker_se]
    xs0, ys0 = _clock_map(rt0, asic.timestamp_hz)

    dws = collections.defaultdict(list)
    state = collections.defaultdict(lambda: [1, 1, 1, 1, 1, 1, 0, 0])
    cpdisp = collections.defaultdict(list)
    for ty, t, w in toks0:
        if ty == REG:
            if (w >> 15) & 1:
                continue
            a = (w >> 16) & 0xffff
            key = ((w >> 9) & 3, (w >> 7) & 3)
            v = w >> 32
            if a == markers.USERDATA_2 or a == markers.USERDATA_3:
                dws[key].append((t, v))
            elif COMPUTE_DIM_X <= a <= COMPUTE_DIM_X + 2:
                state[key][a - COMPUTE_DIM_X] = v
            elif COMPUTE_NUM_THREAD_X <= a <= COMPUTE_NUM_THREAD_X + 2:
                state[key][3 + a - COMPUTE_NUM_THREAD_X] = v
            elif a == COMPUTE_PGM_LO:
                state[key][6] = v
            elif a == COMPUTE_PGM_HI:
                state[key][7] = v
        elif ty == REG_INIT and ((w >> 18) & 3) == 2 and (w >> 20) & 1:
            key = ((w >> 16) & 3, (w >> 14) & 3)
            s = state[key]
            cpdisp[key].append((t, tuple(s[:3]), tuple(s[3:6]), (w >> 20) & 0xffffffff, s[6] << 8 | s[7] << 40))

    cos = _code_objects(chunks)
    pso_to_pipeline = {}
    for c in chunks:
        if c.id == "PsoCorrelation" and container == "AMD_RDF":
            d = bytes(c.data)
            for i in range(0, len(d) - 95, 96):
                api, = struct.unpack_from("<Q", d, i + 8)
                pso_to_pipeline[api] = d[i + 16:i + 32]
    pipelines = {}

    def pipeline_for(pso):
        h = pso_to_pipeline.get(pso)
        if h is None or h not in cos:
            return None
        if h not in pipelines:
            pipelines[h] = _pipeline_from_elf(h.hex(), cos[h].data)
        return pipelines[h]

    loads = []
    for c in chunks:
        if c.id == "COLoadEvent" and container == "AMD_RDF":
            d = bytes(c.data)
            for i in range(0, len(d) - 39, 40):
                _, kind, base, h0, h1, ts = struct.unpack_from("<IIQQQQ", d, i)
                loads.append((ts, kind, base, struct.pack("<QQ", h0, h1)))
    live = {}
    for ts, kind, base, h in sorted(loads):
        if kind == 0:
            live[base] = h
        elif live.get(base) == h:
            del live[base]
    bases = sorted(live)

    def pipeline_at(addr):
        i = bisect.bisect_right(bases, addr) - 1
        if i < 0:
            return None
        h = live[bases[i]]
        if h not in cos or addr - bases[i] > len(cos[h].data):
            return None
        if h not in pipelines:
            pipelines[h] = _pipeline_from_elf(h.hex(), cos[h].data)
        return pipelines[h]

    events = []
    barriers = []
    labels = {}
    queue_events = {}
    cap.marker_unknown = 0
    for key in sorted(dws):
        ms, unknown = markers.parse(dws[key])
        cap.marker_unknown += unknown
        stack = []
        bound = {0: None, 1: None}
        open_barrier = None
        qe = []
        for m in ms:
            if m.kind == "push":
                stack.append(m.f["name"])
            elif m.kind == "pop":
                if stack:
                    stack.pop()
            elif m.kind == "bind":
                bound[m.f["point"] & 1] = m.f["hash"]
            elif m.kind == "event":
                e = Event(len(events), key, m.t, m.f, tuple(stack))
                compute = e.dims is not None or e.api in ("Dispatch", "DispatchIndirect", "TraceRays", "TraceRaysIndirect")
                e.pso = bound[1] if compute else bound[0]
                events.append(e)
                qe.append(e)
            elif m.kind == "barrier_start":
                open_barrier = (m.t, tuple(stack), m.f.get("reason"))
            elif m.kind == "barrier_end" and open_barrier:
                barriers.append({"queue": key, "t0": open_barrier[0], "t1": m.t, "label": open_barrier[1], "reason": open_barrier[2]})
                open_barrier = None
        queue_events[key] = qe
        labels[key] = stack

    owners = {}
    for key, ds in cpdisp.items():
        qe = queue_events.get(key, [])
        times = [e.t_cp for e in qe]
        own = []
        for t, dims, nt, init, addr in ds:
            i = bisect.bisect_right(times, t) - 1
            if i < 0:
                own.append(None)
                continue
            e = qe[i]
            own.append(e)
            e.dispatches += 1
            groups = dims[0] * dims[1] * dims[2]
            e.groups += groups
            e.expected_waves += groups * -(-(nt[0] * nt[1] * nt[2]) // (32 if init & CS_W32_EN else 64))
            if e.pipeline is None:
                p = pipeline_for(e.pso) if e.pso not in (None, 0, 0xFFFFFFFFFFFFFFFF) else None
                e.pipeline = p or pipeline_at(addr)
        owners[key] = own

    gfx_events = [e for e in queue_events.get((0, 0), []) if (e.dispatches == 0 or e.api.startswith("Draw")) and e.api not in NON_WORK_EVENTS]
    for e in gfx_events:
        if e.pipeline is None and e.pso not in (None, 0, 0xFFFFFFFFFFFFFFFF):
            e.pipeline = pipeline_for(e.pso)

    for e in events:
        e.t_cp = float(_to_us(xs0, ys0, e.t_cp))
    for b in barriers:
        b["t0"], b["t1"] = (float(x) for x in _to_us(xs0, ys0, [b["t0"], b["t1"]]))
    gfx_times = [e.t_cp for e in gfx_events]

    mec_keys = {k for k in cpdisp if k[0] >= 1}
    all_starts = []
    all_ends = []
    for si, (toks, realtime) in enumerate(per_se):
        xs, ys = _clock_map(realtime, asic.timestamp_hz)
        seg = collections.defaultdict(lambda: -1)
        live_waves = {}
        rec_owner = []
        rec_t0 = []
        rec_t1 = []
        rec_stage = []
        for ty, t, w in toks:
            if ty == WAVE_START:
                stage = (w >> 18) & 0xf
                me = (w >> 23) & 1
                slot = (w >> 7) & 0x7ff
                if me:
                    key = next(iter(mec_keys)) if len(mec_keys) == 1 else None
                    if key is None:
                        cap.checks["async waves unattributed"] += 1
                        continue
                    live_waves[slot] = (key, seg[key], t, STAGE_CS)
                elif stage == STAGE_CS:
                    live_waves[slot] = ((0, 0), seg[(0, 0)], t, STAGE_CS)
                else:
                    live_waves[slot] = (None, -1, t, stage)
            elif ty == WAVE_END:
                slot = ((w >> 8) & 1) | ((w >> 9) & 3) << 1 | ((w >> 11) & 7) << 3 | ((w >> 15) & 31) << 6
                r = live_waves.pop(slot, None)
                if r is None:
                    cap.checks["wave end without start"] += 1
                    continue
                rec_owner.append((r[0], r[1]))
                rec_t0.append(r[2])
                rec_t1.append(t)
                rec_stage.append(r[3])
            elif ty == REG and (w >> 15) & 1 and (w >> 16) & 0xffff == 0:
                seg[((w >> 9) & 3, (w >> 7) & 3)] += 1
        cap.checks["waves still running at trace end"] += len(live_waves)
        t0s = _to_us(xs, ys, rec_t0)
        t1s = _to_us(xs, ys, rec_t1)
        all_starts.append(t0s)
        all_ends.append(t1s)
        for (key, k), ws, we, stage in zip(rec_owner, t0s.tolist(), t1s.tolist(), rec_stage):
            if key is None:
                j = bisect.bisect_right(gfx_times, ws) - 1
                if j < 0:
                    cap.checks["waves from before the first marker"] += 1
                    continue
                e = gfx_events[j]
                e.approx = True
            else:
                own = owners.get(key)
                if own is None or k < 0 or k >= len(own) or own[k] is None:
                    cap.checks["waves from before the first marker"] += 1
                    continue
                e = own[k]
            e.waves += 1
            e.life += we - ws
            e.stages[STAGE_NAMES.get(stage, str(stage))] += 1
            if e.ws is None or ws < e.ws:
                e.ws = ws
            if e.we is None or we > e.we:
                e.we = we

    if cap.dropped_se:
        cap.checks["SEs skipped (instruction trace stream), wave counts partial"] = len(cap.dropped_se)
    else:
        cut = []
        for key, qe in queue_events.items():
            for e in reversed(qe):
                if e.waves:
                    break
                cut.append(e)
        cut = [e for e in cut if e.dispatches]
        cap.checks["events issued after the trace stopped"] = len(cut)
        cap.checks["dispatch wave mismatches"] = sum(1 for e in events if e.dispatches and not e.api.startswith("Draw") and e.waves != e.expected_waves
                                                     and e not in cut)
    starts = np.concatenate(all_starts)
    ends = np.concatenate(all_ends)
    order = np.argsort(starts)
    cap.busy = _union(starts[order], ends[order])
    worked = [e for e in events if e.ws is not None]
    cap.t0 = min(e.ws for e in worked) if worked else 0.0
    cap.t1 = max(e.we for e in worked) if worked else 0.0
    cap.events = events
    cap.barriers = barriers
    cap.pipelines = pipelines
    cap.queues = sorted(queue_events)
    return cap


def _union(starts, ends):
    out = []
    if len(starts) == 0:
        return out
    cs = float(starts[0])
    ce = float(ends[0])
    for s, e in zip(starts.tolist(), ends.tolist()):
        if s > ce:
            out.append((cs, ce))
            cs = s
            ce = e
        elif e > ce:
            ce = e
    out.append((cs, ce))
    return out


FRAME_SUFFIX = re.compile(r" F\d+$")


def normalize_label(label):
    lab = tuple(FRAME_SUFFIX.sub("", x) for x in label)
    if lab and lab[0] in ("Frame", "Async Cut"):
        lab = lab[1:]
    return lab


def label_of(e):
    return normalize_label(e.label)


def regions(cap):
    reg = collections.OrderedDict()
    for e in cap.events:
        if e.ws is None:
            continue
        lab = label_of(e)
        for d in range(1, len(lab) + 1):
            k = lab[:d]
            a = reg.get(k)
            if a is None:
                reg[k] = [e.ws, e.we, 1, e.duration_us, e.waves]
            else:
                a[0] = min(a[0], e.ws)
                a[1] = max(a[1], e.we)
                a[2] += 1
                a[3] += e.duration_us
                a[4] += e.waves
    return reg


def idle_gaps(cap, min_us):
    gaps = []
    for (s0, e0), (s1, e1) in zip(cap.busy, cap.busy[1:]):
        if cap.t0 <= e0 and s1 <= cap.t1 and s1 - e0 >= min_us:
            gaps.append((e0, s1))
    return gaps
