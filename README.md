# rgp-reader

This repository was made with the assistance of AI.

Reads AMD Radeon GPU Profiler captures (`.rgp`) without RGP and prints event timings as text or JSON. Event durations
match RGP's Event timing pane.

It is designed primarily to be run by an AI coding agent, so the agent can read profiler results directly instead of
working from screenshots. Output is compact, greppable text with a self-check line, plus JSON for further processing.
It reads fine by hand too. 

Built for an RX 7900 XT (RDNA3, gfx11) and Vulkan captures taken with the Radeon Developer Panel on Windows. Other
GPUs are untested.

## Requirements

Python 3.10+ and numpy.

## Usage

```
python -I rgp.py                                       # newest capture, summary
python -I rgp.py latest events --filter ReSTIR         # every event matching a label or API name
python -I rgp.py latest regions --depth 2              # user marker (debug label) tree
python -I rgp.py latest barriers                       # barrier stalls and GPU idle gaps
python -I rgp.py A.rgp diff B.rgp --depth 1            # per region delta, B minus A
python -I rgp.py latest json                           # everything as JSON
```

A capture argument is a path, a file name, or `latest`. Names and `latest` are looked up recursively under
`$RGP_CAPTURE_DIR` (default `~/Documents/rgp_profiles`). Every command also writes its output to `output/`.

Options: `--filter`, `--depth`, `--top`, `--all` (include events without waves), `--min-gap` (us), `--min-delta` (ms).

## Output

```
restir night initial visibility off.rgp  [AMD_RDF]
gpu AMD Radeon RX 7900 XT (0x744c), 6 SE, sclk 2075 MHz
frame span 15.668 ms, GPU busy 14.448 ms (92.2%), events 434, queues [(0, 0), (1, 0)]
checks: clean

top events by duration (us):
   idx  dur(us)  start(us) api                    label                             waves  pipeline
   357    3656.9    +4080.4 Dispatch               [ReSTIR DI] SpatioTemporal        31200 w cs v189 s71 lds16384 w32 occ8
   374    1608.6    +9176.5 Dispatch               GI Diffuse Upscale                31200 w cs v158 s106 w64 occ4
```

- `dur`: first wave start to last wave end, the same definition RGP uses. Overlapping events each count the overlap,
  so durations summed over a frame exceed the frame time. Use `frame span` and `GPU busy` for the frame itself.
- `start`: microseconds from the first wave of the capture.
- `pipeline`: hardware stage, VGPRs, SGPRs, LDS bytes, wave size, and a theoretical waves per SIMD (out of 16)
  estimated from VGPR and LDS use. The occupancy figure is this tool's estimate, not RGP's.
- `~` marks graphics events, whose wave attribution is approximate (see below).
- `checks` reports self-consistency counters. `events issued after the trace stopped` and `waves from before the first
  marker` are normal trace edges.

## How it works

1. **Container.** Reads both `AMD_RDF` (RGP 2.x) and the older `B00P` chunk format.
2. **SQTT tokens.** Each shader engine's thread trace is decoded with the gfx11 token encodings from AMD's
   rocprof-trace-decoder. Shader clock is mapped to the 100 MHz GPU timestamp through the trace's realtime tokens.
3. **Driver markers.** The driver writes RGP markers (events, debug labels, barriers, pipeline binds) into
   `SQ_THREAD_TRACE_USERDATA_2/3`. They arrive as register write tokens at command processor time.
4. **Compute waves.** Each shader engine logs the dispatch initiator in launch order. Waves that start after the k-th
   initiator belong to dispatch k. This is exact: every dispatch's wave count is checked against groups times waves
   per group (`dispatch wave mismatches`).
5. **Graphics waves.** No launch-order draw token has been identified. A graphics wave is assigned to the last
   non-dispatch event whose marker precedes it. This is accurate when barriers separate draws from earlier work, and
   approximate for back-to-back draws.
6. **Pipelines.** Bind markers map through `PsoCorrelation` to the pipeline's code object. Its ELF msgpack metadata
   gives register, LDS and wave-size figures.

## Validation

Region times were compared against RGP's Event timing pane on captures from 2026-07 to 2026-10 and agree to 0.01 ms,
for example 2.442 / 2.274 / 2.139 ms against RGP's 2.44 / 2.27 / 2.14 ms.

## Limitations

- Instruction timing is not decoded. With instruction tracing enabled, the traced shader engine's stream contains a
  token missing from the public gfx11 table. That engine is skipped and reported in `checks`; durations come from the
  remaining engines, and wave counts are partial.
- SPM counters are not read.
- Async compute waves are only attributed when a single compute queue is active.
- Zstd-compressed RDF chunks are not supported. None have been seen in practice.

## Credits

- SQTT token encodings: [rocprof-trace-decoder](https://github.com/ROCm/rocm-systems/tree/develop/projects/rocprof-trace-decoder),
  Copyright (c) Advanced Micro Devices, Inc., MIT License.
- RGP marker layouts: Mesa `src/amd/common/ac_sqtt.h`, MIT License.

## License

MIT, see `LICENSE`.
