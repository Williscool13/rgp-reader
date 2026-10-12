"""
Container readers for Radeon GPU Profiler captures.

Two containers exist: `AMD_RDF ` (RGP 2.x, Windows driver captures from 2026-08 on) and the older chunked
`B00P` format (the Mesa ac_rgp.c layout). Both are reduced to a list of Chunk(id, version, hdr, data).
"""
import re
import struct

B00P_NAMES = {
    0: "AsicInfo", 1: "SqttDesc", 2: "SqttData", 3: "ApiInfo", 5: "QueueEventTimings", 6: "ClockCalibration",
    7: "CpuInfo", 8: "SpmDb", 9: "CodeObjectDb", 10: "COLoadEvent", 11: "PsoCorrelation", 128: "DerivedSpmDb",
}


class Chunk:
    __slots__ = ("id", "version", "hdr", "data")

    def __init__(self, cid, version, hdr, data):
        self.id = cid
        self.version = version
        self.hdr = hdr
        self.data = data


class Asic:
    def __init__(self):
        self.sclk_hz = 0
        self.timestamp_hz = 100000000
        self.device_id = 0
        self.vgprs_per_simd = 1536
        self.shader_engines = 0
        self.cu_per_se = 0
        self.simd_per_cu = 2
        self.waves_per_simd = 16
        self.min_vgpr_alloc = 24
        self.vgpr_granularity = 24
        self.lds_bytes = 65536
        self.gfxip = 0
        self.name = ""


def _read_b00p(d):
    off = struct.unpack_from("<I", d, 0x10)[0]
    out = []
    while off + 16 <= len(d):
        cid, minor, major, size = struct.unpack_from("<IHHI", d, off)
        if size < 16:
            break
        ty = cid & 0xff
        if ty == 2:
            po, ps = struct.unpack_from("<ii", d, off + 0x10)
            data = memoryview(d)[po:po + ps]
        else:
            data = memoryview(d)[off + 16:off + size]
        out.append(Chunk(B00P_NAMES.get(ty, "b00p_%d" % ty), major << 16 | minor, d[off:off + 32], data))
        off += size
    return out


def _read_rdf(d):
    off, size = struct.unpack_from("<qq", d, 16)
    out = []
    for i in range(size // 64):
        e = off + i * 64
        cid = d[e:e + 16].split(b"\0")[0].decode()
        compression = d[e + 16]
        version = struct.unpack_from("<I", d, e + 20)[0]
        ho, hs, do, ds, us = struct.unpack_from("<qqqqq", d, e + 24)
        if compression != 0:
            raise ValueError("zstd compressed RDF chunk %s is not supported" % cid)
        out.append(Chunk(cid, version, d[ho:ho + hs], memoryview(d)[do:do + ds]))
    return out


def read_chunks(path):
    with open(path, "rb") as f:
        d = f.read()
    if d[:4] == b"B00P":
        return "B00P", _read_b00p(d)
    if d[:8] == b"AMD_RDF ":
        return "AMD_RDF", _read_rdf(d)
    raise ValueError("%s is not an RGP capture (magic %r)" % (path, d[:8]))


def parse_asic(container, chunks):
    a = Asic()
    c = next((c for c in chunks if c.id == "AsicInfo"), None)
    if c is None:
        return a
    b = bytes(c.data)
    if container == "AMD_RDF":
        _, a.sclk_hz, _, a.timestamp_hz = struct.unpack_from("<QQQQ", b, 0)
        base = 48
    else:
        a.sclk_hz = struct.unpack_from("<Q", b, 0x08)[0]
        base = 0x18
    (a.device_id, _, a.vgprs_per_simd, _, a.shader_engines, a.cu_per_se, a.simd_per_cu, a.waves_per_simd, a.min_vgpr_alloc,
     a.vgpr_granularity) = struct.unpack_from("<10i", b, base)
    a.gfxip = struct.unpack_from("<i", b, base + 56)[0]
    m = re.search(rb"[ -~]{8,}", b)
    a.name = m.group().decode() if m else ""
    if not a.timestamp_hz:
        a.timestamp_hz = 100000000
    return a
