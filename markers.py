"""
RGP driver markers. The driver writes them as dwords into SQ_THREAD_TRACE_USERDATA_2/3 (0xc342/0xc343, two dwords per
packet) and they arrive as REG tokens at CP time. Layouts match Mesa src/amd/common/ac_sqtt.h.
"""
import struct

USERDATA_2 = 0xc342
USERDATA_3 = 0xc343

EVENT_API = {
    0: "Draw", 1: "DrawIndexed", 2: "DrawIndirect", 3: "DrawIndexedIndirect", 4: "DrawIndirectCountAMD",
    5: "DrawIndexedIndirectCountAMD", 6: "Dispatch", 7: "DispatchIndirect", 8: "CopyBuffer", 9: "CopyImage", 10: "BlitImage",
    11: "CopyBufferToImage", 12: "CopyImageToBuffer", 13: "UpdateBuffer", 14: "FillBuffer", 15: "ClearColorImage",
    16: "ClearDepthStencilImage", 17: "ClearAttachments", 18: "ResolveImage", 19: "WaitEvents", 20: "PipelineBarrier",
    21: "ResetQueryPool", 22: "CopyQueryPoolResults", 23: "RenderPassColorClear", 24: "RenderPassDepthStencilClear",
    25: "RenderPassResolve", 26: "InternalUnknown", 27: "DrawIndirectCount", 28: "DrawIndexedIndirectCount", 30: "TraceRays",
    31: "TraceRaysIndirect", 32: "BuildAccelerationStructures", 33: "BuildAccelerationStructuresIndirect",
    34: "CopyAccelerationStructure", 35: "CopyAccelerationStructureToMemory", 36: "CopyMemoryToAccelerationStructure",
    41: "DrawMeshTasks", 42: "DrawMeshTasksIndirectCount", 43: "DrawMeshTasksIndirect",
}

USER_EVENT_KINDS = ("trigger", "pop", "push", "objname")


class Marker:
    __slots__ = ("kind", "t", "f")

    def __init__(self, kind, t, f):
        self.kind = kind
        self.t = t
        self.f = f

    def __repr__(self):
        return "%s@%d %s" % (self.kind, self.t, self.f)


def parse(dws):
    """dws = [(shader_clock, dword)] in stream order for one queue. Returns (markers, unknown_count)."""
    out = []
    i = 0
    n = len(dws)
    unknown = 0

    def dw(k):
        return dws[i + k][1] if i + k < n else 0

    while i < n:
        t, d = dws[i]
        ident = d & 0xf
        ext = (d >> 4) & 7
        f = {}
        if ident == 0:
            dims = d >> 31
            size = 3 + (3 if dims else 0) + ext
            api = (d >> 7) & 0xffffff
            f = {"api": EVENT_API.get(api, "Api%d" % api), "cb": dw(1) & 0xfffff, "cmd": dw(2)}
            if dims:
                f["dims"] = (dw(3), dw(4), dw(5))
            kind = "event"
        elif ident == 1:
            size = 4 + ext
            kind = "cb_start"
            f = {"cb": (d >> 7) & 0xfffff, "queue": d >> 27}
        elif ident == 2:
            size = 3 + ext
            kind = "cb_end"
            f = {"cb": (d >> 7) & 0xfffff}
        elif ident == 3:
            size = 2 + ext
            kind = "barrier_start"
            f = {"reason": dw(1) & 0x7fffffff, "internal": dw(1) >> 31}
        elif ident == 4:
            size = 2 + ext
            kind = "barrier_end"
            f = {"flags": d >> 27, "flags2": dw(1)}
        elif ident == 5:
            data_type = (d >> 12) & 0xff
            kind = USER_EVENT_KINDS[data_type] if data_type < 4 else "user%d" % data_type
            if data_type == 1:
                size = 1
            else:
                length = dw(1)
                nd = (length + 3) // 4
                size = 2 + nd
                s = b"".join(struct.pack("<I", x[1]) for x in dws[i + 2:i + 2 + nd])[:length]
                f = {"name": s.decode("utf-8", "replace")}
        elif ident == 6:
            size = 1 + ext
            kind = "general_api"
            f = {"api": (d >> 7) & 0xfffff, "end": (d >> 27) & 1}
        elif ident == 9:
            size = 2 + ext
            kind = "layout_transition"
        elif ident == 0xc:
            size = 3 + ext
            kind = "bind"
            f = {"point": (d >> 7) & 3, "hash": dw(1) | dw(2) << 32}
        else:
            size = 1
            kind = "unknown%d" % ident
            unknown += 1
        out.append(Marker(kind, t, f))
        i += size
    return out, unknown
