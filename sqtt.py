"""
gfx11 (SQTT 3.2) thread trace tokenizer.

Bit encodings come from AMD's MIT licensed rocprof-trace-decoder (source/gfx10/gfx10token.cpp plus the gfx11 overrides
in source/gfx11/gfx11token.cpp). The stream is read LSB first in 4 bit steps; the low 8 bits of the window select the
token, every token carries a delta that advances the shader clock, TIMESTAMP tokens with rt=1 pair the shader clock
with the 100 MHz GPU timestamp.
"""
import struct

TOKEN_NAMES = [
    "UNKNOWN", "EVENT", "EVENT_SYNC", "REG", "REG_INIT", "VALU_INST", "IMM_ONE", "IMMEDIATE", "WAVE_READY", "NEW_PC",
    "WAVE_END", "WAVE_START", "WAVE_START_EXT", "WAVE_ALLOC", "SHADER_DATA", "SHADER_DATA_SHORT", "UTIL_COUNTER",
    "TIME", "NOP", "MISC", "TIMESTAMP", "HEADER", "INST",
]
TOKEN = {n: i for i, n in enumerate(TOKEN_NAMES)}

ENCODINGS = [
    ("INST", 0b010, 3, 20, 4, 7), ("VALU_INST", 0b011, 3, 12, 3, 6), ("IMM_ONE", 0b1101, 4, 12, 4, 7),
    ("IMMEDIATE", 0b00100, 5, 24, 5, 8), ("WAVE_READY", 0b10100, 5, 24, 5, 8), ("NEW_PC", 0b0100001, 7, 64, 8, 11),
    ("WAVE_START", 0b01100, 5, 32, 5, 7), ("WAVE_START_EXT", 0b11100, 5, 48, 5, 7), ("WAVE_ALLOC", 0b00101, 5, 20, 5, 8),
    ("WAVE_END", 0b10101, 5, 20, 5, 8), ("SHADER_DATA", 0b00110, 5, 52, 5, 8), ("SHADER_DATA_SHORT", 0b10110, 5, 28, 5, 8),
    ("UTIL_COUNTER", 0b0110001, 7, 64, 7, 9), ("TIME", 0b1000, 4, 8, 4, 8), ("NOP", 0b0000, 4, 4, 0, 0),
    ("MISC", 0b1010001, 7, 24, 7, 16), ("EVENT", 0b01100001, 8, 24, 8, 11), ("EVENT_SYNC", 0b11100001, 8, 32, 8, 11),
    ("REG", 0b1001, 4, 64, 4, 7), ("REG_INIT", 0b1110001, 7, 64, 7, 10), ("TIMESTAMP", 0b0000001, 7, 64, 16, 64),
    ("HEADER", 0b0010001, 7, 64, 0, 0),
    ("MISC", 0b1010001, 7, 24, 7, 16), ("UTIL_COUNTER", 0b0110001, 7, 48, 7, 9), ("TIMESTAMP", 0b0000001, 7, 48, 12, 48),
]


def _build_table():
    tab = [(TOKEN["UNKNOWN"], 4, 0, 0)] * 256
    for name, pattern, plen, length, tbegin, tend in ENCODINGS:
        for i in range(pattern, 256, 1 << plen):
            tab[i] = (TOKEN[name], length, tbegin, (1 << (tend - tbegin)) - 1)
    return tab


TABLE = _build_table()


def tokenize(buf):
    """
    Returns (tokens, realtime, ok). tokens = [(type, shader_clock, raw64)], realtime = [(shader_clock, gpu_timestamp)].
    ok is False when an undefined token appears: RGP instruction-trace streams on the target SE carry a token missing from
    AMD's open gfx11 table, and nothing after it can be trusted.
    """
    b = bytes(buf) + b"\0" * 16
    nbits = len(buf) * 8
    unpack = struct.unpack_from
    tab = TABLE
    nop = TOKEN["NOP"]
    unknown = TOKEN["UNKNOWN"]
    time_tok = TOKEN["TIME"]
    ts_tok = TOKEN["TIMESTAMP"]
    out = []
    realtime = []
    append = out.append
    bp = 0
    t = 0
    while bp < nbits:
        lo, hi = unpack("<QQ", b, bp >> 3)
        w = ((hi << 64 | lo) >> (bp & 7)) & 0xFFFFFFFFFFFFFFFF
        ty, length, tbegin, mask = tab[w & 0xff]
        if ty == nop:
            bp += 4
            continue
        if ty == unknown:
            return out, realtime, False
        bp += length
        if length < 64:
            w &= (1 << length) - 1
        if ty == ts_tok:
            tm = (w >> 12) & 0xFFFFFFFFF
            if not (w >> 9) & 1:
                t += tm
            elif not (w >> 8) & 1:
                realtime.append((t, tm))
            continue
        if ty == time_tok:
            t += ((w >> tbegin) & mask) + 4
            continue
        t += (w >> tbegin) & mask
        append((ty, t, w))
    return out, realtime, True
