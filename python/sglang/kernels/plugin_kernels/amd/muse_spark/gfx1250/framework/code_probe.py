"""Measurement-only self-code L2 reads, inspired by TokenSpeed's code-warm probe.

No cache flush or executable-memory write. Host validates the exact ELF text
range; a runtime zero word count is the same-binary no-read control.
"""

import struct

from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl

MARKER = 0x4520C0DE


def text_info(kernel, marker=False):
    binary = kernel.asm["hsaco"]
    h = struct.unpack_from("<16sHHIQQQIHHHHHH", binary)
    sections = [
        struct.unpack_from("<IIQQQQIIQQ", binary, h[6] + i * h[11])
        for i in range(h[12])
    ]
    strings = sections[h[13]]
    names = binary[strings[4] : strings[4] + strings[5]]

    def name(section):
        return names[section[0] :].split(b"\0", 1)[0].decode()

    text = next(s for s in sections if name(s) == ".text")
    body = binary[text[4] : text[4] + text[5]]
    result = dict(text_bytes=len(body), hsaco_bytes=len(binary))
    if marker:
        needle = struct.pack("<I", MARKER)
        assert body.count(needle) == 1, body.count(needle)
        result.update(pc_offset=body.index(needle) - 4, words=len(body) // 4)
    return result, body


@g.jit
def warm(Control, task):
    pc, _ = gl.inline_asm_elementwise(
        "s_getpc_b64 $0\ns_mov_b32 $1, 0x4520c0de",
        "=s,=s",
        [],
        dtype=(gl.uint64, gl.uint32),
        is_pure=False,
        pack=1,
    )
    base = (pc - gl.load(Control)).to(gl.pointer_type(gl.int32), bitcast=True)
    # Four MiB cap makes the measurement loop finite even for malformed controls.
    words = gl.minimum(gl.load(Control + 1).to(gl.int32), 1048576)
    i = gl.arange(0, 512, layout=gl.BlockedLayout([4], [32], [4], [0]))
    acc = gl.full((512,), 0, gl.int32, i.type.layout)
    for block in range((task // 6) * 512, words, 2 * 512):
        at = block + i
        acc = acc ^ gl.load(base + at, at < words, other=0, cache_modifier=".cg")
    gl.store(Control + 2 + task, gl.xor_sum(acc, 0).to(gl.int64))
