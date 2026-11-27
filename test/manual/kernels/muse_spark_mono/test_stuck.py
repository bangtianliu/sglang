"""GPU0 diagnostics. --trap-child is invoked last and alone by the external harness."""

import ctypes
import json
import mmap
import sys
from pathlib import Path

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import stuck


@gluon.jit
def probe(Flag, RecordAddress, Out, TRAP: gl.constexpr):
    record = RecordAddress.to(gl.pointer_type(gl.int64))
    ready = stuck.wait_or_record(
        Flag,
        77,
        100_000,
        record,
        gl.program_id(0) + 3,
        gl.program_id(0) + 17,
        TRAP=TRAP,
    )
    gl.store(Out, ready)


def mapped_record(path):
    """Pin a file-backed host page so diagnostics survive a fatal child GPU exception."""
    file = open(path, "r+b")
    memory = mmap.mmap(file.fileno(), 4096)
    address = ctypes.addressof(ctypes.c_char.from_buffer(memory))
    hip = ctypes.CDLL("libamdhip64.so")
    hip.hipHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
    hip.hipHostRegister.restype = ctypes.c_int
    hip.hipHostGetDevicePointer.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
        ctypes.c_uint,
    ]
    hip.hipHostGetDevicePointer.restype = ctypes.c_int
    result = hip.hipHostRegister(ctypes.c_void_p(address), 4096, 2)
    if result:
        raise RuntimeError(f"hipHostRegister(mapped local host page): {result}")
    device = ctypes.c_void_p()
    result = hip.hipHostGetDevicePointer(
        ctypes.byref(device), ctypes.c_void_p(address), 0
    )
    if result:
        raise RuntimeError(f"hipHostGetDevicePointer: {result}")
    return file, memory, device.value


def main():
    assert torch.cuda.device_count() == 1, "expose GPU0 only"
    assert torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx1250")
    trap = "--trap-child" in sys.argv
    flag = torch.full((1,), 55, device="cuda", dtype=torch.int32)
    out = torch.ones(1, device="cuda", dtype=torch.int32)
    if len(sys.argv) > 1:
        file, memory, address = mapped_record(sys.argv[-1])
        host = torch.frombuffer(memory, dtype=torch.int64, count=8)
        record = None
    else:
        record = torch.zeros(8, device="cuda", dtype=torch.int64)
        address = record.data_ptr()
    print(
        json.dumps(
            dict(
                flag_address=flag.data_ptr(),
                tag=77,
                observed=55,
                rank=3,
                program_id=0,
                stage=17,
                trap=trap,
            )
        ),
        flush=True,
    )
    kernel = probe.warmup(flag, address, out, TRAP=trap, num_warps=8, grid=(1,))
    # Preserve trap assembly before launching a kernel that may abort the process.
    Path(f"/home/bangtliu/coord/logs/phase2.4-stuck-trap{trap}.amdgcn").write_text(
        kernel.asm["amdgcn"]
    )
    probe[(1,)](flag, address, out, TRAP=trap, num_warps=8)
    torch.cuda.synchronize()
    words = record.cpu().tolist() if record is not None else host.tolist()
    decoded = stuck.decode_record(words)
    assert (
        decoded.tag,
        decoded.observed,
        decoded.rank,
        decoded.program_id,
        decoded.stage,
    ) == (77, 55, 3, 0, 17)
    assert decoded.address == flag.data_ptr() and decoded.clock > 0
    assert not out.item()
    print(f"PASS diagnostic {decoded}", flush=True)
    if record is None:
        memory.flush()


if __name__ == "__main__":
    main()
