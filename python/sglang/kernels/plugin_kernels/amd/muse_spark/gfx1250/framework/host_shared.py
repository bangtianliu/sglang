"""Shared pinned host memory mapped into every rank's GPU (no GPU peer/P2P/IPC).

Each process maps the same /dev/shm file and registers it with hipHostRegister
(mapped + portable). GPU kernels then reach it over PCIe, like RCCL's SHM
transport. Device-side accesses must use SCOPE_SYS.
"""

import ctypes
import mmap
import os

import torch



def _load_hip():
    torch.cuda.init()
    with open("/proc/self/maps") as f:
        paths = {line.split()[-1] for line in f if "libamdhip64.so" in line}
    assert len(paths) == 1, paths  # the HIP runtime torch already uses
    return ctypes.CDLL(paths.pop())


_hip = None
_REGISTER_PORTABLE, _REGISTER_MAPPED = 1, 2


def _check(err, what):
    if err != 0:
        raise RuntimeError(f"{what} failed: hip error {err}")


def _place(path, nbytes, node):
    """Opt-in (MUSE_HOST_NUMA=0|1|i): set the shm object's NUMA policy (bind to a node,
    or interleave over nodes 0-1) and fault every page in before any rank maps it.
    A memory policy on this process's own allocation only."""
    numa = ctypes.CDLL("libnuma.so.1", use_errno=True)
    fd = os.open(path, os.O_RDWR)
    m = mmap.mmap(fd, nbytes)
    addr = ctypes.addressof((ctypes.c_char * nbytes).from_buffer(m))
    mode, mask = (3, 0b11) if node == "i" else (2, 1 << int(node))
    nodemask = ctypes.c_ulong(mask)
    rc = numa.mbind(ctypes.c_void_p(addr), ctypes.c_ulong(nbytes), ctypes.c_int(mode),
                    ctypes.byref(nodemask), ctypes.c_ulong(64), ctypes.c_uint(0))
    if rc != 0:
        raise OSError(ctypes.get_errno(), "mbind failed")
    ctypes.memset(addr, 0, nbytes)
    print(f"host pages for {path}: policy {node}", flush=True)


class HostShared:
    """`nbytes` of host memory shared by all local ranks, as a device int32 tensor view."""

    def __init__(self, name, nbytes, rank, barrier, keep=False):
        global _hip
        _hip = _hip or _load_hip()
        nbytes = (nbytes + mmap.PAGESIZE - 1) // mmap.PAGESIZE * mmap.PAGESIZE
        path = f"/dev/shm/{name}"
        if rank == 0:
            with open(path, "wb") as f:
                f.truncate(nbytes)
            node = os.environ.get("MUSE_HOST_NUMA", "i")  # interleave: MK.27
            if node and node != "none":
                _place(path, nbytes, node)
        barrier()
        self.fd = os.open(path, os.O_RDWR)
        self.map = mmap.mmap(self.fd, nbytes)
        buf = (ctypes.c_char * nbytes).from_buffer(self.map)
        self._buf = buf
        self.hptr = ctypes.addressof(buf)
        _check(_hip.hipHostRegister(ctypes.c_void_p(self.hptr), ctypes.c_size_t(nbytes),
                                    ctypes.c_uint(_REGISTER_PORTABLE | _REGISTER_MAPPED)),
               "hipHostRegister")
        dptr = ctypes.c_void_p()
        _check(_hip.hipHostGetDevicePointer(ctypes.byref(dptr), ctypes.c_void_p(self.hptr),
                                            ctypes.c_uint(0)), "hipHostGetDevicePointer")
        self.dptr = dptr.value
        self.nbytes = nbytes
        self.path = path
        barrier()
        if rank == 0 and not keep:
            os.unlink(path)  # all ranks hold it open now

    def zero(self):
        ctypes.memset(self.hptr, 0, self.nbytes)

    def tensor(self, offset_bytes, numel, dtype=torch.int32):
        """A device tensor aliasing host memory (no copy). Caller keeps `self` alive."""
        return _wrap(self.dptr + offset_bytes, numel, dtype)


class _Iface:
    def __init__(self, ptr, numel, dtype):
        self.__cuda_array_interface__ = dict(
            shape=(numel,), typestr={torch.int32: "<i4", torch.int64: "<i8",
                                     torch.float32: "<f4", torch.uint8: "|u1"}[dtype],
            data=(ptr, False), version=3)


def _wrap(ptr, numel, dtype):
    return torch.as_tensor(_Iface(ptr, numel, dtype), device="cuda")
