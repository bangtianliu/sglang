"""gfx1250 hardware clusters for TLX kernels on stock Triton (MK.65).

A cluster is a group of consecutive workgroups that share one shader engine. Inside a cluster,
CTAs can exchange data with cluster-scope (SCOPE_SE) stores and loads around a cluster barrier:
no flag, no poll, no device-scope round trip (ROCm/ATOM #2894 does this through FlyDSL).

Triton's own ``num_ctas > 1`` also sets the cluster attribute, but it changes the tensor model
(a tensor is split over the cluster's CTAs), which the pinned TLX layouts reject. Here the
kernel is compiled as a normal one-CTA kernel and only two things are changed, at the LLVM
stage: the ``amdgpu-cluster-dims`` function attribute and the launch metadata's ``num_ctas``
(the driver then launches ``grid * N`` workgroups with the cluster launch attribute).

Use: ``tlx_cluster.clustered(kernel, N)`` once, then launch with ``kernel[(clusters,)]``.
In the kernel: ``rank = tl.program_id(0) % N`` and :func:`barrier`.
"""

import re

import triton
import triton.language as tl

_SIZES = {}  # kernel function name -> cluster size
_installed = False


def _install():
    global _installed
    if _installed:
        return
    from triton.backends.amd.compiler import HIPBackend

    original = HIPBackend.make_llir

    def make_llir(src, metadata, options):
        name = metadata.get("name")
        if name is None:  # the name is recorded later in the stage: read it from the module
            found = re.search(r"tt\.func public @(\w+)\(", str(src))
            name = found.group(1) if found else None
        n = _SIZES.get(name)
        if n:
            # after the TTGIR stages (compiled with one CTA per tensor): only the function
            # attribute and the launcher see the cluster
            metadata["num_ctas"] = n
        return original(src, metadata, options)

    HIPBackend.make_llir = staticmethod(make_llir)  # it is a staticmethod upstream
    _installed = True


def clustered(kernel, n: int):
    """Launch ``kernel`` (a triton.jit function) in hardware clusters of ``n`` CTAs.
    Set it before the kernel's first compile; the launch grid is then the cluster count."""
    assert 1 <= n <= 16, n
    _install()
    _SIZES[kernel.fn.__name__] = int(n)
    return kernel


@triton.jit
def barrier(dep, lane):
    """Cluster barrier, as FlyDSL's ``rocdl.cluster.cluster_barrier``: the CTA's waves join
    (workgroup barrier, after their own stores completed), ONE wave of the CTA signals the
    cluster barrier (id -3), and every wave waits on it. Signalling from every wave hangs.

    ``lane``: int32 thread index (element i on thread i, e.g. arange(0, 128) in a one-element-
    per-thread layout); ``dep`` (same shape) is returned unchanged to order later loads."""
    return tl.inline_asm_elementwise(
        "s_wait_storecnt 0x0\ns_barrier_signal -1\ns_barrier_wait -1\n"
        "v_cmp_gt_u32_e64 $1, 32, $2\ns_cmp_eq_u32 $1, 0\ns_cbranch_scc1 1f\n"
        "s_barrier_signal -3\n1:\ns_barrier_wait -3\nv_mov_b32 $0, $3",
        "=&v,=&s,v,v,~{scc}", [lane, dep], dtype=(tl.int32, tl.int32), is_pure=False, pack=1)[0]


@triton.jit
def store_words(ptr, value):
    """32-bit stores at cluster scope (SCOPE_SE). Returns zeros (a dependency token)."""
    return tl.inline_asm_elementwise(
        "global_store_b32 $1, $2, off scope:SCOPE_SE\ns_wait_xcnt 0x0\nv_mov_b32 $0, 0",
        "=&v,v,v", [ptr, value], dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def load_words(ptr):
    """32-bit loads at cluster scope (SCOPE_SE), complete on return."""
    return tl.inline_asm_elementwise(
        "global_load_b32 $0, $1, off scope:SCOPE_SE\ns_wait_loadcnt 0x0\ns_wait_xcnt 0x0",
        "=v,v", [ptr], dtype=tl.int32, is_pure=False, pack=1)
