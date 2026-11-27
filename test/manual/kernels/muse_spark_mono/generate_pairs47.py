"""Derive pair trials from common bodies; only boundary memory scopes change."""

import ast
import builtins
import copy
from pathlib import Path

ROOT = Path("python/sglang/kernels/plugin_kernels/amd/muse_spark/gfx1250/stages")
DEST = Path("test/manual/kernels/muse_spark_mono/pairs47_device.py")


def source(module, name):
    tree = ast.parse((ROOT / f"{module}.py").read_text())
    return copy.deepcopy(
        next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    )


def qualify(fn, module, rename, changes=None):
    fn.name = rename.pop("__name__")
    locals_ = {
        n.id
        for n in ast.walk(fn)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
    } | {a.arg for a in fn.args.args + fn.args.kwonlyargs}
    keep = {"gl", "g", "agent_store", "decision_enabled"} | set(dir(builtins))

    class Edit(ast.NodeTransformer):
        def visit_Call(self, n):
            self.generic_visit(n)
            if changes is not None:
                n = changes(n)
            return n

        def visit_Name(self, n):
            if isinstance(n.ctx, ast.Load):
                if n.id in rename:
                    return ast.Name(rename[n.id], ast.Load())
                if n.id not in locals_ | keep:
                    return ast.Attribute(ast.Name(module, ast.Load()), n.id, ast.Load())
            return n

    result = Edit().visit(fn)
    ast.fix_missing_locations(result)
    return result


def stores(n):
    if ast.unparse(n.func) == "gl.store":
        ptr = n.args[0]
        first = ptr
        while isinstance(first, ast.BinOp) and isinstance(first.op, ast.Add):
            first = first.left
        if isinstance(first, ast.Name) and first.id in {
            "Q",
            "K",
            "V",
            "G",
            "cache",
            "Out",
            "NewStream",
        }:
            base = first.id

            class Offset(ast.NodeTransformer):
                def visit_Name(self, x):
                    return ast.Constant(0) if x.id == base else x

            offset = Offset().visit(copy.deepcopy(ptr))
            value = n.args[1]
            mask = n.args[2] if len(n.args) > 2 else ast.Constant(True)
            return ast.Call(
                ast.Name("agent_store", ast.Load()),
                [ast.Name(base, ast.Load()), offset, value, mask],
                [],
            )
    return n


def main():
    header = '''"""Measurement-only adjacent common-body fusions, preserving row layouts."""
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import scoped,chain_guard,code_probe
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import qkvg as qorig,boundary as borig,streamed_gemm as gemm
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages import common_qkvg as a1,common_attention as a2,common_attn_tail as a3,common_route as a4,common_experts as a5,common_tail as a6

@g.jit
def agent_store(Base,offset,Value,Mask):
    valid=gl.full_like(offset,True).to(gl.int1)&Mask
    gl.amd.gfx1250.buffer_store(Value.to(Base.dtype.element_ty),Base,offset.to(gl.int32),valid,cache=".cg")

@g.jit
def decision_enabled(Decision,STAGE:gl.constexpr):
    failed=scoped.load_flag(Decision+gl.program_id(0)*32)
    return (failed==0)|(failed==STAGE)

@g.jit
def boundary_barrier(Storage,tag,Failure,Decision,GRID:gl.constexpr):
    pid=gl.program_id(0)
    scoped.drain_stores()
    ticket=gl.atomic_add(Storage+(pid//32)*32,1,sem="relaxed",scope="gpu")
    if ticket+1==tag*32:
        top=gl.atomic_add(Storage+(GRID//32)*32,1,sem="relaxed",scope="gpu")
        if top+1==tag*(GRID//32):
            # One reader, private fanout: never send all CTAs to one DEV-scope line.
            failed=scoped.load_flag(Failure)
            i=gl.arange(0,1024,layout=gl.BlockedLayout([1],[32],[4],[0]))
            scoped.store_words(Decision+i*32,gl.full_like(i,failed),i<GRID)
            scoped.publish_flags(Storage+(GRID//32+1)*32,tag,GRID)
    seen,polls=scoped.wait_flag(Storage+(GRID//32+1+pid)*32,tag,100000000,MAX_SPINS=1048576,SHADER_CLOCK=True)
    return seen==tag,polls
'''
    tree = ast.parse(header)
    nodes = []
    nodes.append(
        qualify(
            source("qkvg", "head_epilogue"), "qorig", {"__name__": "head_agent"}, stores
        )
    )
    nodes.append(
        qualify(
            source("boundary", "norm_residual"),
            "borig",
            {"__name__": "boundary_agent"},
            stores,
        )
    )

    def project(n):
        if ast.unparse(n.func) == "gemm.partial_tile":
            n.args[8] = ast.Constant(True)
        return n

    nodes.append(
        qualify(
            source("streamed_gemm", "project_last"),
            "gemm",
            {"__name__": "project_agent"},
            project,
        )
    )
    nodes.append(
        qualify(
            source("common_qkvg", "stage"),
            "a1",
            {"__name__": "stage1", "head_epilogue": "head_agent"},
        )
    )

    def attention(n):
        if ast.unparse(n.func) == "a2.partial":
            n.keywords.append(ast.keyword("SCOPED_IO", ast.Constant(True)))
        return n

    nodes.append(
        qualify(
            source("common_attention", "stage"), "a2", {"__name__": "stage2"}, attention
        )
    )
    nodes.append(
        qualify(
            source("common_attn_tail", "stage"),
            "a3",
            {"__name__": "stage3", "norm_residual": "boundary_agent"},
        )
    )
    nodes.append(
        qualify(
            source("common_route", "stage"),
            "a4",
            {"__name__": "stage4", "project_last": "project_agent"},
        )
    )
    nodes.append(
        qualify(
            source("common_tail", "stage"),
            "a6",
            {"__name__": "stage6", "project_last": "project_agent"},
        )
    )
    schema = {}
    for s, module in enumerate(
        ("qkvg", "attention", "attn_tail", "route", "experts", "tail"), 1
    ):
        fn = source("common_" + module, "_launch")
        runtime = []
        constants = []
        defaults = {
            arg.arg: ast.literal_eval(v)
            for arg, v in zip(fn.args.args[-len(fn.args.defaults) :], fn.args.defaults)
        }
        for arg in fn.args.args:
            if arg.annotation is not None:
                constants.append(arg.arg)
        # All positional tensors precede GRID. Remaining unannotated args are optional.
        start = next(i for i, x in enumerate(fn.args.args) if x.arg == "GRID")
        pos = [x.arg for x in fn.args.args[:start]]
        runtime = [x.arg for x in fn.args.args[start:] if x.annotation is None]

        if s % 2 == 0:
            fn.args.args.append(ast.arg("Decision"))
            fn.args.defaults.append(ast.Constant(None))
        rename = {"__name__": f"root{s}"}
        if s != 5:
            rename["stage"] = f"stage{s}"

        def rootedit(n):
            if (
                s % 2 == 0
                and ast.unparse(n.func) == "a" + str(s) + ".chain_guard.enabled"
            ):
                n.func = ast.Name("decision_enabled", ast.Load())
                n.args[0] = ast.Name("Decision", ast.Load())
            return n

        fn = qualify(fn, "a" + str(s), rename, rootedit)
        nodes.append(fn)
        schema[s] = dict(
            positional=pos, constants=constants, runtime=runtime, defaults=defaults
        )
    for pair in ((1, 2), (3, 4), (5, 6)):
        first, last = pair
        sa, sb = schema[first], schema[last]

        def call(s, letter):
            opts = [
                f"{name}=C{letter}[{i}]"
                for i, name in enumerate(schema[s]["constants"])
            ] + [
                f"{name}=R{letter}[{i}]" for i, name in enumerate(schema[s]["runtime"])
            ]
            return (
                f"root{s}(*{letter},"
                + ",".join(opts)
                + (",Decision=Decision" if s % 2 == 0 else "")
                + ")"
            )

        text = f"""\n@g.jit
def pair{first}{last}(A,CA:gl.constexpr,RA,B,CB:gl.constexpr,RB,Epochs,Sync,Status,Failure,Decision,GRID:gl.constexpr,CodeWarm=None):
    pid=gl.program_id(0)
    if CodeWarm is not None:
        if (pid>=740)&(pid<752):code_probe.warm(CodeWarm,pid-740)
    tag=qorig._advance_tag(Epochs,pid)
    {call(first, "A")}
    ready,polls=boundary_barrier(Sync,tag,Failure,Decision,GRID)
    gl.store(Status+pid,ready)
    if not ready:
        scoped.publish(Decision+pid*32,{first})
        old=gl.atomic_cas(Failure,0,{first},sem="relaxed",scope="gpu")
        if old==0:gl.store(Failure+1,pid)
    {call(last, "B")}
"""
        nodes += ast.parse(text).body
    tree.body += nodes
    ast.fix_missing_locations(tree)
    DEST.write_text(ast.unparse(tree) + "\n")
    import json

    DEST.with_suffix(".schema.json").write_text(json.dumps(schema, indent=2) + "\n")
    print(DEST)


if __name__ == "__main__":
    main()
