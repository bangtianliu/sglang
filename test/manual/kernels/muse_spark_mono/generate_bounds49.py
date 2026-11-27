"""Derive packed-row diagnostic with guards at every actual payload access."""

import ast
import copy
from pathlib import Path

BASE = Path("python/sglang/kernels/plugin_kernels/amd/muse_spark/gfx1250")
DEST = Path("test/manual/kernels/muse_spark_mono/bounds49_device.py")


def function(path, name, new):
    t = ast.parse((BASE / path).read_text())
    fn = copy.deepcopy(
        next(n for n in t.body if isinstance(n, ast.FunctionDef) and n.name == name)
    )
    fn.name = new
    return fn


def args(fn, names, optional=False):
    for name in names:
        field, _, annotation = name.partition(":")
        fn.args.args.append(
            ast.arg(
                field, ast.parse(annotation, mode="eval").body if annotation else None
            )
        )
        if optional:
            fn.args.defaults.append(ast.Constant(None))


def call(text):
    return ast.parse(text, mode="eval").body


def main():
    header = '''"""Generated once-only packed-boundary offset diagnostic; not a selected kernel."""
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import scoped,sync
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.framework import vector_memory as vm
from sglang.kernels.plugin_kernels.amd.muse_spark.gfx1250.stages.boundary import _split_row,_join_row

@g.jit
def checked_ptr(Pointer,ID:gl.constexpr,Span:gl.constexpr,Desc,Records,Row,Mask):
    scalar:gl.constexpr=len(Pointer.shape)==0
    if scalar:
        address=gl.full((1,),0,gl.int64,gl.BlockedLayout([1],[32],[4],[0]))+Pointer.to(gl.int64)
    else:
        address=Pointer.to(gl.int64)
    low=gl.load(Desc+ID*2);high=gl.load(Desc+ID*2+1)
    active=gl.full_like(address,True).to(gl.int1)&Mask
    valid=(address>=low)&(address+Span<=high)
    flat_bad=(active&~valid).reshape((Pointer.numel,))
    count=gl.sum(flat_bad.to(gl.int32),0)
    slot=(Row*16+ID)*8
    gl.atomic_add(Records+slot+7,gl.sum(active.reshape((Pointer.numel,)).to(gl.int32),0).to(gl.int64),sem="relaxed",scope="gpu")
    if count>0:
        old=gl.atomic_cas(Records+slot,low*0,low*0+1,sem="relaxed",scope="gpu")
        if old==0:
            addr=address.reshape((Pointer.numel,))
            gl.store(Records+slot+1,gl.min(gl.where(flat_bad,addr,0x7fffffffffffffff),0))
            gl.store(Records+slot+2,gl.max(gl.where(flat_bad,addr,0),0))
            gl.store(Records+slot+3,low);gl.store(Records+slot+4,high)
            gl.store(Records+slot+5,Span);gl.store(Records+slot+6,count)
    if scalar:
        return gl.sum(gl.where(valid,address,low),0).to(Pointer.dtype)
    else:
        return gl.where(valid,address,low).to(Pointer.dtype)

@g.jit
def words_masked_checked(Pointer,Mask,ID:gl.constexpr,Desc,Records,Row):
    return scoped.load_words_masked(checked_ptr(Pointer,ID,4,Desc,Records,Row,Mask),Mask)
'''
    nodes = ast.parse(header).body
    for original, new in [
        ("load_tile_words", "load_words_checked"),
        ("store_tile_words", "store_words_checked"),
    ]:
        fn = function("framework/packed_memory.py", original, new)
        args(fn, ["ID:gl.constexpr", "Desc", "Records", "Row"])
        at = next(
            i
            for i, n in enumerate(fn.body)
            if isinstance(n, ast.AnnAssign)
            and isinstance(n.target, ast.Name)
            and n.target.id == "asm"
        )
        fn.body[at:at] = ast.parse(
            "ptr=checked_ptr(ptr,ID,16,Desc,Records,Row,valid)"
        ).body
        nodes.append(fn)
    fn = function(
        "framework/packed_memory.py", "store_bf16_vector", "store_bf16_checked"
    )
    args(fn, ["ID:gl.constexpr", "Desc", "Records", "Row"])
    for n in ast.walk(fn):
        if isinstance(n, ast.Call) and ast.unparse(n.func) == "store_tile_words":
            n.func = ast.Name("store_words_checked", ast.Load())
            n.args += [call(x) for x in ("ID", "Desc", "Records", "Row")]
    nodes.append(fn)
    fn = function("stages/boundary.py", "_parameter", "parameter_checked")
    args(fn, ["ID:gl.constexpr", "Desc", "Records", "Row"])
    for n in ast.walk(fn):
        if isinstance(n, ast.Call) and ast.unparse(n.func) == "gl.load":
            n.args[0] = call(
                "checked_ptr(Base+i,ID,2 if Base.dtype.element_ty==gl.bfloat16 else 4,Desc,Records,Row,i>=0)"
            )
    nodes.append(fn)
    fn = function("stages/boundary.py", "_residual_inputs", "residual_checked")
    args(fn, ["Desc", "Records"])
    for n in ast.walk(fn):
        if (
            isinstance(n, ast.Call)
            and ast.unparse(n.func) == "gl.inline_asm_elementwise"
        ):
            for idx, ID in enumerate((1, 2, 3)):
                n.args[2].elts[idx] = ast.Call(
                    ast.Name("checked_ptr", ast.Load()),
                    [
                        n.args[2].elts[idx],
                        ast.Constant(ID),
                        ast.Constant(16),
                        call("Desc"),
                        call("Records"),
                        call("row"),
                        call("q>=0"),
                    ],
                    [],
                )
    nodes.append(fn)
    fn = function("stages/boundary.py", "_norm_residual_scoped", "packed_checked")
    args(fn, ["Desc", "Records"], True)

    class Edit(ast.NodeTransformer):
        def visit_If(self, n):
            if ast.unparse(n.test) == "Trace is not None":
                return None
            return self.generic_visit(n)

        def visit_Call(self, n):
            self.generic_visit(n)
            name = ast.unparse(n.func)
            if name == "_parameter":
                ID = {"Post": 7, "Pre": 4, "Alpha": 2, "Beta": 3}[
                    ast.unparse(n.args[0])
                ]
                n.func = ast.Name("parameter_checked", ast.Load())
                n.args += [ast.Constant(ID), call("Desc"), call("Records"), call("row")]
            elif name == "_residual_inputs":
                n.func = ast.Name("residual_checked", ast.Load())
                n.args += [call("Desc"), call("Records")]
            elif name in (
                "load_tile_words",
                "scoped.load_words_masked",
                "store_tile_words",
                "store_bf16_vector",
            ):
                ID = (
                    1
                    if "Stream" in ast.unparse(n.args[0]) and name == "load_tile_words"
                    else 6
                    if name == "store_tile_words"
                    else 5
                    if name == "store_bf16_vector"
                    else 0
                )
                n.func = ast.Name(
                    {
                        "load_tile_words": "load_words_checked",
                        "scoped.load_words_masked": "words_masked_checked",
                        "store_tile_words": "store_words_checked",
                        "store_bf16_vector": "store_bf16_checked",
                    }[name],
                    ast.Load(),
                )
                n.args += [ast.Constant(ID), call("Desc"), call("Records"), call("row")]
            return n

    fn = Edit().visit(fn)
    nodes.append(fn)
    tree = ast.Module(nodes, type_ignores=[])
    ast.fix_missing_locations(tree)
    tmp = DEST.with_suffix(".tmp")
    tmp.write_text(ast.unparse(tree) + "\n")
    tmp.replace(DEST)


if __name__ == "__main__":
    main()
