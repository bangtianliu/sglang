"""Generate the measurement kernel mechanically from the accepted stage bodies."""

import ast
from pathlib import Path

BASE = Path(
    "python/sglang/kernels/plugin_kernels/amd/muse_spark/gfx1250/stages/layer_accepted.py"
)
DEST = Path("test/manual/kernels/muse_spark_mono/layer46.py")


def parse(s):
    return ast.parse(s).body


def main():
    tree = ast.parse(BASE.read_text())
    fn = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_launch"
    )
    fn.args.args.extend(
        [
            ast.arg("Control"),
            ast.arg("GridStorage"),
            ast.arg(
                "ONLY_STAGE", annotation=ast.parse("gl.constexpr", mode="eval").body
            ),
        ]
    )
    fn.args.defaults.extend([ast.Constant(None), ast.Constant(None), ast.Constant(0)])
    first = next(
        i
        for i, n in enumerate(fn.body)
        if isinstance(n, ast.Expr) and ast.unparse(n).startswith("stamp(Trace, pid, 0)")
    )
    head = fn.body[:first]
    body = fn.body[first:]
    chunks = []
    current = []
    for node in body:
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and ast.unparse(node.value.func) == "stamp"
            and isinstance(node.value.args[-1], ast.Constant)
            and node.value.args[-1].value in (2, 4, 6, 8, 10)
        ):
            chunks.append(current)
            current = []
        current.append(node)
    chunks.append(current)
    assert len(chunks) == 6

    class Rewrite(ast.NodeTransformer):
        def visit_If(self, n):
            self.generic_visit(n)
            if (
                isinstance(n.test, ast.Call)
                and ast.unparse(n.test.func) == "flow.arrive"
                and ast.unparse(n.test.args[0])
                in (
                    "Counters + group * 32",
                    "Counters + (8 + group) * 32",
                    "Counters + 12 * 32",
                )
            ):
                return ast.If(
                    ast.UnaryOp(ast.Not(), ast.Name("barrier_mode", ast.Load())),
                    [n],
                    [],
                )
            return n

        def visit_Expr(self, n):
            self.generic_visit(n)
            if (
                isinstance(n.value, ast.Call)
                and ast.unparse(n.value.func) == "scoped.publish_flags"
                and "flow.S5_INPUT" in ast.unparse(n.value.args[0])
            ):
                return ast.If(
                    ast.UnaryOp(ast.Not(), ast.Name("barrier_mode", ast.Load())),
                    [n],
                    [],
                )
            return n

        def visit_Call(self, n):
            self.generic_visit(n)
            name = ast.unparse(n.func)
            if name == "flow.wait":
                stage = n.args[10].value
                producer = n.args[11].value
                if producer < stage:
                    n.func = ast.Name("input_wait", ast.Load())
                    n.args.append(ast.Name("barrier_mode", ast.Load()))
            if name == "common_experts.stage":
                n.keywords.append(
                    ast.keyword(
                        "LayerPublish",
                        ast.UnaryOp(ast.Not(), ast.Name("barrier_mode", ast.Load())),
                    )
                )
            return n

    rewrite = Rewrite()
    head += parse(
        "mask=gl.load(Control)\nbarrier_mode=gl.load(Control+1)!=0\nfor s in gl.static_range(6):\n gl.store(Status+s*GRID+pid,1)"
    )
    result = head
    for i, chunk in enumerate(chunks):
        chunk = [rewrite.visit(n) for n in chunk]
        gate = ast.If(
            ast.parse(
                f"(ONLY_STAGE == 0) or (ONLY_STAGE == {i + 1})", mode="eval"
            ).body,
            [ast.If(ast.parse(f"mask & {1 << i} != 0", mode="eval").body, chunk, [])],
            [],
        )
        result.append(gate)
        if i < 5:
            result += parse(f"""if ONLY_STAGE == 0:
 if barrier_mode:
  ready, polls, _ = scoped.counter_barrier(GridStorage+{i}*((GRID//32+1+GRID)*32),tag,1000000,GRID,32,MAX_SPINS=65536)
  ok=ok & ready
  gl.store(Status+(7+{i})*GRID+pid,ready)
""")
    fn.body = result
    helper = ast.parse("""
@g.jit
def input_wait(Flag,tag,pid,task,Diag,Metrics,A,B,NA:gl.constexpr,NB:gl.constexpr,STAGE:gl.constexpr,PRODUCER:gl.constexpr,GRID:gl.constexpr,PREFETCH:gl.constexpr,WaitTrace,barrier_mode):
    ready=gl.full((),True,gl.int1,gl.BlockedLayout([1],[32],[4],[0]))
    if not barrier_mode:
        ready=flow.wait(Flag,tag,pid,task,Diag,Metrics,A,B,NA,NB,STAGE,PRODUCER,GRID,PREFETCH,WaitTrace)
    return ready
""").body
    tree.body = [*tree.body[:-1], *helper, fn] if tree.body[-1] is fn else tree.body
    assert tree.body[-1] is fn
    tree.body[0] = ast.Expr(
        ast.Constant(
            "Measurement-only runtime stage mask and grid-barrier control; generated from accepted layer."
        )
    )
    ast.fix_missing_locations(tree)
    DEST.write_text(ast.unparse(tree) + "\n")
    print(DEST)


if __name__ == "__main__":
    main()
