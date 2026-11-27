"""Debug-only lowered-LLVM guards for S6 global, buffer and LDS accesses.

Never used by an adopted kernel. Invalid accesses are redirected to a valid
scratch location and recorded per instruction/workgroup/lane; no traps.
"""

import re

V = r"[%@][-A-Za-z0-9_.$]+"


def width(ty):
    ty = ty.strip()
    m = re.match(r"<(\d+) x (.+)>", ty)
    if m:
        return int(m[1]) * width(m[2])
    m = re.match(r"i(\d+)", ty)
    if m:
        return (int(m[1]) + 7) // 8
    return {"half": 2, "bfloat": 2, "float": 4, "double": 8}[ty]


def dtype_prefix(s):
    return re.match(r"(<\d+ x [^>]+>|i\d+|half|bfloat|float|double)", s).group(1)


def guard_ir(text, bounds, shared, record_ptr, dummy_ptr):
    """bounds maps original pointer-argument numbers to [address, byte length]."""
    original = text.splitlines()
    definition = next(s for s in original if s.startswith("define amdgpu_kernel"))
    args = re.findall(r"ptr addrspace\(1\)[^,%]*(" + V + r")", definition)
    roots = {a: i for i, a in enumerate(args) if i in bounds}
    rsrc = {}
    for line in original:
        m = re.match(r"\s*(" + V + r") = ", line)
        if not m:
            continue
        result = m[1]
        rhs = line.split(" = ", 1)[1]
        if re.match(
            r"(?:getelementptr|addrspacecast|bitcast|ptrtoint|inttoptr|add |or |shufflevector|insertelement|extractelement|select )",
            rhs,
        ):
            dependencies = {roots[v] for v in re.findall(V, rhs) if v in roots}
            if len(dependencies) == 1:
                roots[result] = dependencies.pop()
        if "@llvm.amdgcn.make.buffer.rsrc." in line:
            p = re.search(r"ptr addrspace\(1\) (" + V + ")", line)
            assert p and p[1] in roots, line
            rsrc[result] = (p[1], roots[p[1]])
    n = 0
    sites = []
    out = []
    serial = 0

    def ident():
        nonlocal serial
        serial += 1
        return f"%t2dbg.{serial}"

    def emit(expr):
        name = ident()
        out.append(f"  {name} = {expr}")
        return name

    def check(address, size, root, space, source):
        nonlocal n
        site = n
        n += 1
        if space == 3:
            low, end = 0, shared
        else:
            assert root in bounds, (root, source)
            low, length = bounds[root]
            end = low + length
        a = emit(f"icmp uge i64 {address}, {low}")
        b = emit(f"icmp ule i64 {address}, {end - size}")
        valid = emit(f"and i1 {a}, {b}")
        bad = emit(f"xor i1 {valid}, true")
        bad64 = emit(f"zext i1 {bad} to i64")
        offset = emit(f"add i64 %t2dbg.gid, {site * 1024}")
        offset = emit(f"mul i64 {offset}, 32")
        ptr = emit(f"add i64 {offset}, {record_ptr}")
        ptr = emit(f"inttoptr i64 {ptr} to ptr addrspace(1)")
        for field, value in enumerate((bad64, address, str(root), str(size))):
            at = emit(f"getelementptr i64, ptr addrspace(1) {ptr}, i64 {field}")
            out.append(f"  store volatile i64 {value}, ptr addrspace(1) {at}, align 8")
        sites.append(dict(site=site, space=space, root=root, bytes=size, source=source))
        return valid

    for line in original:
        if line.startswith("define amdgpu_kernel"):
            # The diagnostic records addresses and writes non-argument memory.
            line = re.sub(
                r"\b(?:readonly|writeonly|readnone|nofree)\b|captures\(none\)", "", line
            )
            out.append(line)
            out += [
                "  %t2dbg.row = call i32 @llvm.amdgcn.workgroup.id.x()",
                "  %t2dbg.wave = call i32 @llvm.amdgcn.wave.id()",
                "  %t2dbg.wid = and i32 %t2dbg.wave, 3",
                "  %t2dbg.tid = call i32 @llvm.amdgcn.workitem.id.x()",
                "  %t2dbg.lane = and i32 %t2dbg.tid, 31",
                "  %t2dbg.wbase = shl i32 %t2dbg.wid, 5",
                "  %t2dbg.local = add i32 %t2dbg.wbase, %t2dbg.lane",
                "  %t2dbg.rbase = shl i32 %t2dbg.row, 7",
                "  %t2dbg.gid32 = add i32 %t2dbg.rbase, %t2dbg.local",
                "  %t2dbg.gid = zext i32 %t2dbg.gid32 to i64",
            ]
            continue
        if line.startswith("attributes #0"):
            line = re.sub(r"memory\([^)]*\)", "", line)
        if line.startswith(("declare", "!", "@", ";", "attributes")):
            out.append(line)
            continue
        # Invalid GEPs must remain observable rather than become LLVM poison.
        line = re.sub(r"\b(?:inbounds|nuw|nsw)\s+", "", line)
        if "@llvm.amdgcn.raw.ptr.buffer." in line:
            assert re.search(r"raw.ptr.buffer.(load|store)\.", line), line
            resource = re.search(r"ptr addrspace\(8\) (" + V + ")", line)
            assert resource and resource[1] in rsrc, line
            base, root = rsrc[resource[1]]
            m = re.search(
                r"ptr addrspace\(8\) "
                + re.escape(resource[1])
                + r", i32 ("
                + V
                + r"|-?\d+), i32 ("
                + V
                + r"|-?\d+)",
                line,
            )
            assert m, line
            bits = re.search(
                r"raw.ptr.buffer.(?:load|store)\.((?:v\d+)?[if]\d+)", line
            )[1]
            k = re.fullmatch(r"(?:v(\d+))?[if](\d+)", bits)
            size = int(k[1] or 1) * ((int(k[2]) + 7) // 8)
            address = emit(f"ptrtoint ptr addrspace(1) {base} to i64")
            for operand in m.groups():
                v = emit(f"zext i32 {operand} to i64")
                address = emit(f"add i64 {address}, {v}")
            valid = check(address, size, root, 1, line)
            # Triton's resource range is INT32_MAX-1; the standard AMD mask
            # offset 0x80000000 is therefore guaranteed out of range.
            assert "i64 2147483646" in next(
                x for x in original if x.strip().startswith(resource[1] + " =")
            )
            vo = emit(f"select i1 {valid}, i32 {m[1]}, i32 -2147483648")
            so = emit(f"select i1 {valid}, i32 {m[2]}, i32 0")
            replacement = f"ptr addrspace(8) {resource[1]}, i32 {vo}, i32 {so}"
            line = line[: m.start()] + replacement + line[m.end() :]
        elif "asm sideeffect" in line and re.search(r"global_(?:load|store)_", line):
            size = max(
                int(x) // 8 for x in re.findall(r"global_(?:load|store)_b(\d+)", line)
            )
            pointers = list(re.finditer(r"ptr addrspace\(1\) (" + V + ")", line))
            assert pointers, line
            replacements = []
            for p in pointers:
                # S6's only VMEM inline asm reads Branch/P. Gluon may
                # round-trip its pointer integers through LDS, losing SSA roots.
                if p[1] not in roots:
                    assert "global_load_b128" in line and "global_store_" not in line
                    roots[p[1]] = 0
                address = emit(f"ptrtoint ptr addrspace(1) {p[1]} to i64")
                valid = check(address, size, roots[p[1]], 1, line)
                safe = emit(
                    f"select i1 {valid}, ptr addrspace(1) {p[1]}, ptr addrspace(1) inttoptr (i64 {dummy_ptr} to ptr addrspace(1))"
                )
                replacements.append((p.start(1), p.end(1), safe))
            for start, end, safe in reversed(replacements):
                line = line[:start] + safe + line[end:]
        elif (
            re.search(r"\b(?:load|store)\b", line)
            and re.search(r"ptr addrspace\([13]\)", line)
            and not line.startswith(("declare", "!"))
        ):
            p = re.search(r"ptr addrspace\(([13])\) (" + V + ")", line)
            if not p:
                raise AssertionError(line)
            space = int(p[1])
            pointer = p[2]
            m = re.search(
                r"\b(load|store)\s+(?:volatile\s+)?(.+?), ptr addrspace", line
            )
            assert m, line
            size = width(dtype_prefix(m[2]))
            root = -1 if space == 3 else roots.get(pointer)
            address = emit(
                f"ptrtoint ptr addrspace({space}) {pointer} to i{32 if space == 3 else 64}"
            )
            if space == 3:
                address = emit(f"zext i32 {address} to i64")
            valid = check(address, size, root, space, line)
            safeaddr = (
                "null"
                if space == 3
                else f"inttoptr (i64 {dummy_ptr} to ptr addrspace(1))"
            )
            safe = emit(
                f"select i1 {valid}, ptr addrspace({space}) {pointer}, ptr addrspace({space}) {safeaddr}"
            )
            line = line[: p.start(2)] + safe + line[p.end(2) :]
        out.append(line)
    assert n > 0
    kinds = {s["space"] for s in sites}
    assert kinds == {1, 3}, kinds
    for name in ("workgroup.id.x", "workitem.id.x", "wave.id"):
        if not re.search(r"declare .*@llvm.amdgcn." + re.escape(name) + r"\(", text):
            out.append(f"declare i32 @llvm.amdgcn.{name}()")
    return "\n".join(out) + "\n", sites
