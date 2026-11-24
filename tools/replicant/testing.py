"""Test helpers shared by selftest.py and the tests (pytest-compatible)."""


def slow(fn):
    """Mark a test as slow: selftest.py runs it only with --full."""
    fn.slow = True
    return fn


def imports(src):
    """Every import of a module source as a string: "module" for `import module`, and "<dots>module:name1:name2"
    for `from <dots>module import name1, name2`, so an audit sees the imported names too (`from .. import inference`
    is "..:inference", not just ".."; final review: one shared helper for the layering audits)."""
    import ast
    out = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.ImportFrom):
            out.add(("." * node.level) + (node.module or "") + "".join(f":{a.name}" for a in node.names))
        elif isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
    return out


def import_parts(entry):
    """The dotted and imported-name components of one `imports` string ("..lattice.bethe:x" -> lattice, bethe, x)."""
    return [x for x in entry.replace(":", ".").split(".") if x]
