"""Convert Colab text exports (markdown + ``` fenced code) back into .ipynb."""
import ast, re, sys
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

def convert(text):
    cells, buf, in_code, nested = [], [], False, False
    def flush(kind):
        src = "\n".join(buf).strip("\n")
        if src.strip():
            cells.append(new_code_cell(src) if kind == "code" else new_markdown_cell(src))
        buf.clear()
    for line in text.splitlines():
        s = line.rstrip()
        if not in_code and s == "```":
            flush("md"); in_code = True; continue
        if in_code:
            if re.fullmatch(r"```[A-Za-z0-9_+-]+", s):
                nested = True; buf.append(line); continue
            if s == "```":
                if nested:
                    nested = False; buf.append(line); continue
                flush("code"); in_code = False; continue
        buf.append(line)
    flush("code" if in_code else "md")
    nb = new_notebook(cells=cells)
    nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                   "language_info": {"name": "python"}, "accelerator": "GPU",
                   "colab": {"provenance": []}}
    return nb

def check(nb):
    bad = 0
    for i, c in enumerate(nb.cells):
        if c.cell_type != "code": continue
        src = "\n".join(l for l in c.source.splitlines() if not l.lstrip().startswith(("%", "!")))
        try: ast.parse(src)
        except SyntaxError as e:
            bad += 1; print(f"  cell {i}: {e.msg} line {e.lineno}")
    return bad

if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    nb = convert(open(src).read())
    nbformat.validate(nb)
    n_code = sum(c.cell_type == "code" for c in nb.cells)
    print(dst, "cells", len(nb.cells), "code", n_code, "syntax-errors", check(nb))
    nbformat.write(nb, dst)
