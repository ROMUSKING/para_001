---
name: notebook-hygiene
description: Add, update or convert a Jupyter/Colab notebook in this repo so it is valid, output-free and correctly placed. Use when a notebook is added or edited, or when converting a Colab text export (markdown with ``` fenced code) back into .ipynb.
---

# Notebook hygiene

## Converting a Colab text export into `.ipynb`

Colab "download as text" exports are markdown with bare ``` fences around code cells. Convert them with:

```bash
python .agents/skills/notebook-hygiene/scripts/colab_text_to_ipynb.py <export.txt> <out.ipynb>
```

The script:

- validates the result with `nbformat`;
- syntax-checks every code cell, ignoring `%` and `!` lines, and reports failures by cell index;
- handles nested ```` ```json ```` blocks inside f-strings.

If a cell fails to parse because the export was truncated, don't repair it by guessing. Leave it as is and note it in `notebooks/README.md`.

## Adding or updating a notebook

1. **Placement.** Put it in the right stage folder (`notebooks/AGENTS.md`). Superseded versions go to `archive/`.
2. **Strip outputs:**
   ```python
   import nbformat
   nb = nbformat.read(p, 4)
   for c in nb.cells:
       if c.cell_type == "code":
           c.outputs = []
           c.execution_count = None
   nbformat.write(nb, p)
   ```
3. **Move shared logic out.** Anything reused goes into `src/adjointrwm/` with a test, and the notebook imports it.
4. **Record it.** Add a row to `notebooks/README.md` and a line to `CHANGELOG.md`. If the executed original is on Drive, add it to `docs/DRIVE_INVENTORY.csv`.
5. **Check.** `python harness/check.py` validates every notebook and rejects outputs.
