#!/usr/bin/env python3
"""Definition of done: the checks every agent (and CI) runs before calling work finished.

    python harness/check.py                 all checks
    python harness/check.py --fast          skip pytest
    python harness/check.py --base origin/main   also enforce immutability against a git ref

Each check prints PASS or FAIL with actionable detail. Exit code 1 if any check fails.
Only needs the standard library, plus pytest and nbformat (from `pip install -e ".[dev]"`).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness"))

MAX_FILE_BYTES = 5 * 1024 * 1024
# Imported documents are kept verbatim, so their internal links aren't checked.
LINK_CHECK_EXCLUDE = ("docs/research-plan/", "docs/production/", "docs/audits/2026-09-28_")
IMMUTABLE_PREFIXES = ("results/runs/",)
APPEND_ONLY_GLOB = re.compile(r"^docs/audits/\d{4}-\d{2}-\d{2}_.*\.md$")
SECRET_PATTERNS = [
    re.compile(r"ghp_[A-Za-z0-9]{36}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{40,}"),
    re.compile(r"sk-[A-Za-z0-9]{32,}"),
    re.compile(r"sk-ant-[A-Za-z0-9\-_]{20,}"),
    re.compile(r"AIza[0-9A-Za-z\-_]{35}"),
    re.compile(r"hf_[A-Za-z0-9]{30,}"),
    re.compile(r"-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r'"type":\s*"service_account"'),
]
SKILL_NAME = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout


def tracked_files() -> list[str]:
    try:
        out = git("ls-files", "--cached", "--others", "--exclude-standard")
        return [f for f in out.splitlines() if (ROOT / f).is_file()]
    except Exception:
        return [str(p.relative_to(ROOT)) for p in ROOT.rglob("*") if p.is_file() and ".git" not in p.parts]


# ---------------------------------------------------------------------------

def check_pytest() -> list[str]:
    r = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, capture_output=True, text=True)
    return [] if r.returncode == 0 else [(r.stdout + r.stderr).strip().splitlines()[-1] if (r.stdout + r.stderr).strip() else "pytest failed"]


def check_notebooks() -> list[str]:
    try:
        import nbformat
    except ImportError:
        return ["nbformat not installed (pip install -e '.[dev]')"]
    problems = []
    for path in sorted(glob.glob(str(ROOT / "notebooks/**/*.ipynb"), recursive=True)):
        name = os.path.relpath(path, ROOT)
        try:
            nb = nbformat.read(path, as_version=4)
            nbformat.validate(nb)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{name}: invalid notebook ({exc.__class__.__name__})")
            continue
        dirty = [i for i, c in enumerate(nb.cells) if c.cell_type == "code" and c.get("outputs")]
        if dirty:
            problems.append(f"{name}: outputs in code cells {dirty[:5]} (strip them; see notebook-hygiene skill)")
    return problems


def check_harness_sync() -> list[str]:
    r = subprocess.run([sys.executable, str(ROOT / "harness/sync.py"), "--check"], cwd=ROOT, capture_output=True, text=True)
    return [] if r.returncode == 0 else [l for l in r.stdout.splitlines() if l.startswith(("DRIFT", "STALE"))] or [r.stderr.strip()]


def check_skills() -> list[str]:
    import sync  # harness/sync.py

    problems = []
    for skill_md in sorted((ROOT / ".agents/skills").glob("*/SKILL.md")):
        meta = sync.frontmatter(skill_md.read_text(encoding="utf-8"))
        folder = skill_md.parent.name
        if meta.get("name") != folder:
            problems.append(f"{skill_md.relative_to(ROOT)}: frontmatter name {meta.get('name')!r} must equal folder {folder!r}")
        if not SKILL_NAME.match(folder) or len(folder) > 64:
            problems.append(f"{folder}: skill names must be lowercase-hyphenated, at most 64 chars")
        desc = meta.get("description", "")
        if not desc or len(desc) > 1024:
            problems.append(f"{folder}: description missing or longer than 1024 chars")
    return problems


def check_links(files: list[str]) -> list[str]:
    problems = []
    for f in files:
        if not f.endswith(".md") or f.startswith(LINK_CHECK_EXCLUDE):
            continue
        text = (ROOT / f).read_text(encoding="utf-8", errors="replace")
        text = re.sub(r"```.*?```", "", text, flags=re.S)
        for m in re.finditer(r"\]\(([^)\s#]+)(#[^)]*)?\)", text):
            target = m.group(1)
            if re.match(r"^[a-z]+:", target):
                continue
            if not (ROOT / os.path.dirname(f) / target).resolve().exists():
                problems.append(f"{f}: broken link -> {target}")
    return problems


def check_sizes_and_secrets(files: list[str]) -> list[str]:
    problems = []
    for f in files:
        p = ROOT / f
        size = p.stat().st_size
        if size > MAX_FILE_BYTES:
            problems.append(f"{f}: {size / 1e6:.1f} MB > 5 MB (keep on Drive; record in docs/DRIVE_INVENTORY.csv)")
        if size < 2_000_000 and p.suffix not in {".png", ".jpg", ".parquet", ".pdf"}:
            text = p.read_text(encoding="utf-8", errors="ignore")
            for pat in SECRET_PATTERNS:
                if pat.search(text):
                    problems.append(f"{f}: looks like a credential ({pat.pattern[:20]}…)")
    return problems


def check_immutability(base: str) -> list[str]:
    try:
        diff = git("diff", "--name-status", "-M", base)
    except subprocess.CalledProcessError:
        return [f"cannot diff against {base!r} (fetch it, or omit --base)"]
    problems = []
    for line in diff.splitlines():
        status, *paths = line.split("\t")
        if status.startswith("A"):
            continue
        old = paths[0]
        if old.startswith(IMMUTABLE_PREFIXES):
            problems.append(f"{status} {old}: run artefacts are immutable; add a new file or a docs/audits entry instead")
        elif APPEND_ONLY_GLOB.match(old):
            problems.append(f"{status} {old}: dated audits are append-only; write a new dated audit")
    return problems


# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fast", action="store_true", help="skip pytest")
    parser.add_argument("--base", help="git ref to enforce immutability against (e.g. origin/main)")
    args = parser.parse_args()

    files = tracked_files()
    checks = [
        ("harness in sync", check_harness_sync),
        ("skills well-formed", check_skills),
        ("notebooks valid, no outputs", check_notebooks),
        ("markdown links resolve", lambda: check_links(files)),
        ("no large files or secrets", lambda: check_sizes_and_secrets(files)),
    ]
    if not args.fast:
        checks.insert(0, ("unit tests", check_pytest))
    base = args.base or os.environ.get("HARNESS_BASE_REF")
    if base:
        checks.append((f"immutable paths unchanged vs {base}", lambda: check_immutability(base)))

    failed = 0
    for name, fn in checks:
        problems = fn()
        print(f"{'PASS' if not problems else 'FAIL'}  {name}")
        for p in problems:
            print(f"      - {p}")
        failed += bool(problems)
    print(f"\n{len(checks) - failed}/{len(checks)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
