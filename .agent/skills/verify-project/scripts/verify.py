#!/usr/bin/env python3
"""Iterative project verification harness.

Runs several independent checks designed to surface bugs that a single test
pass can hide: syntax errors anywhere in the tree, import-time failures,
non-deterministic / order-dependent test failures, resource leaks and
deprecations (warnings promoted to errors), and — for projects that expose a
``ToolSelector`` — exception-safety under random/Unicode fuzzing.

Usage:
    python verify.py [--root DIR] [--tests DIR] [--iterations N] [--no-fuzz]

Defaults are auto-detected: the project root is the current directory (or its
nearest ancestor containing a ``tests`` directory), source packages are the
top-level directories containing an ``__init__.py``.

Exit code is 0 only when every axis passes, so this doubles as a CI gate.
"""
from __future__ import annotations

import argparse
import importlib
import io
import os
import py_compile
import random
import string
import sys
import unittest
import warnings


def banner(msg: str) -> None:
    print(f"\n{'=' * 70}\n{msg}\n{'=' * 70}")


def find_root(start: str) -> str:
    """Walk upward from *start* to the nearest dir containing ``tests/``."""
    cur = os.path.abspath(start)
    while True:
        if os.path.isdir(os.path.join(cur, "tests")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return os.path.abspath(start)
        cur = parent


def find_py(root: str) -> list[str]:
    out = []
    for dirpath, dirs, files in os.walk(root):
        # Skip hidden / vendored / cache dirs for speed and signal.
        dirs[:] = [d for d in dirs if d not in {".git", "__pycache__", ".venv", "venv", "node_modules"}]
        for f in files:
            if f.endswith(".py"):
                out.append(os.path.join(dirpath, f))
    return sorted(out)


def detect_packages(root: str) -> list[str]:
    """Top-level importable packages (dirs with __init__.py), excluding tests."""
    pkgs = []
    for name in sorted(os.listdir(root)):
        if name == "tests":
            continue
        if os.path.isfile(os.path.join(root, name, "__init__.py")):
            pkgs.append(name)
    return pkgs


# ----------------------------------------------------------------------------
# Axes
# ----------------------------------------------------------------------------

def check_compile(root: str, tests_dir: str) -> bool:
    banner("1) Syntax compile check (py_compile across the tree)")
    ok = True
    targets = []
    for pkg in detect_packages(root):
        targets += find_py(os.path.join(root, pkg))
    targets += find_py(tests_dir)
    for f in targets:
        try:
            py_compile.compile(f, doraise=True)
        except py_compile.PyCompileError as e:
            ok = False
            print(f"  FAIL {f}: {e}")
    print(f"  {len(targets)} files compiled — {'OK' if ok else 'FAILURES'}")
    return ok


def check_imports(root: str) -> bool:
    banner("2) Import check (every module in detected packages)")
    mods: list[str] = []
    for pkg in detect_packages(root):
        for f in find_py(os.path.join(root, pkg)):
            rel = os.path.relpath(f, root)[:-3].replace(os.sep, ".")
            if rel.endswith(".__init__"):
                rel = rel[: -len(".__init__")]
            mods.append(rel)
    ok = True
    for m in mods:
        try:
            importlib.import_module(m)
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"  FAIL import {m}: {type(e).__name__}: {e}")
    print(f"  {len(mods)} modules imported — {'OK' if ok else 'FAILURES'}")
    return ok


def _discover_test_names(tests_dir: str) -> list[str]:
    loader = unittest.TestLoader()
    suite = loader.discover(tests_dir, pattern="test_*.py")
    names: list[str] = []

    def walk(s):
        for item in s:
            if isinstance(item, unittest.TestSuite):
                walk(item)
            else:
                names.append(item.id())

    walk(suite)
    return names


def _run_suite(names: list[str], warnings_as_errors: bool) -> tuple[bool, int]:
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for n in names:
        suite.addTests(loader.loadTestsFromName(n))
    stream = io.StringIO()
    runner = unittest.TextTestRunner(stream=stream, verbosity=0)
    with warnings.catch_warnings():
        if warnings_as_errors:
            warnings.simplefilter("error", ResourceWarning)
            warnings.simplefilter("error", DeprecationWarning)
        result = runner.run(suite)
    if not result.wasSuccessful():
        print(stream.getvalue())
        for kind, lst in (("ERROR", result.errors), ("FAIL", result.failures)):
            for test, tb in lst:
                print(f"  {kind}: {test.id()}")
                print(tb)
    return result.wasSuccessful(), result.testsRun


def check_repeated_randomized(tests_dir: str, iterations: int) -> bool:
    banner(f"3) Repeated + randomized order ({iterations}x, fresh seed each run)")
    names = _discover_test_names(tests_dir)
    print(f"  discovered tests: {len(names)}")
    if not names:
        print("  (no tests found — skipping)")
        return True
    all_ok = True
    for i in range(iterations):
        rng = random.Random(i * 7919 + 13)
        shuffled = names[:]
        rng.shuffle(shuffled)
        ok, count = _run_suite(shuffled, warnings_as_errors=False)
        print(f"  iter {i + 1:>2}: {count} tests -> {'OK' if ok else 'FAIL'}")
        all_ok = all_ok and ok
    return all_ok


def check_warnings_as_errors(tests_dir: str) -> bool:
    banner("4) Warnings promoted to errors (ResourceWarning/DeprecationWarning)")
    names = _discover_test_names(tests_dir)
    if not names:
        print("  (no tests found — skipping)")
        return True
    ok, count = _run_suite(names, warnings_as_errors=True)
    print(f"  {count} tests, warnings-as-errors -> {'OK' if ok else 'FAIL'}")
    return ok


def check_fuzz_tool_selector(root: str, trials: int = 2000) -> bool:
    banner(f"5) Fuzz ToolSelector ({trials}x random + Unicode inputs)")
    try:
        for pkg in detect_packages(root):
            try:
                tool_mod = importlib.import_module(f"{pkg}.core.tool")
                sel_mod = importlib.import_module(f"{pkg}.core.tool_selector")
                break
            except Exception:
                continue
        else:
            print("  (no ToolSelector found — skipping)")
            return True
        ToolRegistry = tool_mod.ToolRegistry
        ToolSelector = sel_mod.ToolSelector
    except Exception as e:  # noqa: BLE001
        print(f"  (could not load ToolSelector: {e} — skipping)")
        return True

    rng = random.Random(42)
    alphabet = string.ascii_letters + string.digits + " ._!,가나다한글テスト"
    issues = 0
    for _ in range(trials):
        reg = ToolRegistry()
        for i in range(rng.randint(0, 6)):
            desc = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 14)))

            def _f(**kw):
                return "ok"

            reg.register(_f, name=f"t{i}", description=desc)
        top_k = rng.randint(0, 8)
        sel = ToolSelector(
            reg,
            top_k=top_k,
            min_score=rng.choice([0.0, 0.1, 0.5, 0.99]),
            fallback_to_all=rng.choice([True, False]),
        )
        q = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 14)))
        try:
            ranked = sel.rank(q)
            selected = sel.select(q)
            schemas = sel.select_schemas(q)
        except Exception as e:  # noqa: BLE001
            issues += 1
            print(f"  EXCEPTION {type(e).__name__}: {e} | q={q!r}")
            continue
        if len(selected) > top_k:
            issues += 1
            print(f"  INVARIANT top_k: {len(selected)} > {top_k}")
        if len(schemas) != len(selected):
            issues += 1
            print("  INVARIANT schema count mismatch")
        for _, s in ranked:
            if not (-1e-9 <= s <= 1.0 + 1e-9):
                issues += 1
                print(f"  INVARIANT score out of [0,1]: {s}")
        for t in selected:
            if reg.get(t.name) is None:
                issues += 1
                print(f"  INVARIANT selected tool missing from registry: {t.name}")
    print(f"  {trials} trials — issues: {issues}")
    return issues == 0


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None, help="Project root (default: auto-detect)")
    parser.add_argument("--tests", default=None, help="Tests directory (default: <root>/tests)")
    parser.add_argument("--iterations", type=int, default=10, help="Randomized-order runs")
    parser.add_argument("--fuzz-trials", type=int, default=2000)
    parser.add_argument("--no-fuzz", action="store_true", help="Skip the fuzz axis")
    args = parser.parse_args()

    root = os.path.abspath(args.root) if args.root else find_root(os.getcwd())
    tests_dir = os.path.abspath(args.tests) if args.tests else os.path.join(root, "tests")
    os.chdir(root)
    if root not in sys.path:
        sys.path.insert(0, root)

    print(f"Project root : {root}")
    print(f"Tests dir    : {tests_dir}")
    print(f"Packages     : {', '.join(detect_packages(root)) or '(none)'}")

    results = {
        "compile": check_compile(root, tests_dir),
        "imports": check_imports(root),
        "repeated_randomized": check_repeated_randomized(tests_dir, args.iterations),
        "warnings_as_errors": check_warnings_as_errors(tests_dir),
    }
    if not args.no_fuzz:
        results["fuzz_tool_selector"] = check_fuzz_tool_selector(root, args.fuzz_trials)

    banner("Summary")
    for k, v in results.items():
        print(f"  {k:<22}: {'PASS' if v else 'FAIL'}")
    overall = all(results.values())
    print(f"\n  OVERALL: {'ALL PASS' if overall else 'FAILURES'}")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
