---
name: verify-project
description: >-
  Iteratively test and verify the whole BaseAgent project for errors, bugs,
  regressions, and side effects. Runs a multi-axis harness — syntax compile,
  import checks, repeated randomized-order test runs (to catch flaky / order-
  dependent failures), warnings-promoted-to-errors (resource leaks &
  deprecations), and ToolSelector fuzzing. Use this whenever the user asks to
  "verify the project", "check for bugs", "run all the tests repeatedly", "make
  sure nothing is broken", "confirm no side effects/regressions", do a pre-commit
  or pre-PR sanity sweep, or validate that a change didn't break anything —
  even if they don't name the tests explicitly. Prefer this over a single plain
  `unittest` run whenever thoroughness or repeated/iterative checking is wanted.
---

# Verify Project

A repeatable, multi-axis verification sweep for this Python project. A single
`python -m unittest` pass is necessary but not sufficient — it misses
non-deterministic failures, inter-test state leakage, resource leaks, and
exception-safety gaps. This skill runs several complementary checks and, on
failure, drives a fix → re-verify loop until everything is green.

## When to use

Trigger on requests like "verify the project", "test everything repeatedly",
"any bugs or side effects?", "make sure my change didn't break anything", or a
pre-commit/pre-PR check. The goal is confidence that the whole tree is sound,
not just that one test file passes.

## How to run

The harness is bundled at `scripts/verify.py`. Run it from the project root (it
auto-detects the root, source packages, and the `tests/` directory):

```bash
python .claude/skills/verify-project/scripts/verify.py
```

Useful flags:
- `--iterations N` — number of randomized-order test runs (default 10).
- `--fuzz-trials N` — ToolSelector fuzz iterations (default 2000).
- `--no-fuzz` — skip the fuzz axis.
- `--root DIR` / `--tests DIR` — override auto-detection.

Exit code is 0 only when every axis passes, so the same command works as a CI
gate. For a quick smoke check use `--iterations 3 --fuzz-trials 500`; for a
thorough sweep use the defaults or higher.

## The five axes (and why each matters)

1. **Syntax compile** (`py_compile` across the tree) — catches syntax errors in
   files no test imports yet.
2. **Imports** — importing every module surfaces import-time errors, circular
   imports, and missing symbols that test discovery alone can miss.
3. **Repeated + randomized order** — runs the full suite several times, each
   with a different test order. Catches flaky tests and inter-test state
   leakage (a test that only passes because another ran first). Stable code is
   order-independent and deterministic.
4. **Warnings → errors** — promotes `ResourceWarning` and `DeprecationWarning`
   to failures. This is how unclosed files/sockets/DB connections and
   deprecated API usage get caught instead of being silently printed.
5. **ToolSelector fuzz** — feeds random and Unicode (incl. Korean/CJK) inputs to
   `ToolSelector` and asserts invariants: no exceptions, scores stay in `[0,1]`,
   `top_k` is respected, schema counts match, and every selected tool exists in
   the registry. Skipped automatically if the project has no `ToolSelector`.

## On failure: the fix → re-verify loop

This skill is iterative. When an axis fails:

1. Read the printed failure (traceback, failing test id, or invariant message).
2. Locate the root cause in the source — distinguish a real product bug from a
   stale/incorrect test. If the code contradicts what a test or comment claims,
   surface that rather than blindly "fixing" to green.
3. Apply a minimal, well-explained fix. For each bug fixed, add a regression
   test that fails before the fix and passes after, so it can't silently return.
4. Re-run the harness. Repeat until `OVERALL: ALL PASS`.

Report findings plainly: what was broken, why, the fix, and the regression
test. If a check was skipped (e.g., no ToolSelector), say so.

## Extending the harness

The fuzz axis is project-specific by design. When new components with
non-obvious input handling are added (parsers, tokenizers, scorers), add a
matching fuzz/invariant check to `scripts/verify.py` following the
`check_fuzz_tool_selector` pattern: build random + Unicode inputs, exercise the
component, and assert its invariants hold without exceptions.
```
