#!/usr/bin/env python3
"""Unit tests for `profile_render.py` — pure Python, no pytest, no network.

    python3 scripts/test_profile_render.py

Why a test file at all: `modelctl` used to build its command line by string eval,
so a wrong argument was only visible when a model failed to start. Now that the
command line is *data*, the renderer is the one place a bug can silently corrupt
every profile — so the expansion rules and the conditional/`pre`/`stop` shapes
get asserted here instead of being discovered on the box.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
# Overridable so the same file can run from the box's scratch dir against an
# installed copy:  PROFILES_DIR=~/deploy-5060ti/profiles RENDER=~/deploy-5060ti/profile_render.py
PROFILES = pathlib.Path(os.environ.get("PROFILES_DIR", HERE.parent / "profiles"))
RENDER_PATH = pathlib.Path(os.environ.get("RENDER", HERE / "profile_render.py"))

CASES: list = []


def case(fn):
    CASES.append(fn)
    return fn


def load_renderer():
    spec = importlib.util.spec_from_file_location("profile_render", RENDER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R = load_renderer()


def eq(got, want, what):
    assert got == want, f"{what}: got {got!r}, want {want!r}"


# ── expansion ────────────────────────────────────────────────────────────────

@case
def test_expand_forms():
    env = {"A": "1", "EMPTY": "", "HOME": "/home/lcy"}
    eq(R.expand("$A", env), "1", "bare $VAR")
    eq(R.expand("${A}", env), "1", "braced")
    eq(R.expand("x${A}y", env), "x1y", "embedded")
    eq(R.expand("${MISSING}", env), "", "missing -> empty")
    eq(R.expand("${MISSING:-dflt}", env), "dflt", "missing -> default")
    eq(R.expand("${EMPTY:-dflt}", env), "dflt", "empty counts as unset, like ${VAR:-}")
    eq(R.expand("${A:-dflt}", env), "1", "set wins over default")
    eq(R.expand("no dollars here", env), "no dollars here", "passthrough")


@case
def test_expand_recursive_default():
    """The bug that made `${VLLM_BIN:-$HOME/vllm-current/bin/vllm}` render a literal `$HOME`."""
    env = {"HOME": "/home/lcy"}
    eq(R.expand("${VLLM_BIN:-$HOME/vllm-current/bin/vllm}", env),
       "/home/lcy/vllm-current/bin/vllm", "default is expanded too")
    eq(R.expand("${A:-${B:-${C:-deep}}}", env), "deep", "nested defaults")


@case
def test_expand_depth_cap():
    """A self-referential definition must terminate, not hang the renderer."""
    env = {"A": "${B}", "B": "${A}"}
    out = R.expand("${A}", env)          # must return, whatever it returns
    assert isinstance(out, str)


@case
def test_expand_pair_pattern():
    """`$HOME/.triton:/root/.triton` — the triton mount is written exactly like this."""
    eq(R.expand("$HOME/.triton:/root/.triton", {"HOME": "/home/lcy"}),
       "/home/lcy/.triton:/root/.triton", "var followed by a path join")


# ── conditions ───────────────────────────────────────────────────────────────

@case
def test_when_holds():
    env = {"SET": "1", "ZERO": "0", "EMPTY": "", "K": "10"}
    eq(R.when_holds("SET", env), True, "non-empty")
    eq(R.when_holds("ZERO", env), True, "'0' is non-empty text, so truthy")
    eq(R.when_holds("EMPTY", env), False, "empty")
    eq(R.when_holds("MISSING", env), False, "missing")
    eq(R.when_holds("KV_METRICS==1", env), False, "==1 requires the value")
    eq(R.when_holds("SET==1", env), True, "==1 matched")
    eq(R.when_holds("K==1", env), False, "==1 is exact, not a prefix")


# ── render ───────────────────────────────────────────────────────────────────

@case
def test_render_conditional_args():
    doc = {"profile": {"bin": "b", "args": [
        "always",
        {"when": "ON", "args": ["--on", "${ON}"]},
        {"when": "OFF", "args": ["--off"]},
        {"when": "N==1", "args": ["--flag"]},
    ]}}
    eq(R.render(doc, {"ON": "yes", "N": "1"})[1],
       ["always", "--on", "yes", "--flag"], "only satisfied conditions expand")
    eq(R.render(doc, {"N": "0"})[1], ["always"], "nothing satisfied")
    eq(R.render({"profile": {"bin": "b"}}, {})[1], [], "no args key at all")


@case
def test_render_bin_and_launch_command():
    eq(R.render({"profile": {"bin": "${B:-/usr/bin/x}", "args": []}}, {})[0],
       "/usr/bin/x", "bin expands with a default")
    doc = {"profile": {"bin": "b", "launch_command": "run  --a   'q u o'",
                       "args": ["ignored"]}}
    eq(R.render(doc, {})[1], ["run", "--a", "q u o"],
       "launch_command is shell-split and replaces args")


@case
def test_shell_list_skips_blanks_and_expands():
    profile = {
        "pre": ["echo ${A}", "   ", ""],
        "stop": ["docker rm -f '${NAME:-c}'"],
        "detect": ["docker inspect '${NAME:-c}'"],
        "kill": [],
    }
    eq(R.shell_list(profile, "pre", {"A": "1"}), ["echo 1"], "blanks dropped")
    eq(R.shell_list(profile, "stop", {}), ["docker rm -f 'c'"], "default applied")
    eq(R.shell_list(profile, "detect", {}), ["docker inspect 'c'"], "detect rendered")
    eq(R.shell_list(profile, "kill", {}), [], "absent key -> empty list")


# ── the shipped profiles ─────────────────────────────────────────────────────

def shipped():
    return sorted(PROFILES.glob("*.json"))


FAKE_ENV = {
    "HOME": "/home/lcy",
    "PATH": "/usr/bin:/bin",
    "MODELS": "/home/lcy/Models",
}


@case
def test_shipped_profiles_are_valid():
    files = shipped()
    assert files, f"no profiles found in {PROFILES}"
    for path in files:
        doc = json.loads(path.read_text(encoding="utf-8"))
        profile = doc["profile"]
        eq(doc.get("schema"), "modelctl/profile@1", f"{path.name}: schema")
        eq(profile["id"], path.stem, f"{path.name}: id must match the filename")
        assert doc.get("project", {}).get("work_dir"), f"{path.name}: work_dir"
        assert profile.get("health"), f"{path.name}: health"
        assert profile.get("log"), f"{path.name}: log"


@case
def test_shipped_profiles_render_clean():
    """Every shipped profile must render with nothing left unexpanded.

    A surviving `$` means an expansion the renderer cannot perform — exactly the
    class of bug that produced `--model /Merkyor-W4A4/...` (unset `MODELS`).
    """
    for path in shipped():
        doc = json.loads(path.read_text(encoding="utf-8"))
        bin_path, argv = R.render(doc, dict(FAKE_ENV))
        assert bin_path and not bin_path.startswith("$"), f"{path.name}: bin {bin_path!r}"
        assert argv, f"{path.name}: rendered no argv"
        for item in [bin_path, *argv]:
            assert "$" not in item, f"{path.name}: unexpanded variable in {item!r}"
        for key in ("pre", "stop", "detect", "kill"):
            for line in R.shell_list(doc["profile"], key, dict(FAKE_ENV)):
                assert "$" not in line, f"{path.name}: unexpanded variable in {key}: {line!r}"


@case
def test_shipped_profiles_render_clean_without_models():
    """Same, but with `MODELS` absent — the box sets it nowhere.

    `vllm-mtp` and `llama256` used to render `--model /Merkyor-W4A4/...` because
    their command was `MODEL=$MODELS/...` and nothing defines `MODELS`.
    """
    bare = {"HOME": "/home/lcy", "PATH": "/usr/bin:/bin"}
    for path in shipped():
        doc = json.loads(path.read_text(encoding="utf-8"))
        bin_path, argv = R.render(doc, dict(bare))
        for item in [bin_path, *argv]:
            assert "$" not in item, f"{path.name}: unexpanded variable in {item!r}"
            # a filesystem path must not start with a doubled slash — the fingerprint
            # of `${UNSET}/sub/path` collapsing to `/sub/path`
            if item.startswith("/"):
                assert not item.startswith("//"), f"{path.name}: doubled slash in {item!r}"


@case
def test_shipped_profiles_serve_the_canonical_id():
    """Invariant (AGENTS §11): port 8080 must answer as `Qwen3.8-27B-Q6-dual-5060ti`.

    The DSH provider `qwen-local` names that id, so a profile that serves
    something else breaks the harness the moment it is selected. `vllm-mtp` used
    to serve `qwen3.8-27b-merkyor` and `llama256` served the raw gguf path.
    """
    canonical = "Qwen3.8-27B-Q6-dual-5060ti"
    for path in shipped():
        doc = json.loads(path.read_text(encoding="utf-8"))
        _, argv = R.render(doc, dict(FAKE_ENV))
        flags = [f for f in ("--served-model-name", "--alias") if f in argv]
        assert flags, f"{path.name}: declares no served model id"
        for flag in flags:
            eq(argv[argv.index(flag) + 1], canonical, f"{path.name}: {flag}")


@case
def test_shipped_profiles_log_is_absolute():
    """`--log` feeds `mkdir -p $(dirname ...)` and the redirect target."""
    for path in shipped():
        doc = json.loads(path.read_text(encoding="utf-8"))
        log = R.expand(doc["profile"]["log"], dict(FAKE_ENV))
        assert log.startswith("/"), f"{path.name}: log is not absolute: {log!r}"
        assert log.endswith(".log"), f"{path.name}: log should be a .log file: {log!r}"


@case
def test_shipped_profiles_declare_no_duplicate_ids():
    ids = [json.loads(p.read_text(encoding="utf-8"))["profile"]["id"] for p in shipped()]
    eq(len(ids), len(set(ids)), "profile ids are unique")


def main() -> int:
    failures = []
    for fn in CASES:
        try:
            fn()
            print(f"  ok   {fn.__name__}")
        except AssertionError as error:
            failures.append((fn.__name__, error))
            print(f"  FAIL {fn.__name__}: {error}")
        except Exception as error:  # noqa: BLE001 - report anything, never mask
            failures.append((fn.__name__, error))
            print(f"  ERR  {fn.__name__}: {type(error).__name__}: {error}")
    print()
    if failures:
        print(f"FAILED: {len(failures)}/{len(CASES)}")
        return 1
    print(f"PASSED: {len(CASES)}/{len(CASES)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
