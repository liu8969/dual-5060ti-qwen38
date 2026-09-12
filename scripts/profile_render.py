#!/usr/bin/env python3
"""Render a modelctl profile JSON into an environment block and an argv.

Why this exists: today a profile lives in three places (an env string inside
modelctl's `case`, a dedicated `*-launch.sh`, and `~/.modelctl.env`). This
renders ONE data file — same field names as the SSH Manager's launch projects
(`work_dir` / `environment_command` / `profiles[].launch_command`) — into the
exact env + argv that used to come out of that trio.

Expansion: `${VAR}`, `${VAR:-default}`, `$VAR` are resolved from the current
environment. The caller is expected to have already sourced the override layer
(`~/.modelctl.env`) with `set -a`, so env still wins over the JSON defaults —
that is the A/B experiment channel and must keep working.

Conditional args: an item may be `{"when": "VAR"} `(truthy = non-empty) or
`{"when": "VAR==1"}`; it expands only when the condition holds, which is how
`KV_METRICS`/`SCHED_TOKENS`-style toggles stay reproducible.

Not every launch is one argv: a container profile needs work before and after
(`docker rm -f` the old name, tear the container down on stop). Those live in
`profile.pre` / `profile.stop` as shell strings and are rendered the same way.

    profile_render.py <profile.json> --argv-nul      # NUL-delimited argv
    profile_render.py <profile.json> --argv          # one item per line
    profile_render.py <profile.json> --env           # KEY=VAL lines
    profile_render.py <profile.json> --json          # {"bin":..., "args":[...]}
    profile_render.py <profile.json> --pre-nul       # pre-launch shell commands
    profile_render.py <profile.json> --stop-nul      # teardown shell commands
    profile_render.py <profile.json> --log           # expanded log path
"""

import argparse
import json
import os
import re
import sys

VAR = re.compile(
    r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}|\$([A-Za-z_][A-Za-z0-9_]*)")


def expand(text: str, env: dict, depth: int = 0) -> str:
    """Expand $VAR / ${VAR} / ${VAR:-default}, recursively.

    Recursion matters: a default may itself contain a variable
    (`${VLLM_BIN:-$HOME/vllm-current/bin/vllm}`), and the first pass only
    substitutes the literal default text. Depth-capped so a self-referential
    definition cannot loop.
    """
    if depth > 8 or "$" not in text:
        return text

    def repl(m):
        name = m.group(1) or m.group(3)
        default = m.group(2)
        value = env.get(name)
        if value is None or value == "":
            return default if default is not None else ""
        return value

    out = VAR.sub(repl, text)
    return out if out == text else expand(out, env, depth + 1)


def when_holds(spec: str, env: dict) -> bool:
    if "==" in spec:
        key, want = spec.split("==", 1)
        return str(env.get(key.strip(), "")) == want.strip()
    return bool(str(env.get(spec.strip(), "")).strip())


def render(doc: dict, env: dict):
    profile = doc["profile"]
    args = []
    for item in profile.get("args", []):
        if isinstance(item, dict):
            if when_holds(str(item.get("when", "")), env):
                args.extend(expand(a, env) for a in item.get("args", []))
        else:
            args.append(expand(str(item), env))

    launch_command = profile.get("launch_command")
    if launch_command:
        # SSH-Manager-compatible free-form escape hatch: shell-split it.
        import shlex
        args = shlex.split(expand(launch_command, env))

    return expand(str(profile["bin"]), env), args


def shell_list(profile: dict, key: str, env: dict) -> list:
    """`pre` / `stop`: shell strings, expanded but NOT word-split (they are
    handed to `eval`, exactly like project.environment_command)."""
    return [expand(str(item), env) for item in profile.get(key, []) if str(item).strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("profile")
    ap.add_argument("--argv-nul", action="store_true")
    ap.add_argument("--argv", action="store_true")
    ap.add_argument("--env", dest="env_out", action="store_true")
    ap.add_argument("--json", dest="json_out", action="store_true")
    ap.add_argument("--work-dir", action="store_true")
    ap.add_argument("--env-command", action="store_true")
    ap.add_argument("--log", action="store_true")
    ap.add_argument("--pre-nul", action="store_true")
    ap.add_argument("--stop-nul", action="store_true")
    ap.add_argument("--detect-nul", action="store_true")
    ap.add_argument("--kill-nul", action="store_true")
    args = ap.parse_args()

    with open(args.profile, encoding="utf-8") as fh:
        doc = json.load(fh)

    env = dict(os.environ)
    bin_path, argv = render(doc, env)

    if args.env_out:
        merged = {}
        for key, value in doc["profile"].get("env", {}).items():
            merged[key] = expand(str(value), env)
        for key, value in doc.get("project", {}).get("env", {}).items():
            merged.setdefault(key, expand(str(value), env))
        for key in sorted(merged):
            print(f"{key}={merged[key]}")
        return 0

    if args.work_dir:
        print(expand(str(doc.get("project", {}).get("work_dir", ".")), env))
        return 0

    if args.env_command:
        print(expand(str(doc.get("project", {}).get("environment_command", "")), env))
        return 0

    if args.log:
        print(expand(str(doc["profile"].get("log", "")), env))
        return 0

    list_out = {"pre": args.pre_nul, "stop": args.stop_nul,
                "detect": args.detect_nul, "kill": args.kill_nul}
    for key, wanted in list_out.items():
        if wanted:
            items = shell_list(doc["profile"], key, env)
            if items:
                sys.stdout.write("\0".join(items) + "\0")
            return 0

    if args.json_out:
        print(json.dumps({"bin": bin_path, "args": argv}, ensure_ascii=False))
        return 0

    if args.argv_nul:
        sys.stdout.write("\0".join([bin_path, *argv]) + "\0")
        return 0
    for item in [bin_path, *argv]:
        print(item)
    return 0


if __name__ == "__main__":
    sys.exit(main())
