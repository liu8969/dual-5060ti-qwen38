#!/usr/bin/env python3
"""argv/env parity check: rendered profile vs the live production process.

Ground truth is captured from /proc/<pid>: the full argv and the interesting
environment variables of the running vLLM. Phase 1 must produce the *same*
argv before anything is allowed to replace the running modelctl.

    parity_check.py /tmp/parity/live.argv /tmp/parity/live.env \
                    /home/lcy/deploy-5060ti/profiles/vllm-dflash.json
"""

import json
import os
import pathlib
import sys

SCRIPTS = pathlib.Path("/home/lcy/deploy-5060ti")
sys.path.insert(0, str(SCRIPTS))
import profile_render  # noqa: E402

EXPECT_ENV = [
    "HF_ENDPOINT",
    "PYTORCH_CUDA_ALLOC_CONF",
    "VLLM_ALLOW_LONG_MAX_MODEL_LEN",
    "VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE",
    "MAX_JOBS",
    "CUDA_HOME",
    "LD_LIBRARY_PATH",
]
INTENTIONAL = {
    "VLLM_BIN": "旧 modelctl 把 VLLM_BIN 一起注入子进程（vLLM 会警告 Unknown vLLM environment variable）；"
                "新方案只用它解析 bin，不再注入",
    "PATH": "旧 PATH 里 /usr/local/cuda-13.3/bin 被 systemd 单元和启动脚本各加了一次（重复）；新方案只加一次",
}
ENV_LAYER = pathlib.Path.home() / ".modelctl.env"


def load_layer(path: pathlib.Path) -> dict:
    out = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        if "=" in line:
            key, value = line.split("=", 1)
            out[key.strip()] = value.strip().strip('"').strip("'")
    return out


def read_env_file(path: pathlib.Path) -> dict:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if "=" in line:
            key, value = line.split("=", 1)
            out[key] = value
    return out


def main() -> int:
    argv_file = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/parity/live.argv")
    env_file = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "/tmp/parity/live.env")
    prof_file = pathlib.Path(sys.argv[3] if len(sys.argv) > 3
                            else "/home/lcy/deploy-5060ti/profiles/vllm-dflash.json")

    live = argv_file.read_text(encoding="utf-8").splitlines()
    live_bin, live_args = live[1], live[2:]
    live_env = read_env_file(env_file)

    env = dict(os.environ)
    env.update(load_layer(ENV_LAYER))
    # NOTE: deliberately NOT injecting VLLM_BIN. The runner does not set it
    # either, so this exercises the `${VLLM_BIN:-$HOME/...}` default branch —
    # hiding that path was exactly how the first bug slipped past this check.
    doc = json.loads(prof_file.read_text(encoding="utf-8"))
    bin_path, args = profile_render.render(doc, env)

    print(f"profile : {doc['profile']['id']}   (from {prof_file})")
    print(f"live    : argv {len(live)} 项（bin={live_bin}，args={len(live_args)}）")
    print(f"render  : bin={bin_path}  args={len(args)}")
    print()

    ok = True

    if bin_path != live_bin:
        print(f"✗ bin 不同:\n    live   {live_bin}\n    render {bin_path}")
        ok = False
    else:
        print(f"✓ bin 一致：{bin_path}")

    if args == live_args:
        print(f"✓ argv 逐字一致（{len(args)} 项）")
    else:
        ok = False
        print(f"✗ argv 不同（live {len(live_args)} vs render {len(args)}）")
        for i in range(max(len(args), len(live_args))):
            got = args[i] if i < len(args) else "<缺失>"
            want = live_args[i] if i < len(live_args) else "<多余>"
            if got != want:
                print(f"    第 {i+1} 项:\n      live   {want!r}\n      render {got!r}")
        diffs = [i for i in range(max(len(args), len(live_args)))
                 if (args[i] if i < len(args) else None) != (live_args[i] if i < len(live_args) else None)]
        print(f"    共 {len(diffs)} 项不同（只列首项）")

    print()
    print("env（profile 声明的键，对比运行中进程）:")
    rendered_env = {}
    for key, value in doc["profile"].get("env", {}).items():
        rendered_env[key] = profile_render.expand(str(value), env)
    for key in EXPECT_ENV:
        want = live_env.get(key)
        got = rendered_env.get(key)
        if got is None:
            print(f"  ? {key}: profile 未声明（live={want!r}）")
            continue
        mark = "✓" if got == want else "✗"
        if got != want:
            ok = False
        print(f"  {mark} {key}: render={got!r} live={want!r}")

    print()
    print("有意为之的差异（不是缺陷）:")
    for key, why in INTENTIONAL.items():
        print(f"  - {key}: {why}")

    print()
    print("PARITY: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
