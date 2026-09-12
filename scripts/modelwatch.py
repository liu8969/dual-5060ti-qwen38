#!/usr/bin/env python3
"""Compat shim — keeps old callers of ~/deploy-5060ti/modelwatch.py working.

The monitor was extracted into its own project on 2026-09-12:

    ~/gpu-model-dashboard/          (repo: gpu-model-dashboard, entry point dashboard.py)

This file forwards argv to the real script. Set MODELWATCH_PY to point somewhere else.
"""
import os
import runpy
import sys

TARGET = os.environ.get("MODELWATCH_PY", os.path.expanduser("~/gpu-model-dashboard/dashboard.py"))
if not os.path.exists(TARGET):
    sys.exit(f"modelwatch.py has moved: {TARGET} not found (see ~/gpu-model-dashboard/README.md)")

sys.argv[0] = TARGET
runpy.run_path(TARGET, run_name="__main__")
