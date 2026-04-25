#!/usr/bin/env python
from __future__ import annotations

import compileall
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_CHECK_DIRS = (
    PROJECT_ROOT / "mask_iteration_webapp",
    PROJECT_ROOT / "Validate_tools",
)
REQUIRED_FILES = (
    PROJECT_ROOT / "start_webapp.py",
    PROJECT_ROOT / "requirements.txt",
    PROJECT_ROOT / "README.md",
    PROJECT_ROOT / "web" / "mask_iteration_app" / "index.html",
    PROJECT_ROOT / "web" / "mask_iteration_app" / "index_merged.html",
    PROJECT_ROOT / "Validate_tools" / "rules.json",
)


def check_required_files() -> list[str]:
    errors: list[str] = []
    for path in REQUIRED_FILES:
        if not path.is_file():
            errors.append(f"Missing required file: {path.relative_to(PROJECT_ROOT)}")
    return errors


def check_json_files() -> list[str]:
    errors: list[str] = []
    for path in PROJECT_ROOT.glob("**/*.json"):
        if ".git" in path.parts:
            continue
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"Invalid JSON in {path.relative_to(PROJECT_ROOT)}: {exc}")
    return errors


def check_python_syntax() -> list[str]:
    errors: list[str] = []
    for path in PYTHON_CHECK_DIRS:
        if not path.exists():
            errors.append(f"Missing Python source directory: {path.relative_to(PROJECT_ROOT)}")
            continue
        ok = compileall.compile_dir(
            str(path),
            quiet=1,
            force=True,
            legacy=False,
            optimize=0,
        )
        if not ok:
            errors.append(f"Python syntax check failed under {path.relative_to(PROJECT_ROOT)}")

    ok = compileall.compile_file(str(PROJECT_ROOT / "start_webapp.py"), quiet=1, force=True)
    if not ok:
        errors.append("Python syntax check failed for start_webapp.py")
    return errors


def main() -> int:
    errors = []
    errors.extend(check_required_files())
    errors.extend(check_json_files())
    errors.extend(check_python_syntax())

    if errors:
        print("Repository health check failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Repository health check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
