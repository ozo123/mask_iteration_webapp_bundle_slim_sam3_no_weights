# Development Guide

This repository packages a local SAM3 mask iteration web app and the related
validation tools. Large runtime files are intentionally kept outside Git.

## Local Setup

Create a virtual environment and install the lightweight project dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Install the correct `torch` build separately for the target machine.

Download the SAM3 checkpoint outside Git and place it here:

```text
third_party/sam3/checkpoints/sam3.pt
```

## Run The Web App

```bash
python start_webapp.py --sam3-repo-dir ./third_party/sam3 --checkpoint ./third_party/sam3/checkpoints/sam3.pt --device auto
```

Then open:

```text
http://127.0.0.1:8765/
```

## Lightweight Checks

The repository health check does not load SAM3 weights and does not require
`torch`. It verifies Python syntax for first-party tools, JSON files, and the
expected frontend entry points:

```bash
python scripts/check_repo.py
```

Run this before opening a pull request. The same command runs in GitHub Actions.

## GitHub Flow

Use short-lived branches with the `codex/` prefix for Codex-maintained work:

```bash
git switch -c codex/my-change
```

Keep generated files, checkpoints, local virtual environments, and uploaded
runtime data out of commits. Open a pull request into `main` after the health
check passes and the web workflow has been manually tested.
