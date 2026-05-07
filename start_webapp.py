#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mask_iteration_webapp.server import run_server
from mask_iteration_webapp.service import (
    MaskIterationService,
    Sam3InferenceService,
    SessionStore,
    UploadedTargetStore,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the packaged SAM3 mask iteration web app."
    )
    parser.add_argument("--host", default="127.0.0.1", help="HTTP host.")
    parser.add_argument("--port", type=int, default=8765, help="HTTP port.")
    parser.add_argument(
        "--imports-root",
        type=Path,
        default=PROJECT_ROOT / "runs" / "work_dataset",
        help="Directory where uploaded image and annotation working copies are maintained.",
    )
    parser.add_argument(
        "--sessions-root",
        type=Path,
        default=PROJECT_ROOT / "runs" / "sessions",
        help="Directory where session JSON and logits files are stored.",
    )
    parser.add_argument(
        "--sam3-repo-dir",
        type=Path,
        default=PROJECT_ROOT / "third_party" / "sam3",
        help="Local SAM3 repository root. Download SAM3 separately and point here.",
    )
    parser.add_argument(
        "--local-deps-dir",
        type=Path,
        default=PROJECT_ROOT / ".local_deps",
        help="Optional local dependency directory appended to PYTHONPATH.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Explicit SAM3 checkpoint path. If omitted, common checkpoint locations are searched.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Inference device. auto prefers cuda, then mps, then cpu.",
    )
    parser.add_argument(
        "--reference-box-expand-px",
        type=float,
        default=10.0,
        help="Pixels added to each side of bbox for prompt box.",
    )
    parser.add_argument(
        "--static-dir",
        type=Path,
        default=PROJECT_ROOT / "web" / "mask_iteration_app",
        help="Static frontend directory.",
    )
    parser.add_argument(
        "--validate-tools-dir",
        type=Path,
        default=None,
        help="Optional external Validate_tools directory for the review panel.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.static_dir.exists():
        raise FileNotFoundError(f"Missing static frontend dir: {args.static_dir}")

    target_store = UploadedTargetStore(args.imports_root.resolve())
    session_store = SessionStore(args.sessions_root.resolve())
    inference_service = Sam3InferenceService(
        project_root=PROJECT_ROOT,
        sam3_repo_dir=args.sam3_repo_dir.resolve(),
        local_deps_dir=args.local_deps_dir.resolve(),
        checkpoint=args.checkpoint.resolve() if args.checkpoint else None,
        device=args.device,
        reference_box_expand_px=float(args.reference_box_expand_px),
    )
    validate_tools_dir = args.validate_tools_dir.resolve() if args.validate_tools_dir else None
    service = MaskIterationService(
        target_store,
        session_store,
        inference_service,
        validate_tools_dir=validate_tools_dir,
    )

    readiness = inference_service.readiness()
    print(f"[bundle] work dataset root: {args.imports_root.resolve()}")
    print(f"[bundle] sessions root: {args.sessions_root.resolve()}")
    print(f"[bundle] SAM3 repo exists: {readiness['repo_exists']}")
    print(f"[bundle] SAM3 checkpoint: {readiness['checkpoint']}")
    print(f"[bundle] checkpoint exists: {readiness['checkpoint_exists']}")
    print(f"[bundle] requested device: {args.device}")
    print(f"[bundle] merged UI: http://{args.host}:{args.port}/merged.html")
    print(f"[bundle] root redirect: http://{args.host}:{args.port}/")

    run_server(service=service, host=args.host, port=args.port, static_dir=args.static_dir.resolve())


if __name__ == "__main__":
    main()
