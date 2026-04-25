from __future__ import annotations

import base64
import contextlib
import importlib
import importlib.util
import json
import os
import sys
import threading
from copy import deepcopy
from io import BytesIO
from math import hypot
from pathlib import Path
from typing import Any
from uuid import uuid4

from PIL import Image, ImageDraw

from .dataset import expand_box_xyxy, round_floats, sanitize_component
from .models import (
    HistoryRecord,
    LineStrokeRecord,
    LockedRegionRecord,
    PointRecord,
    SessionState,
    StrokePointRecord,
    TargetRecord,
    copy_locked_regions,
    copy_line_strokes,
    copy_points,
    utc_now_iso,
)

DEFAULT_VALIDATE_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_VALIDATE_MODEL = "Qwen/Qwen3-VL-235B-A22B-Instruct"
VALIDATE_API_KEY_ENV_VARS = ("SILICONFLOW_API_KEY", "OPENAI_API_KEY", "DASHSCOPE_API_KEY")
VALIDATE_REVIEW_MODE_ORIGINAL = "original_image_box"
VALIDATE_REVIEW_MODE_CROP_BOX2X = "crop_box2x"
DEFAULT_VALIDATE_REVIEW_MODE = VALIDATE_REVIEW_MODE_CROP_BOX2X


class SessionStore:
    def __init__(self, sessions_root: Path):
        self.sessions_root = sessions_root
        self.sessions_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def session_dir(self, target_key: str) -> Path:
        return self.sessions_root / sanitize_component(target_key)

    def session_json_path(self, target_key: str) -> Path:
        return self.session_dir(target_key) / "session.json"

    def image_deletions_path(self) -> Path:
        return self.sessions_root / "image_deletions.json"

    def load_session(self, target_key: str) -> SessionState | None:
        path = self.session_json_path(target_key)
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return SessionState.from_dict(payload)

    def save_session(self, session: SessionState) -> None:
        path = self.session_json_path(session.session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            path.write_text(json.dumps(session.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    def save_logits(self, session: SessionState, history_id: str, logits: Any) -> str:
        np = importlib.import_module("numpy")
        relpath = Path("artifacts") / "logits" / f"{sanitize_component(history_id)}.npy"
        output_path = self.session_dir(session.session_id) / relpath
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, logits)
        return str(relpath).replace("\\", "/")

    def load_logits(self, session: SessionState, relpath: str) -> Any:
        np = importlib.import_module("numpy")
        logits_path = self.session_dir(session.session_id) / relpath
        if not logits_path.exists():
            raise FileNotFoundError(f"Missing logits file for session {session.session_id}: {logits_path}")
        return np.load(logits_path, allow_pickle=False)

    def write_export(self, session: SessionState, export_name: str, payload: dict[str, Any]) -> Path:
        export_dir = self.session_dir(session.session_id) / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        output_path = export_dir / export_name
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return output_path

    def list_image_deletion_records(self) -> list[dict[str, Any]]:
        path = self.image_deletions_path()
        if not path.exists():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return []
        records = payload.get("records") or []
        return [record for record in records if isinstance(record, dict)]

    def add_image_deletion_record(self, record: dict[str, Any]) -> None:
        path = self.image_deletions_path()
        with self._lock:
            payload = {"schema_version": 1, "records": self.list_image_deletion_records()}
            existing_ids = {str(item.get("deletion_id") or "") for item in payload["records"]}
            if str(record.get("deletion_id") or "") not in existing_ids:
                payload["records"].append(record)
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def list_session_meta(self) -> dict[str, dict[str, Any]]:
        results: dict[str, dict[str, Any]] = {}
        for path in self.sessions_root.glob("*/session.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            target_key = str(payload.get("session_id", "")).strip()
            if not target_key:
                continue
            history = payload.get("history") or []
            results[target_key] = {
                "has_session": True,
                "updated_at": payload.get("updated_at"),
                "history_count": len(history),
                "current_history_id": payload.get("current_history_id"),
                "is_deleted": bool(payload.get("is_deleted", False)),
                "deleted_at": payload.get("deleted_at"),
            }
        return results


class UploadedTargetStore:
    def __init__(self, imports_root: Path):
        self.imports_root = imports_root
        self.imports_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._targets_by_key: dict[str, TargetRecord] = {}
        self._imports_by_id: dict[str, dict[str, Any]] = {}
        self._load_existing_manifests()

    def get_target(self, key: str) -> TargetRecord:
        if key not in self._targets_by_key:
            raise KeyError(f"Unknown target key: {key}")
        return self._targets_by_key[key]

    def targets_for_image(self, target: TargetRecord) -> list[TargetRecord]:
        return [
            candidate
            for candidate in self._targets_by_key.values()
            if candidate.image_path == target.image_path
            and candidate.image_file_name == target.image_file_name
            and candidate.import_id == target.import_id
        ]

    def recent_targets(self, limit: int = 200) -> list[TargetRecord]:
        imports = sorted(
            self._imports_by_id.values(),
            key=lambda item: str(item.get("imported_at") or ""),
            reverse=True,
        )
        targets: list[TargetRecord] = []
        for import_payload in imports:
            targets.extend(import_payload["targets"])
            if len(targets) >= limit:
                break
        return targets[:limit]

    def import_bundle(
        self,
        image_file_name: str,
        image_data_url: str,
        annotation_file_name: str,
        annotation_text: str,
    ) -> dict[str, Any]:
        image_file_name = Path(image_file_name or "uploaded_image").name
        annotation_file_name = Path(annotation_file_name or "annotation.json").name
        image_bytes = self._decode_data_url(image_data_url)
        annotation_payload = json.loads(annotation_text)
        if isinstance(annotation_payload, list):
            if len(annotation_payload) != 1 or not isinstance(annotation_payload[0], dict):
                raise ValueError("Annotation JSON must describe exactly one image.")
            annotation_payload = annotation_payload[0]
        if not isinstance(annotation_payload, dict):
            raise ValueError("Annotation JSON must be an object.")

        imported_at = utc_now_iso()
        import_id = f"upload_{uuid4().hex[:12]}"
        import_dir = self.imports_root / import_id
        source_dir = import_dir / "source"
        source_dir.mkdir(parents=True, exist_ok=True)

        image_suffix = Path(image_file_name).suffix or ".png"
        image_path = source_dir / f"image{image_suffix}"
        annotation_path = source_dir / annotation_file_name
        image_path.write_bytes(image_bytes)
        annotation_path.write_text(annotation_text, encoding="utf-8")

        with Image.open(BytesIO(image_bytes)) as image:
            actual_width, actual_height = image.size

        targets = self._build_targets(
            annotation_payload=annotation_payload,
            annotation_file_name=annotation_file_name,
            annotation_path=annotation_path,
            image_file_name=image_file_name,
            image_path=image_path,
            actual_width=actual_width,
            actual_height=actual_height,
            import_id=import_id,
            imported_at=imported_at,
        )
        if not targets:
            raise ValueError("No rectanglelabels targets were found in the uploaded annotation JSON.")

        manifest_payload = {
            "schema_version": 1,
            "import_id": import_id,
            "imported_at": imported_at,
            "image_file_name": image_file_name,
            "image_path": str(image_path.resolve()),
            "annotation_file_name": annotation_file_name,
            "annotation_json_path": str(annotation_path.resolve()),
            "targets": [target.to_dict() for target in targets],
        }
        (import_dir / "manifest.json").write_text(
            json.dumps(manifest_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        with self._lock:
            self._register_manifest_payload(manifest_payload)

        return manifest_payload

    def _load_existing_manifests(self) -> None:
        for manifest_path in sorted(self.imports_root.glob("*/manifest.json"), reverse=True):
            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            self._register_manifest_payload(payload)

    def _register_manifest_payload(self, payload: dict[str, Any]) -> None:
        import_id = str(payload.get("import_id") or "").strip()
        if not import_id:
            return
        targets = [TargetRecord.from_dict(item) for item in payload.get("targets", [])]
        self._imports_by_id[import_id] = {
            "import_id": import_id,
            "imported_at": payload.get("imported_at"),
            "image_file_name": payload.get("image_file_name"),
            "image_path": payload.get("image_path"),
            "annotation_file_name": payload.get("annotation_file_name"),
            "annotation_json_path": payload.get("annotation_json_path"),
            "targets": targets,
        }
        for target in targets:
            self._targets_by_key[target.key] = target

    @staticmethod
    def _decode_data_url(data_url: str) -> bytes:
        raw = str(data_url or "").strip()
        if not raw:
            raise ValueError("Uploaded image data is empty.")
        if "," in raw:
            raw = raw.split(",", 1)[1]
        return base64.b64decode(raw)

    def _build_targets(
        self,
        annotation_payload: dict[str, Any],
        annotation_file_name: str,
        annotation_path: Path,
        image_file_name: str,
        image_path: Path,
        actual_width: int,
        actual_height: int,
        import_id: str,
        imported_at: str,
    ) -> list[TargetRecord]:
        targets: list[TargetRecord] = []
        seen_keys: set[str] = set()
        sort_index = 0

        for result_index, result in enumerate(annotation_payload.get("results", [])):
            if result.get("type") != "rectanglelabels":
                continue
            value = result.get("value") or {}
            labels = value.get("rectanglelabels") or []
            category_name = str(labels[0]).strip() if labels else "object"

            image_width = int(result.get("original_width") or annotation_payload.get("image_width") or actual_width)
            image_height = int(result.get("original_height") or annotation_payload.get("image_height") or actual_height)
            if image_width <= 0:
                image_width = actual_width
            if image_height <= 0:
                image_height = actual_height

            x = float(value["x"]) * image_width / 100.0
            y = float(value["y"]) * image_height / 100.0
            w = float(value["width"]) * image_width / 100.0
            h = float(value["height"]) * image_height / 100.0
            annotation_id = str(result.get("id", f"result_{result_index}"))
            key = sanitize_component(f"upload__{import_id}__{category_name}__{annotation_id}__{result_index}")
            if key in seen_keys:
                key = sanitize_component(f"{key}__dup_{result_index}")
            seen_keys.add(key)

            targets.append(
                TargetRecord(
                    key=key,
                    annotation_file_name=annotation_file_name,
                    annotation_json_path=str(annotation_path.resolve()),
                    image_path=str(image_path.resolve()),
                    image_file_name=image_file_name,
                    annotation_id=annotation_id,
                    source_annotation_id=str(result.get("id")) if result.get("id") is not None else None,
                    result_index=result_index,
                    category_name=category_name,
                    category_id=None,
                    image_width=image_width,
                    image_height=image_height,
                    bbox_xywh=round_floats([x, y, w, h]),
                    bbox_xyxy=round_floats([x, y, x + w, y + h]),
                    sort_index=sort_index,
                    import_id=import_id,
                    imported_at=imported_at,
                )
            )
            sort_index += 1

        targets.sort(key=lambda item: (item.category_name.lower(), item.sort_index))
        return targets


class Sam3InferenceService:
    def __init__(
        self,
        project_root: Path,
        sam3_repo_dir: Path,
        local_deps_dir: Path,
        checkpoint: Path | None = None,
        device: str = "auto",
        reference_box_expand_px: float = 10.0,
    ):
        self.project_root = project_root
        self.sam3_repo_dir = sam3_repo_dir
        self.local_deps_dir = local_deps_dir
        self.checkpoint = checkpoint
        self.device = device
        self.reference_box_expand_px = reference_box_expand_px
        self._model_lock = threading.RLock()
        self._runtime: dict[str, Any] | None = None
        self._model: Any = None
        self._processor: Any = None
        self._resolved_device: str | None = None

    @staticmethod
    def _torch_mps_available(torch: Any) -> bool:
        try:
            return bool(
                hasattr(torch, "backends")
                and hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            )
        except Exception:
            return False

    def _resolve_device(self, torch: Any) -> str:
        requested_device = str(self.device or "auto").strip().lower()
        if requested_device != "auto":
            return requested_device
        if torch.cuda.is_available():
            return "cuda"
        if self._torch_mps_available(torch):
            return "mps"
        return "cpu"

    def readiness(self) -> dict[str, Any]:
        resolved_checkpoint = self.resolve_checkpoint(raise_if_missing=False)
        return {
            "sam3_repo_dir": str(self.sam3_repo_dir),
            "checkpoint": str(resolved_checkpoint) if resolved_checkpoint else None,
            "checkpoint_exists": bool(resolved_checkpoint and resolved_checkpoint.exists()),
            "repo_exists": self.sam3_repo_dir.exists(),
            "reference_box_expand_px": self.reference_box_expand_px,
            "device": self.device,
        }

    def resolve_checkpoint(self, raise_if_missing: bool = True) -> Path | None:
        candidates: list[Path] = []
        if self.checkpoint is not None:
            candidates.append(self.checkpoint)
        candidates.extend(
            [
                self.sam3_repo_dir / "checkpoints" / "sam3.pt",
                self.sam3_repo_dir / "checkpoints" / "sam3.1_multiplex.pt",
                self.sam3_repo_dir / "sam3.pt",
                self.sam3_repo_dir / "sam3.1_multiplex.pt",
            ]
        )
        seen: set[Path] = set()
        for candidate in candidates:
            resolved = candidate.expanduser().resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            if resolved.exists():
                return resolved
        if raise_if_missing:
            raise FileNotFoundError(
                "Could not resolve a SAM3 checkpoint. Expected one of:\n"
                + "\n".join(str(path) for path in candidates)
            )
        return None

    def _configure_python_path(self) -> None:
        for candidate in (self.local_deps_dir, self.sam3_repo_dir):
            if candidate.exists():
                path_str = str(candidate)
                if path_str not in sys.path:
                    sys.path.insert(0, path_str)

    def _load_runtime(self) -> dict[str, Any]:
        if self._runtime is not None:
            return self._runtime
        self._configure_python_path()
        np = importlib.import_module("numpy")
        torch = importlib.import_module("torch")
        pillow_image = importlib.import_module("PIL.Image")
        try:
            mask_utils = importlib.import_module("pycocotools.mask")
        except ModuleNotFoundError:
            mask_utils = None
        sam3_module = importlib.import_module("sam3")
        processor_module = importlib.import_module("sam3.model.sam3_image_processor")
        self._runtime = {
            "np": np,
            "torch": torch,
            "Image": pillow_image,
            "mask_utils": mask_utils,
            "build_sam3_image_model": getattr(sam3_module, "build_sam3_image_model"),
            "Sam3Processor": getattr(processor_module, "Sam3Processor"),
        }
        return self._runtime

    def _disable_interactive_bfloat16_autocast_if_needed(self, model: Any, resolved_device: str, torch: Any) -> None:
        if resolved_device != "cuda":
            return
        predictor = getattr(model, "inst_interactive_predictor", None)
        interactive_model = getattr(predictor, "model", None)
        bf16_context = getattr(interactive_model, "bf16_context", None)
        if bf16_context is None:
            return
        try:
            bf16_context.__exit__(None, None, None)
        except Exception:
            pass
        try:
            interactive_model.bf16_context = None
        except Exception:
            pass
        try:
            torch.set_autocast_enabled(False)
        except Exception:
            pass

    def _ensure_model(self) -> tuple[Any, Any, str]:
        with self._model_lock:
            if self._model is not None and self._processor is not None and self._resolved_device is not None:
                return self._model, self._processor, self._resolved_device

            runtime = self._load_runtime()
            torch = runtime["torch"]
            build_sam3_image_model = runtime["build_sam3_image_model"]
            Sam3Processor = runtime["Sam3Processor"]

            resolved_device = self._resolve_device(torch)

            checkpoint = self.resolve_checkpoint(raise_if_missing=True)
            bpe_path = self.sam3_repo_dir / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
            if not bpe_path.exists():
                raise FileNotFoundError(f"Missing SAM3 BPE asset: {bpe_path}")

            model = build_sam3_image_model(
                bpe_path=str(bpe_path),
                checkpoint_path=str(checkpoint),
                load_from_HF=False,
                device=resolved_device,
                enable_inst_interactivity=True,
            )
            self._disable_interactive_bfloat16_autocast_if_needed(model, resolved_device, torch)
            processor = Sam3Processor(model, device=resolved_device)
            self._model = model
            self._processor = processor
            self._resolved_device = resolved_device
            return model, processor, resolved_device

    def _load_image(self, image_path: str) -> Any:
        runtime = self._load_runtime()
        ImageModule = runtime["Image"]
        return ImageModule.open(image_path).convert("RGB")

    @staticmethod
    def _center_of_box_xyxy(box_xyxy: list[float]) -> list[float]:
        x0, y0, x1, y1 = [float(value) for value in box_xyxy]
        return [(x0 + x1) / 2.0, (y0 + y1) / 2.0]

    @staticmethod
    def _box_xyxy_to_normalized_cxcywh(
        box_xyxy: list[float], image_width: int, image_height: int
    ) -> list[float]:
        x0, y0, x1, y1 = [float(value) for value in box_xyxy]
        width = max(1.0, float(image_width))
        height = max(1.0, float(image_height))
        cx = ((x0 + x1) / 2.0) / width
        cy = ((y0 + y1) / 2.0) / height
        bw = max(1.0, x1 - x0) / width
        bh = max(1.0, y1 - y0) / height
        return [cx, cy, bw, bh]

    @staticmethod
    def _box_iou_xyxy(box_a: list[float], box_b: list[float]) -> float:
        ax0, ay0, ax1, ay1 = [float(value) for value in box_a]
        bx0, by0, bx1, by1 = [float(value) for value in box_b]
        inter_x0 = max(ax0, bx0)
        inter_y0 = max(ay0, by0)
        inter_x1 = min(ax1, bx1)
        inter_y1 = min(ay1, by1)
        inter_w = max(0.0, inter_x1 - inter_x0)
        inter_h = max(0.0, inter_y1 - inter_y0)
        inter_area = inter_w * inter_h
        area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
        area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
        union = area_a + area_b - inter_area
        if union <= 0:
            return 0.0
        return inter_area / union

    @staticmethod
    def _to_numpy(value: Any) -> Any:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            return value.numpy()
        return value

    def _flatten_prediction_outputs(self, masks: Any, scores: Any, logits: Any) -> tuple[Any, Any, Any]:
        np = self._load_runtime()["np"]
        masks_np = np.asarray(masks)
        scores_np = np.asarray(scores, dtype=np.float32).reshape(-1)
        logits_np = np.asarray(logits, dtype=np.float32)
        if masks_np.ndim == 4 and masks_np.shape[0] == 1:
            masks_np = masks_np[0]
        if logits_np.ndim == 4 and logits_np.shape[0] == 1:
            logits_np = logits_np[0]
        if masks_np.ndim == 2:
            masks_np = masks_np[None, ...]
        if logits_np.ndim == 2:
            logits_np = logits_np[None, ...]
        return masks_np, scores_np, logits_np

    def _select_best_prediction(self, masks: Any, scores: Any, logits: Any) -> tuple[Any, float, Any]:
        np = self._load_runtime()["np"]
        masks_np, scores_np, logits_np = self._flatten_prediction_outputs(masks, scores, logits)
        best_index = int(np.argmax(scores_np))
        best_mask = masks_np[best_index]
        if best_mask.dtype != np.bool_:
            best_mask = best_mask > 0
        return best_mask.astype(bool), float(scores_np[best_index]), logits_np[best_index]

    def _mask_to_rle(self, mask: Any) -> dict[str, Any]:
        np = self._load_runtime()["np"]
        pixels = mask.astype(np.uint8).reshape(-1, order="F")
        counts: list[int] = []
        current = 0
        run_length = 0
        for pixel in pixels:
            pixel = int(pixel)
            if pixel != current:
                counts.append(run_length)
                run_length = 1
                current = pixel
            else:
                run_length += 1
        counts.append(run_length)
        return {"size": [int(mask.shape[0]), int(mask.shape[1])], "counts": counts}

    def mask_from_rle(self, rle: dict[str, Any]) -> Any:
        runtime = self._load_runtime()
        np = runtime["np"]
        mask_utils = runtime["mask_utils"]
        counts = rle["counts"]
        if mask_utils is not None and not isinstance(counts, list):
            normalized = dict(rle)
            normalized_counts = normalized.get("counts")
            if isinstance(normalized_counts, str):
                normalized["counts"] = normalized_counts.encode("utf-8")
            decoded = mask_utils.decode(normalized)
            if decoded.ndim == 3:
                decoded = decoded[:, :, 0]
            return decoded.astype(bool)

        height, width = [int(value) for value in rle["size"]]
        if not isinstance(counts, list):
            raise ValueError("RLE fallback decoder expects list counts when pycocotools is unavailable.")
        flat = np.zeros(height * width, dtype=np.uint8)
        cursor = 0
        current = 0
        for count in counts:
            count = int(count)
            if count > 0:
                flat[cursor : cursor + count] = current
                cursor += count
            current = 1 - current
        return flat.reshape((height, width), order="F").astype(bool)

    def _mask_to_xywh(self, mask: Any) -> list[float] | None:
        np = self._load_runtime()["np"]
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return None
        x0 = int(xs.min())
        y0 = int(ys.min())
        x1 = int(xs.max()) + 1
        y1 = int(ys.max()) + 1
        return [float(x0), float(y0), float(x1 - x0), float(y1 - y0)]

    def ground_text_with_box(
        self,
        target: TargetRecord,
        prompt_box_xyxy: list[float],
        text_prompt: str,
    ) -> dict[str, Any] | None:
        prompt = str(text_prompt or "").strip()
        if not prompt:
            return None

        _, processor, _ = self._ensure_model()
        np = self._load_runtime()["np"]
        image = self._load_image(target.image_path)
        inference_state = processor.set_image(image)
        processor.set_text_prompt(prompt=prompt, state=inference_state)
        output = processor.add_geometric_prompt(
            box=self._box_xyxy_to_normalized_cxcywh(
                prompt_box_xyxy, target.image_width, target.image_height
            ),
            label=True,
            state=inference_state,
        )

        masks = np.asarray(self._to_numpy(output.get("masks")))
        boxes = np.asarray(self._to_numpy(output.get("boxes")), dtype=np.float32)
        scores = np.asarray(self._to_numpy(output.get("scores")), dtype=np.float32).reshape(-1)
        if masks.size == 0 or boxes.size == 0 or scores.size == 0:
            return None
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
        if masks.ndim == 2:
            masks = masks[None, ...]
        if boxes.ndim == 1:
            boxes = boxes[None, ...]

        best_index = 0
        best_rank = float("-inf")
        for index in range(min(len(scores), len(boxes), len(masks))):
            candidate_box = boxes[index].tolist()
            overlap = self._box_iou_xyxy(candidate_box, prompt_box_xyxy)
            rank = float(scores[index]) + overlap * 0.35
            if rank > best_rank:
                best_index = index
                best_rank = rank

        best_mask = masks[best_index]
        if best_mask.dtype != np.bool_:
            best_mask = best_mask > 0.5
        best_box = boxes[best_index].tolist()
        return {
            "mask": best_mask.astype(bool),
            "box_xyxy": [float(value) for value in best_box],
            "score": float(scores[best_index]),
            "text_prompt": prompt,
        }

    def predict_initial(self, target: TargetRecord) -> dict[str, Any]:
        model, processor, resolved_device = self._ensure_model()
        runtime = self._load_runtime()
        np = runtime["np"]
        torch = runtime["torch"]
        if resolved_device == "cuda":
            try:
                torch.set_autocast_enabled(False)
            except Exception:
                pass
        image = self._load_image(target.image_path)
        prompt_box_xyxy = expand_box_xyxy(
            target.bbox_xyxy,
            self.reference_box_expand_px,
            target.image_width,
            target.image_height,
        )
        center_point = self._center_of_box_xyxy(prompt_box_xyxy)
        autocast_context = (
            torch.autocast(device_type=resolved_device, enabled=False)
            if resolved_device in {"cuda", "mps"}
            else contextlib.nullcontext()
        )
        with autocast_context:
            state = processor.set_image(image)
            masks, scores, logits = model.predict_inst(
                state,
                point_coords=np.asarray([center_point], dtype=np.float32),
                point_labels=np.asarray([1], dtype=np.int32),
                box=np.asarray([prompt_box_xyxy], dtype=np.float32),
                mask_input=None,
                multimask_output=False,
            )
        best_mask, score, best_logits = self._select_best_prediction(masks, scores, logits)
        return {
            "prompt_box_xyxy": [float(value) for value in prompt_box_xyxy],
            "system_prompt_points": [
                PointRecord(
                    point_id="system_center_prompt",
                    x=float(center_point[0]),
                    y=float(center_point[1]),
                    label=1,
                    created_at=utc_now_iso(),
                    source="system",
                )
            ],
            "mask_rle": self._mask_to_rle(best_mask),
            "mask_area": int(best_mask.sum()),
            "mask_bbox_xywh": self._mask_to_xywh(best_mask),
            "score": float(score),
            "logits": best_logits,
        }

    def iterate(
        self,
        target: TargetRecord,
        prompt_box_xyxy: list[float],
        working_points: list[PointRecord],
        previous_logits: Any,
    ) -> dict[str, Any]:
        model, processor, resolved_device = self._ensure_model()
        runtime = self._load_runtime()
        np = runtime["np"]
        torch = runtime["torch"]
        if resolved_device == "cuda":
            try:
                torch.set_autocast_enabled(False)
            except Exception:
                pass
        image = self._load_image(target.image_path)

        point_coords = None
        point_labels = None
        if not working_points:
            working_points = [
                PointRecord(
                    point_id="system_box_center_iteration",
                    x=float(self._center_of_box_xyxy(prompt_box_xyxy)[0]),
                    y=float(self._center_of_box_xyxy(prompt_box_xyxy)[1]),
                    label=1,
                    created_at=utc_now_iso(),
                    source="system",
                )
            ]
        if working_points:
            point_coords = np.asarray([[float(point.x), float(point.y)] for point in working_points], dtype=np.float32)
            point_labels = np.asarray([int(point.label) for point in working_points], dtype=np.int32)

        mask_input = np.asarray(previous_logits, dtype=np.float32)
        if mask_input.ndim == 2:
            mask_input = mask_input[None, :, :]

        autocast_context = (
            torch.autocast(device_type=resolved_device, enabled=False)
            if resolved_device in {"cuda", "mps"}
            else contextlib.nullcontext()
        )
        with autocast_context:
            state = processor.set_image(image)
            masks, scores, logits = model.predict_inst(
                state,
                point_coords=point_coords,
                point_labels=point_labels,
                box=np.asarray([prompt_box_xyxy], dtype=np.float32),
                mask_input=mask_input,
                multimask_output=False,
            )
        best_mask, score, best_logits = self._select_best_prediction(masks, scores, logits)
        return {
            "mask_rle": self._mask_to_rle(best_mask),
            "mask_area": int(best_mask.sum()),
            "mask_bbox_xywh": self._mask_to_xywh(best_mask),
            "score": float(score),
            "logits": best_logits,
        }


class MaskIterationService:
    def __init__(
        self,
        target_store: UploadedTargetStore,
        session_store: SessionStore,
        inference_service: Sam3InferenceService,
        validate_tools_dir: Path | None = None,
    ):
        self.target_store = target_store
        self.session_store = session_store
        self.inference_service = inference_service
        self.validate_tools_dir = validate_tools_dir
        self._lock = threading.RLock()
        self._validate_runtime: dict[str, Any] | None = None
        self._validate_lock = threading.RLock()

    def bootstrap_payload(self) -> dict[str, Any]:
        session_meta = self.session_store.list_session_meta()
        image_deletion_records = self.session_store.list_image_deletion_records()
        image_deleted_keys = {
            str(target_key)
            for record in image_deletion_records
            for target_key in (record.get("target_keys") or [])
        }
        recent_targets = [
            self._target_payload(target, session_meta.get(target.key, {}))
            for target in self.target_store.recent_targets()
            if not bool((session_meta.get(target.key, {}) or {}).get("is_deleted", False))
            and target.key not in image_deleted_keys
        ]
        return {
            "app": {
                "title": "SAM3 Mask Iteration Web App",
                "reference_box_expand_px": self.inference_service.reference_box_expand_px,
                "sam3": self.inference_service.readiness(),
            },
            "recent_targets": recent_targets,
            "image_deletions": image_deletion_records,
            "validate_tools": self._build_validate_tools_payload(),
        }

    def _build_validate_tools_payload(self) -> dict[str, Any]:
        base_dir = self.validate_tools_dir
        files: list[dict[str, Any]] = []
        rules_count = 0
        if base_dir and base_dir.exists():
            for name in [
                "annotation_validator.py",
                "visualization_tool.py",
                "run.py",
                "rules.json",
                "requirements.txt",
                "README.md",
            ]:
                path = base_dir / name
                files.append(
                    {
                        "name": name,
                        "exists": path.exists(),
                        "path": str(path),
                    }
                )
            rules_path = base_dir / "rules.json"
            if rules_path.exists():
                try:
                    rules_count = len(json.loads(rules_path.read_text(encoding="utf-8")))
                except Exception:
                    rules_count = 0
        return {
            "available": bool(base_dir and base_dir.exists()),
            "base_dir": str(base_dir) if base_dir else None,
            "rules_count": rules_count,
            "workflow": [
                "输入原图和每张图对应的 JSON 标注",
                "调用 LLM 检查错标、漏标和整体质量",
                "输出 validation_results.json",
                "生成错误高亮的可视化图片",
            ],
            "capabilities": [
                "单张或批量标注质检",
                "错误标注和漏标检测",
                "质量评分与备注",
                "可视化结果导出",
            ],
            "commands": [
                "python run.py validate -s image.jpg",
                "python run.py validate -l 20 -c 3",
                "python run.py visualize -v validation_results.json --only-errors",
                "python run.py full -l 50 -c 5",
            ],
            "files": files,
        }

    def _ensure_validate_tools_runtime(self) -> dict[str, Any]:
        with self._validate_lock:
            if self._validate_runtime is not None:
                return self._validate_runtime
            base_dir = self.validate_tools_dir
            if base_dir is None or not base_dir.exists():
                raise FileNotFoundError("Validate_tools directory is not configured or does not exist.")

            def load_module(module_name: str, file_name: str) -> Any:
                module_path = base_dir / file_name
                if not module_path.exists():
                    raise FileNotFoundError(f"Missing Validate_tools module: {module_path}")
                spec = importlib.util.spec_from_file_location(module_name, module_path)
                if spec is None or spec.loader is None:
                    raise ImportError(f"Unable to load module from {module_path}")
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module

            annotation_validator = load_module("validate_tools_annotation_validator", "annotation_validator.py")
            visualization_tool = load_module("validate_tools_visualization_tool", "visualization_tool.py")
            rules_path = base_dir / "rules.json"
            if not rules_path.exists():
                raise FileNotFoundError(f"Missing Validate_tools rules file: {rules_path}")
            self._validate_runtime = {
                "base_dir": base_dir,
                "annotation_validator": annotation_validator,
                "visualization_tool": visualization_tool,
                "rules_path": rules_path,
            }
            return self._validate_runtime

    def _session_validate_dir(self, session: SessionState) -> Path:
        path = self.session_store.session_dir(session.session_id) / "validate_tools"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _save_validate_job(self, session: SessionState, manifest: dict[str, Any]) -> None:
        job_dir = self._session_validate_dir(session) / str(manifest["job_id"])
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "job.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _list_validate_jobs(self, session: SessionState, limit: int = 20) -> list[dict[str, Any]]:
        jobs_root = self._session_validate_dir(session)
        jobs: list[dict[str, Any]] = []
        for path in jobs_root.glob("*/job.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            jobs.append(payload)
        jobs.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
        return jobs[:limit]

    def _build_validate_job_payload(self, session: SessionState, payload: dict[str, Any]) -> dict[str, Any]:
        relpaths = payload.get("artifacts", {}) or {}
        result_relpath = relpaths.get("result_json_relpath")
        image_relpath = relpaths.get("visualization_relpath")
        return {
            **payload,
            "artifact_urls": {
                "result_json": (
                    f"/api/sessions/{session.session_id}/validate-tools/artifact?relpath={result_relpath}"
                    if result_relpath else None
                ),
                "visualization": (
                    f"/api/sessions/{session.session_id}/validate-tools/artifact?relpath={image_relpath}"
                    if image_relpath else None
                ),
            },
        }

    def _validate_tools_session_payload(self, session: SessionState) -> dict[str, Any]:
        jobs = [self._build_validate_job_payload(session, item) for item in self._list_validate_jobs(session)]
        return {
            "jobs": jobs,
            "latest_job": jobs[0] if jobs else None,
        }

    def get_validate_artifact_path(self, target_key: str, relpath: str) -> Path:
        session = self._require_session(target_key)
        base_dir = self._session_validate_dir(session)
        candidate = (base_dir / relpath).resolve()
        if base_dir.resolve() not in candidate.parents and candidate != base_dir.resolve():
            raise PermissionError("Forbidden artifact path")
        if not candidate.exists() or not candidate.is_file():
            raise FileNotFoundError(f"Validate artifact not found: {candidate}")
        return candidate

    @staticmethod
    def _manifest_relpath(base_dir: Path, path: Path | None) -> str | None:
        if path is None:
            return None
        return str(path.resolve().relative_to(base_dir.resolve())).replace("\\", "/")

    @staticmethod
    def _resolve_validate_api_key(api_key: str | None) -> str | None:
        if api_key:
            return api_key
        for env_name in VALIDATE_API_KEY_ENV_VARS:
            candidate = os.environ.get(env_name)
            if candidate:
                return candidate
        return None

    @staticmethod
    def _normalize_validate_review_mode(review_mode: str | None) -> str:
        candidate = str(review_mode or "").strip().lower()
        if candidate == VALIDATE_REVIEW_MODE_ORIGINAL:
            return VALIDATE_REVIEW_MODE_ORIGINAL
        return VALIDATE_REVIEW_MODE_CROP_BOX2X

    @staticmethod
    def _validate_review_scope(review_mode: str) -> str:
        if review_mode == VALIDATE_REVIEW_MODE_ORIGINAL:
            return "single_annotation_original_image_box"
        return "single_annotation_crop_box2x"

    def _write_validate_json_artifact(self, session: SessionState, job_dir: Path, file_name: str, payload: Any) -> Path:
        path = job_dir / file_name
        if hasattr(payload, "to_dict"):
            payload = payload.to_dict()
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def _find_target_annotation_result(self, session: SessionState, payload: dict[str, Any]) -> dict[str, Any]:
        results = payload.get("results") or []
        match: dict[str, Any] | None = None
        target_ids = {
            str(session.target.annotation_id),
            str(session.target.source_annotation_id),
        }
        for index, result in enumerate(results):
            result_id = result.get("id")
            if result_id is not None and str(result_id) in target_ids:
                match = result
                break
            if index == int(session.target.result_index):
                match = result
                break
        if match is None:
            annotation_path = Path(session.target.annotation_json_path)
            raise ValueError(
                f"Unable to locate target annotation for session {session.session_id} "
                f"in {annotation_path}"
            )
        return deepcopy(match)

    def _compute_single_target_crop_box(self, session: SessionState) -> tuple[int, int, int, int]:
        x0, y0, x1, y1 = [float(value) for value in session.target.bbox_xyxy]
        box_w = max(1.0, x1 - x0)
        box_h = max(1.0, y1 - y0)
        center_x = (x0 + x1) / 2.0
        center_y = (y0 + y1) / 2.0
        crop_w = max(2.0, box_w * 2.0)
        crop_h = max(2.0, box_h * 2.0)

        crop_x0 = max(0, int(round(center_x - crop_w / 2.0)))
        crop_y0 = max(0, int(round(center_y - crop_h / 2.0)))
        crop_x1 = min(int(session.target.image_width), int(round(center_x + crop_w / 2.0)))
        crop_y1 = min(int(session.target.image_height), int(round(center_y + crop_h / 2.0)))

        if crop_x1 <= crop_x0:
            crop_x1 = min(int(session.target.image_width), crop_x0 + 1)
        if crop_y1 <= crop_y0:
            crop_y1 = min(int(session.target.image_height), crop_y0 + 1)
        return crop_x0, crop_y0, crop_x1, crop_y1

    def _write_single_target_review_assets(
        self,
        session: SessionState,
        job_dir: Path,
        review_mode: str,
    ) -> tuple[Path, Path, dict[str, Any], str, str]:
        annotation_path = Path(session.target.annotation_json_path)
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        match = self._find_target_annotation_result(session, payload)
        normalized_review_mode = self._normalize_validate_review_mode(review_mode)
        review_scope = self._validate_review_scope(normalized_review_mode)

        if normalized_review_mode == VALIDATE_REVIEW_MODE_ORIGINAL:
            review_image_path = Path(session.target.image_path)
            crop_x0 = 0
            crop_y0 = 0
            crop_x1 = int(session.target.image_width)
            crop_y1 = int(session.target.image_height)
            review_width = max(1, int(session.target.image_width))
            review_height = max(1, int(session.target.image_height))
        else:
            crop_x0, crop_y0, crop_x1, crop_y1 = self._compute_single_target_crop_box(session)
            review_width = max(1, crop_x1 - crop_x0)
            review_height = max(1, crop_y1 - crop_y0)
            with Image.open(session.target.image_path) as source_image:
                cropped = source_image.convert("RGB").crop((crop_x0, crop_y0, crop_x1, crop_y1))
                review_image_path = job_dir / "single_target_review.png"
                cropped.save(review_image_path)

        rel_x0 = float(session.target.bbox_xyxy[0]) - float(crop_x0)
        rel_y0 = float(session.target.bbox_xyxy[1]) - float(crop_y0)
        rel_x1 = float(session.target.bbox_xyxy[2]) - float(crop_x0)
        rel_y1 = float(session.target.bbox_xyxy[3]) - float(crop_y0)
        rel_w = max(1.0, rel_x1 - rel_x0)
        rel_h = max(1.0, rel_y1 - rel_y0)
        center_x_pct = ((rel_x0 + rel_w / 2.0) / float(review_width)) * 100.0
        center_y_pct = ((rel_y0 + rel_h / 2.0) / float(review_height)) * 100.0
        width_pct = (rel_w / float(review_width)) * 100.0
        height_pct = (rel_h / float(review_height)) * 100.0

        review_result = deepcopy(match)
        review_value = deepcopy(review_result.get("value") or {})
        review_value["x"] = round(center_x_pct, 6)
        review_value["y"] = round(center_y_pct, 6)
        review_value["width"] = round(width_pct, 6)
        review_value["height"] = round(height_pct, 6)
        review_value["rectanglelabels"] = [session.target.category_name]
        review_result["value"] = review_value
        review_result["original_width"] = int(review_width)
        review_result["original_height"] = int(review_height)
        filtered_payload = deepcopy(payload)
        filtered_payload["results"] = [review_result]
        filtered_payload["review_scope"] = review_scope
        filtered_payload["review_mode"] = normalized_review_mode
        filtered_payload["review_target"] = {
            "session_id": session.session_id,
            "annotation_id": session.target.annotation_id,
            "source_annotation_id": session.target.source_annotation_id,
            "result_index": session.target.result_index,
            "category_name": session.target.category_name,
            "review_mode": normalized_review_mode,
            "review_scope": review_scope,
            "original_bbox_xyxy": [float(value) for value in session.target.bbox_xyxy],
            "crop_xyxy_on_original": [crop_x0, crop_y0, crop_x1, crop_y1],
            "review_image_size": [review_width, review_height],
            "bbox_xyxy_on_review_image": [round(rel_x0, 3), round(rel_y0, 3), round(rel_x1, 3), round(rel_y1, 3)],
            "bbox_xywh_on_review_image": [round(rel_x0, 3), round(rel_y0, 3), round(rel_w, 3), round(rel_h, 3)],
        }
        path = job_dir / "single_target_annotation.json"
        path.write_text(json.dumps(filtered_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return review_image_path, path, filtered_payload["review_target"], review_scope, normalized_review_mode

    def _run_validate_single_job(
        self,
        session: SessionState,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        strict_mode: bool = False,
        review_mode: str | None = None,
    ) -> dict[str, Any]:
        runtime = self._ensure_validate_tools_runtime()
        validate_single = getattr(runtime["annotation_validator"], "validate_single")
        validate_root = self._session_validate_dir(session)
        job_id = f"validate_{uuid4().hex[:12]}"
        job_dir = validate_root / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        review_image_path, target_anno_path, review_target, review_scope, normalized_review_mode = (
            self._write_single_target_review_assets(session, job_dir, review_mode or DEFAULT_VALIDATE_REVIEW_MODE)
        )
        resolved_api_key = self._resolve_validate_api_key(api_key)
        resolved_base_url = base_url or DEFAULT_VALIDATE_BASE_URL
        resolved_model = model or DEFAULT_VALIDATE_MODEL
        result = validate_single(
            image_path=str(review_image_path),
            anno_path=str(target_anno_path),
            rules_path=str(runtime["rules_path"]),
            api_key=resolved_api_key,
            base_url=resolved_base_url,
            model=resolved_model,
            strict_mode=bool(strict_mode),
        )
        result_dict = result.to_dict() if hasattr(result, "to_dict") else result
        result_dict["review_scope"] = review_scope
        result_dict["review_mode"] = normalized_review_mode
        result_dict["target_annotation_id"] = session.target.annotation_id
        result_dict["target_category"] = session.target.category_name
        result_dict["review_target"] = review_target
        result_path = self._write_validate_json_artifact(session, job_dir, "validation_result.json", result_dict)
        comments = str(result_dict.get("comments", "") or "")
        is_error = comments.startswith("验证失败:")
        summary = {
            "image_name": result_dict.get("image_name"),
            "review_scope": review_scope,
            "review_mode": normalized_review_mode,
            "target_annotation_id": session.target.annotation_id,
            "target_category": session.target.category_name,
            "incorrect_count": len(result_dict.get("incorrect_annotations", []) or []),
            "missing_count": len(result_dict.get("missing_annotations", []) or []),
            "score": result_dict.get("score"),
            "comments": comments,
            "review_target": review_target,
        }
        manifest = {
            "job_id": job_id,
            "created_at": utc_now_iso(),
            "kind": "validate",
            "status": "error" if is_error else "success",
            "error_message": comments if is_error else None,
            "target_key": session.session_id,
            "target_image": session.target.image_file_name,
            "review_scope": review_scope,
            "review_mode": normalized_review_mode,
            "target_annotation_id": session.target.annotation_id,
            "target_category": session.target.category_name,
            "strict_mode": bool(strict_mode),
            "base_url": resolved_base_url,
            "model": resolved_model,
            "summary": summary,
            "review_target": review_target,
            "artifacts": {
                "result_json_relpath": self._manifest_relpath(validate_root, result_path),
                "visualization_relpath": None,
            },
        }
        self._save_validate_job(session, manifest)
        return manifest

    def _run_visualize_single_job(
        self,
        session: SessionState,
        validation_result: dict[str, Any] | None = None,
        parent_job_id: str | None = None,
        review_mode: str | None = None,
    ) -> dict[str, Any]:
        runtime = self._ensure_validate_tools_runtime()
        visualize_single = getattr(runtime["visualization_tool"], "visualize_single")
        validate_root = self._session_validate_dir(session)
        job_id = f"visualize_{uuid4().hex[:12]}"
        job_dir = validate_root / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        image_path = job_dir / f"{sanitize_component(session.target.image_file_name)}__visualization.jpg"
        review_image_path, target_anno_path, review_target, review_scope, normalized_review_mode = (
            self._write_single_target_review_assets(session, job_dir, review_mode or DEFAULT_VALIDATE_REVIEW_MODE)
        )
        visualize_single(
            image_path=str(review_image_path),
            anno_path=str(target_anno_path),
            output_path=str(image_path),
            validation_result=validation_result,
            use_pil=True,
        )
        summary = {
            "image_name": session.target.image_file_name,
            "review_scope": review_scope,
            "review_mode": normalized_review_mode,
            "target_annotation_id": session.target.annotation_id,
            "target_category": session.target.category_name,
            "review_target": review_target,
            "overlay_mode": "validation" if validation_result else "plain",
        }
        manifest = {
            "job_id": job_id,
            "created_at": utc_now_iso(),
            "kind": "visualize",
            "status": "success",
            "error_message": None,
            "target_key": session.session_id,
            "target_image": session.target.image_file_name,
            "review_scope": review_scope,
            "review_mode": normalized_review_mode,
            "target_annotation_id": session.target.annotation_id,
            "target_category": session.target.category_name,
            "parent_job_id": parent_job_id,
            "summary": summary,
            "review_target": review_target,
            "artifacts": {
                "result_json_relpath": None,
                "visualization_relpath": self._manifest_relpath(validate_root, image_path),
            },
        }
        self._save_validate_job(session, manifest)
        return manifest

    def run_validate_tools_validate(
        self,
        target_key: str,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        strict_mode: bool = False,
        review_mode: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            session = self._require_session(target_key)
            manifest = self._run_validate_single_job(
                session=session,
                api_key=api_key,
                base_url=base_url,
                model=model,
                strict_mode=strict_mode,
                review_mode=review_mode,
            )
            return {
                **self._session_payload(session),
                "validate_tools": self._validate_tools_session_payload(session),
                "validate_job": self._build_validate_job_payload(session, manifest),
            }

    def run_validate_tools_visualize(
        self,
        target_key: str,
        use_latest_validation: bool = True,
        review_mode: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            session = self._require_session(target_key)
            validation_result = None
            parent_job_id = None
            resolved_review_mode = self._normalize_validate_review_mode(review_mode or DEFAULT_VALIDATE_REVIEW_MODE)
            if use_latest_validation:
                for item in self._list_validate_jobs(session):
                    if item.get("kind") not in {"validate", "full"}:
                        continue
                    result_relpath = (item.get("artifacts") or {}).get("result_json_relpath")
                    if not result_relpath:
                        continue
                    result_path = self.get_validate_artifact_path(target_key, result_relpath)
                    validation_result = json.loads(result_path.read_text(encoding="utf-8"))
                    parent_job_id = str(item.get("job_id") or "")
                    resolved_review_mode = self._normalize_validate_review_mode(
                        str(item.get("review_mode") or (item.get("summary") or {}).get("review_mode") or resolved_review_mode)
                    )
                    break
            manifest = self._run_visualize_single_job(
                session=session,
                validation_result=validation_result,
                parent_job_id=parent_job_id,
                review_mode=resolved_review_mode,
            )
            return {
                **self._session_payload(session),
                "validate_tools": self._validate_tools_session_payload(session),
                "validate_job": self._build_validate_job_payload(session, manifest),
            }

    def run_validate_tools_full(
        self,
        target_key: str,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        strict_mode: bool = False,
        review_mode: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            session = self._require_session(target_key)
            validate_manifest = self._run_validate_single_job(
                session=session,
                api_key=api_key,
                base_url=base_url,
                model=model,
                strict_mode=strict_mode,
                review_mode=review_mode,
            )
            result_relpath = (validate_manifest.get("artifacts") or {}).get("result_json_relpath")
            validation_result = None
            if result_relpath:
                result_path = self.get_validate_artifact_path(target_key, result_relpath)
                validation_result = json.loads(result_path.read_text(encoding="utf-8"))
            visualize_manifest = self._run_visualize_single_job(
                session=session,
                validation_result=validation_result,
                parent_job_id=str(validate_manifest.get("job_id") or ""),
                review_mode=str(validate_manifest.get("review_mode") or review_mode or DEFAULT_VALIDATE_REVIEW_MODE),
            )
            full_manifest = {
                "job_id": f"full_{uuid4().hex[:12]}",
                "created_at": utc_now_iso(),
                "kind": "full",
                "status": str(validate_manifest.get("status") or "success"),
                "error_message": validate_manifest.get("error_message"),
                "target_key": session.session_id,
                "target_image": session.target.image_file_name,
                "review_scope": str(validate_manifest.get("review_scope") or self._validate_review_scope(DEFAULT_VALIDATE_REVIEW_MODE)),
                "review_mode": str(validate_manifest.get("review_mode") or self._normalize_validate_review_mode(review_mode)),
                "target_annotation_id": session.target.annotation_id,
                "target_category": session.target.category_name,
                "strict_mode": bool(strict_mode),
                "base_url": base_url or DEFAULT_VALIDATE_BASE_URL,
                "model": model or DEFAULT_VALIDATE_MODEL,
                "summary": validate_manifest.get("summary", {}),
                "review_target": validate_manifest.get("review_target") or (validate_manifest.get("summary") or {}).get("review_target"),
                "artifacts": {
                    "result_json_relpath": (validate_manifest.get("artifacts") or {}).get("result_json_relpath"),
                    "visualization_relpath": (visualize_manifest.get("artifacts") or {}).get("visualization_relpath"),
                },
                "child_job_ids": [validate_manifest.get("job_id"), visualize_manifest.get("job_id")],
            }
            self._save_validate_job(session, full_manifest)
            return {
                **self._session_payload(session),
                "validate_tools": self._validate_tools_session_payload(session),
                "validate_job": self._build_validate_job_payload(session, full_manifest),
            }

    def _normalize_line_strokes(
        self,
        session: SessionState,
        line_strokes_payload: Any,
    ) -> list[LineStrokeRecord]:
        normalized: list[LineStrokeRecord] = []
        if not isinstance(line_strokes_payload, list):
            return normalized
        for item in line_strokes_payload:
            if not isinstance(item, dict):
                continue
            raw_points = item.get("points")
            if not isinstance(raw_points, list) or not raw_points:
                continue
            points: list[StrokePointRecord] = []
            last_xy: tuple[float, float] | None = None
            for point in raw_points:
                if not isinstance(point, dict):
                    continue
                x = self._clamp(float(point.get("x", 0.0)), 0.0, float(session.target.image_width - 1))
                y = self._clamp(float(point.get("y", 0.0)), 0.0, float(session.target.image_height - 1))
                current_xy = (round(x, 2), round(y, 2))
                if last_xy == current_xy:
                    continue
                points.append(StrokePointRecord(x=x, y=y))
                last_xy = current_xy
            if not points:
                continue
            normalized.append(
                LineStrokeRecord(
                    stroke_id=str(item.get("stroke_id") or f"stroke_{uuid4().hex[:12]}"),
                    label=1 if int(item.get("label", 1)) == 1 else 0,
                    created_at=str(item.get("created_at") or utc_now_iso()),
                    source=str(item.get("source", "manual") or "manual"),
                    points=points,
                )
            )
        return normalized

    def _normalize_locked_regions(
        self,
        session: SessionState,
        locked_regions_payload: Any,
    ) -> list[LockedRegionRecord]:
        normalized: list[LockedRegionRecord] = []
        if not isinstance(locked_regions_payload, list):
            return normalized
        for item in locked_regions_payload:
            if not isinstance(item, dict):
                continue
            raw_points = item.get("points")
            if not isinstance(raw_points, list) or len(raw_points) < 3:
                continue
            points: list[StrokePointRecord] = []
            last_xy: tuple[float, float] | None = None
            for point in raw_points:
                if not isinstance(point, dict):
                    continue
                x = self._clamp(float(point.get("x", 0.0)), 0.0, float(session.target.image_width - 1))
                y = self._clamp(float(point.get("y", 0.0)), 0.0, float(session.target.image_height - 1))
                current_xy = (round(x, 2), round(y, 2))
                if current_xy == last_xy:
                    continue
                points.append(StrokePointRecord(x=x, y=y))
                last_xy = current_xy
            if len(points) < 3:
                continue
            distinct_xy = {(round(float(point.x), 2), round(float(point.y), 2)) for point in points}
            if len(distinct_xy) < 3:
                continue
            polygon_area = 0.0
            for index, point in enumerate(points):
                next_point = points[(index + 1) % len(points)]
                polygon_area += (float(point.x) * float(next_point.y)) - (float(next_point.x) * float(point.y))
            if abs(polygon_area) * 0.5 < 16.0:
                continue
            normalized.append(
                LockedRegionRecord(
                    region_id=str(item.get("region_id") or f"region_{uuid4().hex[:12]}"),
                    created_at=str(item.get("created_at") or utc_now_iso()),
                    source=str(item.get("source", "manual") or "manual"),
                    points=points,
                )
            )
        return normalized

    def _locked_regions_mask(self, session: SessionState, regions: list[LockedRegionRecord]) -> Any:
        runtime = self.inference_service._load_runtime()
        np = runtime["np"]
        if not regions:
            return np.zeros((session.target.image_height, session.target.image_width), dtype=bool)
        canvas = Image.new("1", (session.target.image_width, session.target.image_height), 0)
        drawer = ImageDraw.Draw(canvas)
        for region in regions:
            points = [(float(point.x), float(point.y)) for point in region.points]
            if len(points) < 3:
                continue
            drawer.polygon(points, fill=1, outline=1)
        return np.asarray(canvas, dtype=bool)

    @staticmethod
    def _resize_bool_mask(mask: Any, height: int, width: int) -> Any:
        np = importlib.import_module("numpy")
        mask_image = Image.fromarray(np.asarray(mask).astype("uint8") * 255, mode="L")
        resized = mask_image.resize((int(width), int(height)), resample=Image.NEAREST)
        return np.asarray(resized, dtype=bool)

    def _apply_locked_regions_to_mask_and_logits(
        self,
        session: SessionState,
        mask: Any,
        logits: Any,
    ) -> tuple[Any, Any]:
        runtime = self.inference_service._load_runtime()
        np = runtime["np"]
        if not session.locked_regions:
            return np.asarray(mask).astype(bool), np.asarray(logits, dtype=np.float32)
        locked_mask = self._locked_regions_mask(session, session.locked_regions)
        combined_mask = np.asarray(mask).astype(bool) | locked_mask
        updated_logits = np.asarray(logits, dtype=np.float32).copy()
        if updated_logits.ndim == 3 and updated_logits.shape[0] >= 1:
            logits_locked_mask = self._resize_bool_mask(
                locked_mask,
                updated_logits.shape[-2],
                updated_logits.shape[-1],
            )
            updated_logits[0][logits_locked_mask] = np.maximum(updated_logits[0][logits_locked_mask], 32.0)
        elif updated_logits.ndim == 2:
            logits_locked_mask = self._resize_bool_mask(
                locked_mask,
                updated_logits.shape[0],
                updated_logits.shape[1],
            )
            updated_logits[logits_locked_mask] = np.maximum(updated_logits[logits_locked_mask], 32.0)
        return combined_mask, updated_logits

    def _build_history_from_mask_and_logits(
        self,
        session: SessionState,
        current: HistoryRecord,
        history_id: str,
        name: str,
        kind: str,
        created_at: str,
        mask: Any,
        logits: Any,
        score: float | None = None,
    ) -> HistoryRecord:
        runtime = self.inference_service._load_runtime()
        np = runtime["np"]
        mask_bool = np.asarray(mask).astype(bool)
        return HistoryRecord(
            history_id=history_id,
            parent_history_id=current.history_id if current else None,
            name=name,
            kind=kind,
            created_at=created_at,
            score=float(score) if score is not None else current.score,
            mask_rle=self.inference_service._mask_to_rle(mask_bool),
            mask_area=int(mask_bool.sum()),
            mask_bbox_xywh=self.inference_service._mask_to_xywh(mask_bool),
            prompt_box_xyxy=deepcopy(session.prompt_box_xyxy),
            manual_points_snapshot=copy_points(session.working_points),
            line_strokes_snapshot=copy_line_strokes(session.line_strokes),
            locked_regions_snapshot=copy_locked_regions(session.locked_regions),
            system_prompt_points=copy_points(session.system_prompt_points),
            text_prompt=session.text_prompt,
            used_mask_prompt=True,
            mask_logits_relpath=None,
        )

    def _fallback_logits_from_mask(self, mask: Any) -> Any:
        runtime = self.inference_service._load_runtime()
        np = runtime["np"]
        mask_bool = np.asarray(mask).astype(bool)
        logits = np.full((1, mask_bool.shape[0], mask_bool.shape[1]), -32.0, dtype=np.float32)
        logits[0][mask_bool] = 32.0
        return logits

    def _ensure_history_logits(self, session: SessionState, history_item: HistoryRecord) -> Any:
        if history_item.mask_logits_relpath:
            try:
                return self.session_store.load_logits(session, history_item.mask_logits_relpath)
            except FileNotFoundError:
                pass
        mask = self.inference_service.mask_from_rle(history_item.mask_rle)
        logits = self._fallback_logits_from_mask(mask)
        history_item.mask_logits_relpath = self.session_store.save_logits(session, history_item.history_id, logits)
        self.session_store.save_session(session)
        return logits

    @staticmethod
    def _stroke_to_prompt_points(stroke: LineStrokeRecord, spacing_px: float = 10.0) -> list[PointRecord]:
        raw_points = stroke.points
        if not raw_points:
            return []
        sampled: list[tuple[float, float]] = [(float(raw_points[0].x), float(raw_points[0].y))]
        for start, end in zip(raw_points, raw_points[1:]):
            x0, y0 = float(start.x), float(start.y)
            x1, y1 = float(end.x), float(end.y)
            length = hypot(x1 - x0, y1 - y0)
            steps = max(1, int(length / max(1.0, spacing_px)))
            for step in range(1, steps + 1):
                t = step / steps
                sampled.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t))

        compact: list[tuple[float, float]] = []
        for x, y in sampled:
            if compact and hypot(compact[-1][0] - x, compact[-1][1] - y) < max(1.0, spacing_px * 0.45):
                continue
            compact.append((x, y))

        max_points = 80
        if len(compact) > max_points:
            stride = max(1, len(compact) // max_points)
            compact = compact[::stride][:max_points]
            if compact[-1] != sampled[-1]:
                compact.append(sampled[-1])

        return [
            PointRecord(
                point_id=f"{stroke.stroke_id}_pt_{index}",
                x=float(x),
                y=float(y),
                label=stroke.label,
                created_at=stroke.created_at,
                source="stroke",
            )
            for index, (x, y) in enumerate(compact)
        ]

    def _line_strokes_to_prompt_points(self, strokes: list[LineStrokeRecord]) -> list[PointRecord]:
        points: list[PointRecord] = []
        for stroke in strokes:
            points.extend(self._stroke_to_prompt_points(stroke))
        return points

    def _text_prompt_to_points(
        self,
        session: SessionState,
        text_prompt: str,
    ) -> list[PointRecord]:
        grounded = self.inference_service.ground_text_with_box(
            target=session.target,
            prompt_box_xyxy=session.prompt_box_xyxy,
            text_prompt=text_prompt,
        )
        if not grounded:
            return []
        np = self.inference_service._load_runtime()["np"]
        mask = np.asarray(grounded["mask"]).astype(bool)
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return []
        centroid_x = float(xs.mean())
        centroid_y = float(ys.mean())
        candidate_points: list[tuple[float, float]] = [(centroid_x, centroid_y)]
        extrema = [
            (float(xs[xs.argmin()]), float(ys[xs.argmin()])),
            (float(xs[xs.argmax()]), float(ys[xs.argmax()])),
            (float(xs[ys.argmin()]), float(ys.min())),
            (float(xs[ys.argmax()]), float(ys.max())),
        ]
        for point in extrema:
            if all(hypot(point[0] - other[0], point[1] - other[1]) > 8.0 for other in candidate_points):
                candidate_points.append(point)
        return [
            PointRecord(
                point_id=f"text_prompt_{index}",
                x=float(x),
                y=float(y),
                label=1,
                created_at=utc_now_iso(),
                source="text",
            )
            for index, (x, y) in enumerate(candidate_points[:5])
        ]

    def import_targets(
        self,
        image_file_name: str,
        image_data_url: str,
        annotation_file_name: str,
        annotation_text: str,
    ) -> dict[str, Any]:
        manifest = self.target_store.import_bundle(
            image_file_name=image_file_name,
            image_data_url=image_data_url,
            annotation_file_name=annotation_file_name,
            annotation_text=annotation_text,
        )
        session_meta = self.session_store.list_session_meta()
        targets = [
            self._target_payload(TargetRecord.from_dict(item), session_meta.get(item["key"], {}))
            for item in manifest["targets"]
        ]
        return {
            "import_id": manifest["import_id"],
            "imported_at": manifest["imported_at"],
            "image_file_name": manifest["image_file_name"],
            "annotation_file_name": manifest["annotation_file_name"],
            "targets": targets,
        }

    def get_target_image_path(self, target_key: str) -> Path:
        return Path(self.target_store.get_target(target_key).image_path)

    def get_existing_session_payload(self, target_key: str) -> dict[str, Any] | None:
        session = self.session_store.load_session(target_key)
        if session is None:
            return None
        return self._session_payload(session)

    def open_session(self, target_key: str) -> dict[str, Any]:
        with self._lock:
            existing = self.session_store.load_session(target_key)
            if existing is not None:
                return self._session_payload(existing)

            target = self.target_store.get_target(target_key)
            created_at = utc_now_iso()
            initial = self.inference_service.predict_initial(target)
            history_id = "init"

            session = SessionState(
                schema_version=1,
                session_id=target.key,
                created_at=created_at,
                updated_at=created_at,
                target=target,
                prompt_box_xyxy=initial["prompt_box_xyxy"],
                system_prompt_points=copy_points(initial["system_prompt_points"]),
                working_points=[],
                line_strokes=[],
                locked_regions=[],
                text_prompt="",
                history=[
                    HistoryRecord(
                        history_id=history_id,
                        parent_history_id=None,
                        name="init",
                        kind="initial",
                        created_at=created_at,
                        score=float(initial["score"]),
                        mask_rle=initial["mask_rle"],
                        mask_area=int(initial["mask_area"]),
                        mask_bbox_xywh=initial["mask_bbox_xywh"],
                        prompt_box_xyxy=[float(value) for value in initial["prompt_box_xyxy"]],
                        manual_points_snapshot=[],
                        line_strokes_snapshot=[],
                        locked_regions_snapshot=[],
                        system_prompt_points=copy_points(initial["system_prompt_points"]),
                        text_prompt="",
                        used_mask_prompt=False,
                        mask_logits_relpath=None,
                    )
                ],
                current_history_id=history_id,
            )
            logits_relpath = self.session_store.save_logits(session, history_id, initial["logits"])
            session.history[0].mask_logits_relpath = logits_relpath
            self.session_store.save_session(session)
            return self._session_payload(session)

    def add_point(self, target_key: str, x: float, y: float, label: int) -> dict[str, Any]:
        with self._lock:
            session = self._require_session(target_key)
            point = PointRecord(
                point_id=f"point_{uuid4().hex}",
                x=self._clamp(float(x), 0.0, float(session.target.image_width - 1)),
                y=self._clamp(float(y), 0.0, float(session.target.image_height - 1)),
                label=1 if int(label) == 1 else 0,
                created_at=utc_now_iso(),
                source="manual",
            )
            session.working_points.append(point)
            session.updated_at = utc_now_iso()
            self.session_store.save_session(session)
            return self._session_payload(session)

    def delete_point(self, target_key: str, point_id: str) -> dict[str, Any]:
        with self._lock:
            session = self._require_session(target_key)
            session.working_points = [point for point in session.working_points if point.point_id != point_id]
            session.updated_at = utc_now_iso()
            self.session_store.save_session(session)
            return self._session_payload(session)

    def undo_point(self, target_key: str) -> dict[str, Any]:
        with self._lock:
            session = self._require_session(target_key)
            if session.working_points:
                session.working_points.pop()
                session.updated_at = utc_now_iso()
                self.session_store.save_session(session)
            return self._session_payload(session)

    def clear_points(self, target_key: str) -> dict[str, Any]:
        with self._lock:
            session = self._require_session(target_key)
            session.working_points = []
            session.updated_at = utc_now_iso()
            self.session_store.save_session(session)
            return self._session_payload(session)

    def update_prompt_state(
        self,
        target_key: str,
        text_prompt: str | None,
        line_strokes: Any = None,
    ) -> dict[str, Any]:
        with self._lock:
            session = self._require_session(target_key)
            if text_prompt is not None:
                session.text_prompt = str(text_prompt or "").strip()
            if line_strokes is not None:
                session.line_strokes = self._normalize_line_strokes(session, line_strokes)
            session.updated_at = utc_now_iso()
            self.session_store.save_session(session)
            return self._session_payload(session)

    def lock_region(self, target_key: str, points: Any) -> dict[str, Any]:
        with self._lock:
            session = self._require_session(target_key)
            current = session.current_history()
            normalized = self._normalize_locked_regions(
                session,
                [{"points": points, "source": "manual"}],
            )
            if not normalized:
                raise ValueError("Locked region requires at least 3 distinct points and a non-trivial polygon area.")

            session.locked_regions.append(normalized[0])
            current_mask = self.inference_service.mask_from_rle(current.mask_rle)
            current_logits = self._ensure_history_logits(session, current)
            combined_mask, updated_logits = self._apply_locked_regions_to_mask_and_logits(
                session,
                current_mask,
                current_logits,
            )

            history_index = sum(1 for item in session.history if item.kind == "region_lock") + 1
            history_id = f"lock_{uuid4().hex[:12]}"
            created_at = utc_now_iso()
            new_history = self._build_history_from_mask_and_logits(
                session=session,
                current=current,
                history_id=history_id,
                name=f"lock{history_index}",
                kind="region_lock",
                created_at=created_at,
                mask=combined_mask,
                logits=updated_logits,
                score=current.score,
            )
            session.history.append(new_history)
            logits_relpath = self.session_store.save_logits(session, history_id, updated_logits)
            new_history.mask_logits_relpath = logits_relpath
            session.current_history_id = history_id
            session.updated_at = created_at
            self.session_store.save_session(session)
            return self._session_payload(session)

    def delete_locked_region(self, target_key: str, region_id: str) -> dict[str, Any]:
        with self._lock:
            session = self._require_session(target_key)
            before_count = len(session.locked_regions)
            session.locked_regions = [
                region for region in session.locked_regions if region.region_id != region_id
            ]
            if len(session.locked_regions) == before_count:
                raise KeyError(f"Locked region not found: {region_id}")

            current = session.current_history()
            current_mask = self.inference_service.mask_from_rle(current.mask_rle)
            updated_logits = self._fallback_logits_from_mask(current_mask)
            history_index = sum(1 for item in session.history if item.kind == "region_unlock") + 1
            history_id = f"unlock_{uuid4().hex[:12]}"
            created_at = utc_now_iso()
            new_history = self._build_history_from_mask_and_logits(
                session=session,
                current=current,
                history_id=history_id,
                name=f"unlock{history_index}",
                kind="region_unlock",
                created_at=created_at,
                mask=current_mask,
                logits=updated_logits,
                score=current.score,
            )
            session.history.append(new_history)
            logits_relpath = self.session_store.save_logits(session, history_id, updated_logits)
            new_history.mask_logits_relpath = logits_relpath
            session.current_history_id = history_id
            session.updated_at = created_at
            self.session_store.save_session(session)
            return self._session_payload(session)

    def delete_target(self, target_key: str) -> dict[str, Any]:
        with self._lock:
            session = self._require_session(target_key)
            deleted_at = utc_now_iso()
            session.is_deleted = True
            session.deleted_at = deleted_at
            session.updated_at = deleted_at
            self.session_store.save_session(session)
            return {
                "ok": True,
                "deleted_target_key": target_key,
                "deleted_at": deleted_at,
                "target": session.target.to_dict(),
                "bootstrap": self.bootstrap_payload(),
            }

    def delete_image(self, target_key: str) -> dict[str, Any]:
        with self._lock:
            target = self.target_store.get_target(target_key)
            image_targets = sorted(
                self.target_store.targets_for_image(target),
                key=lambda item: int(item.result_index),
            )
            if not image_targets:
                raise KeyError(f"No targets found for image: {target.image_file_name}")

            deleted_at = utc_now_iso()
            for image_target in image_targets:
                session = self.session_store.load_session(image_target.key)
                if session is None:
                    continue
                session.is_deleted = True
                session.deleted_at = deleted_at
                session.updated_at = deleted_at
                self.session_store.save_session(session)

            record = {
                "schema_version": 1,
                "deletion_id": f"image_delete_{uuid4().hex[:12]}",
                "scope": "image",
                "deleted_at": deleted_at,
                "source": "mask_iteration_webapp",
                "image_file_name": target.image_file_name,
                "image_path": target.image_path,
                "import_id": target.import_id,
                "target_count": len(image_targets),
                "target_keys": [item.key for item in image_targets],
                "annotation_file_names": sorted({item.annotation_file_name for item in image_targets}),
                "annotation_json_paths": sorted({item.annotation_json_path for item in image_targets}),
                "targets": [item.to_dict() for item in image_targets],
            }
            self.session_store.add_image_deletion_record(record)
            return {
                "ok": True,
                "deleted_scope": "image",
                "deleted_at": deleted_at,
                "deleted_image_file_name": target.image_file_name,
                "deleted_target_keys": [item.key for item in image_targets],
                "deleted_target_count": len(image_targets),
                "image_deletion": record,
                "bootstrap": self.bootstrap_payload(),
            }

    def update_metadata(self, target_key: str, variant_name: str | None, notes: str | None) -> dict[str, Any]:
        with self._lock:
            session = self._require_session(target_key)
            if variant_name is not None:
                session.variant_name = (variant_name or "manual_clicks").strip() or "manual_clicks"
            if notes is not None:
                session.notes = notes
            session.updated_at = utc_now_iso()
            self.session_store.save_session(session)
            return self._session_payload(session)

    def iterate(
        self,
        target_key: str,
        text_prompt: str | None = None,
        line_strokes: Any = None,
    ) -> dict[str, Any]:
        with self._lock:
            session = self._require_session(target_key)
            if text_prompt is not None:
                session.text_prompt = str(text_prompt or "").strip()
            if line_strokes is not None:
                session.line_strokes = self._normalize_line_strokes(session, line_strokes)
            current = session.current_history()
            previous_logits = self._ensure_history_logits(session, current)
            effective_points = copy_points(session.working_points)
            effective_points.extend(self._line_strokes_to_prompt_points(session.line_strokes))
            if session.text_prompt:
                effective_points.extend(self._text_prompt_to_points(session, session.text_prompt))
            result = self.inference_service.iterate(
                target=session.target,
                prompt_box_xyxy=session.prompt_box_xyxy,
                working_points=effective_points,
                previous_logits=previous_logits,
            )
            combined_mask, updated_logits = self._apply_locked_regions_to_mask_and_logits(
                session,
                self.inference_service.mask_from_rle(result["mask_rle"]),
                result["logits"],
            )

            history_index = sum(1 for item in session.history if item.kind == "iteration") + 1
            history_id = f"iter_{uuid4().hex[:12]}"
            created_at = utc_now_iso()
            new_history = self._build_history_from_mask_and_logits(
                session=session,
                current=current,
                history_id=history_id,
                name=f"a{history_index}",
                kind="iteration",
                created_at=created_at,
                mask=combined_mask,
                logits=updated_logits,
                score=float(result["score"]),
            )
            session.history.append(new_history)
            logits_relpath = self.session_store.save_logits(session, history_id, updated_logits)
            new_history.mask_logits_relpath = logits_relpath
            session.current_history_id = history_id
            session.updated_at = created_at
            self.session_store.save_session(session)
            return self._session_payload(session)

    def rollback(self, target_key: str, history_id: str) -> dict[str, Any]:
        with self._lock:
            session = self._require_session(target_key)
            history_item = session.history_by_id(history_id)
            session.current_history_id = history_item.history_id
            session.working_points = copy_points(history_item.manual_points_snapshot)
            session.line_strokes = copy_line_strokes(history_item.line_strokes_snapshot)
            session.locked_regions = copy_locked_regions(history_item.locked_regions_snapshot)
            session.text_prompt = history_item.text_prompt
            session.updated_at = utc_now_iso()
            self.session_store.save_session(session)
            return self._session_payload(session)

    def export_legacy_payload(self, target_key: str) -> tuple[dict[str, Any], str]:
        session = self._require_session(target_key)
        current = session.current_history()
        payload = {
            "variants": [
                {
                    "name": session.variant_name,
                    "points": [{"x": point.x, "y": point.y, "label": point.label} for point in session.working_points],
                    "line_strokes": [stroke.to_dict() for stroke in session.line_strokes],
                    "locked_regions": [region.to_dict() for region in session.locked_regions],
                    "text_prompt": session.text_prompt or None,
                    "notes": session.notes or None,
                }
            ],
            "target": {
                "annotation_id": session.target.annotation_id,
                "source_annotation_id": session.target.source_annotation_id,
                "category_id": session.target.category_id,
                "category_name": session.target.category_name,
                "image_file_name": session.target.image_file_name,
                "image_width": session.target.image_width,
                "image_height": session.target.image_height,
                "bbox_xywh": session.target.bbox_xywh,
                "bbox_xyxy": session.target.bbox_xyxy,
                "source_json": session.target.annotation_json_path,
                "sort_index": session.target.sort_index,
                "import_id": session.target.import_id,
                "imported_at": session.target.imported_at,
            },
            "meta": {
                "annotation_file_name": session.target.annotation_file_name,
                "exported_at": utc_now_iso(),
                "session_id": session.session_id,
                "current_history_id": current.history_id,
                "current_history_name": current.name,
                "current_mask_area": current.mask_area,
                "prompt_box_xyxy": session.prompt_box_xyxy,
                "text_prompt": session.text_prompt,
                "is_deleted": session.is_deleted,
                "deleted_at": session.deleted_at,
                "source": "mask_iteration_webapp",
            },
        }
        export_name = f"{sanitize_component(session.target.image_file_name)}__{sanitize_component(session.target.annotation_id)}__legacy.json"
        self.session_store.write_export(session, export_name, payload)
        return payload, export_name

    def export_session_payload(self, target_key: str) -> tuple[dict[str, Any], str]:
        session = self._require_session(target_key)
        payload = {
            "meta": {
                "exported_at": utc_now_iso(),
                "source": "mask_iteration_webapp",
            },
            "session": session.to_dict(),
        }
        export_name = f"{sanitize_component(session.target.image_file_name)}__{sanitize_component(session.target.annotation_id)}__session.json"
        self.session_store.write_export(session, export_name, payload)
        return payload, export_name

    def export_deleted_targets_payload(self) -> tuple[dict[str, Any], str]:
        deleted_targets: list[dict[str, Any]] = []
        grouped: dict[str, dict[str, Any]] = {}
        image_deletions = self.session_store.list_image_deletion_records()

        for path in sorted(self.session_store.sessions_root.glob("*/session.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                session = SessionState.from_dict(payload)
            except Exception:
                continue
            if not session.is_deleted:
                continue

            target = session.target
            record = {
                "session_id": session.session_id,
                "target_key": target.key,
                "annotation_id": target.annotation_id,
                "source_annotation_id": target.source_annotation_id,
                "result_index": target.result_index,
                "category_id": target.category_id,
                "category_name": target.category_name,
                "image_file_name": target.image_file_name,
                "annotation_file_name": target.annotation_file_name,
                "annotation_json_path": target.annotation_json_path,
                "import_id": target.import_id,
                "imported_at": target.imported_at,
                "deleted_at": session.deleted_at,
                "bbox_xywh": target.bbox_xywh,
                "bbox_xyxy": target.bbox_xyxy,
            }
            deleted_targets.append(record)

            group_key = target.annotation_json_path
            group = grouped.setdefault(
                group_key,
                {
                    "annotation_file_name": target.annotation_file_name,
                    "annotation_json_path": target.annotation_json_path,
                    "image_file_name": target.image_file_name,
                    "deleted_count": 0,
                    "delete_result_indices": [],
                    "delete_annotation_ids": [],
                    "deleted_targets": [],
                },
            )
            group["deleted_count"] += 1
            group["delete_result_indices"].append(target.result_index)
            group["delete_annotation_ids"].append(target.annotation_id)
            group["deleted_targets"].append(record)

        deleted_targets.sort(
            key=lambda item: (
                str(item.get("deleted_at") or ""),
                str(item.get("annotation_file_name") or ""),
                int(item.get("result_index") or 0),
            ),
            reverse=True,
        )
        grouped_records = []
        for entry in grouped.values():
            entry["delete_result_indices"] = sorted({int(value) for value in entry["delete_result_indices"]}, reverse=True)
            entry["delete_annotation_ids"] = sorted({str(value) for value in entry["delete_annotation_ids"]})
            entry["deleted_targets"] = sorted(
                entry["deleted_targets"],
                key=lambda item: int(item.get("result_index") or 0),
                reverse=True,
            )
            grouped_records.append(entry)
        grouped_records.sort(key=lambda item: str(item.get("annotation_file_name") or ""))

        payload = {
            "meta": {
                "exported_at": utc_now_iso(),
                "source": "mask_iteration_webapp",
                "deleted_target_count": len(deleted_targets),
                "deleted_image_count": len(image_deletions),
                "annotation_file_count": len(grouped_records),
            },
            "deleted_images": image_deletions,
            "deleted_targets": deleted_targets,
            "by_annotation_file": grouped_records,
        }
        file_name = f"deleted_targets__{sanitize_component(utc_now_iso())}.json"
        return payload, file_name

    def _require_session(self, target_key: str) -> SessionState:
        session = self.session_store.load_session(target_key)
        if session is None:
            raise FileNotFoundError(f"Session not found for target {target_key}")
        return session

    def _session_payload(self, session: SessionState) -> dict[str, Any]:
        current = session.current_history()
        return {
            "session": {
                "schema_version": session.schema_version,
                "session_id": session.session_id,
                "created_at": session.created_at,
                "updated_at": session.updated_at,
                "variant_name": session.variant_name,
                "notes": session.notes,
                "target": session.target.to_dict(),
                "prompt_box_xyxy": session.prompt_box_xyxy,
                "system_prompt_points": [point.to_dict() for point in session.system_prompt_points],
                "working_points": [point.to_dict() for point in session.working_points],
                "line_strokes": [stroke.to_dict() for stroke in session.line_strokes],
                "locked_regions": [region.to_dict() for region in session.locked_regions],
                "text_prompt": session.text_prompt,
                "current_history_id": session.current_history_id,
                "current_history_name": current.name,
                "history": [item.to_dict() for item in session.history],
                "is_deleted": session.is_deleted,
                "deleted_at": session.deleted_at,
            }
            ,
            "validate_tools": self._validate_tools_session_payload(session),
        }

    @staticmethod
    def _target_payload(target: TargetRecord, meta: dict[str, Any]) -> dict[str, Any]:
        return {
            "key": target.key,
            "category_name": target.category_name,
            "image_file_name": target.image_file_name,
            "annotation_file_name": target.annotation_file_name,
            "annotation_id": target.annotation_id,
            "bbox_xywh": target.bbox_xywh,
            "bbox_xyxy": target.bbox_xyxy,
            "sort_index": target.sort_index,
            "import_id": target.import_id,
            "imported_at": target.imported_at,
            "has_session": bool(meta.get("has_session")),
            "history_count": int(meta.get("history_count", 0)),
            "updated_at": meta.get("updated_at"),
            "is_deleted": bool(meta.get("is_deleted", False)),
            "deleted_at": meta.get("deleted_at"),
        }

    @staticmethod
    def _clamp(value: float, lower: float, upper: float) -> float:
        return min(max(value, lower), upper)
