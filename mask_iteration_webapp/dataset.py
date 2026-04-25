from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .models import TargetRecord


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def sanitize_component(text: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in str(text))
    cleaned = cleaned.strip("_")
    return cleaned or "item"


def round_floats(values: list[float], digits: int = 3) -> list[float]:
    return [round(float(value), digits) for value in values]


def expand_box_xyxy(box_xyxy: list[float], expand_px: float, width: int, height: int) -> list[float]:
    x0, y0, x1, y1 = [float(value) for value in box_xyxy]
    return [
        max(0.0, x0 - expand_px),
        max(0.0, y0 - expand_px),
        min(float(width - 1), x1 + expand_px),
        min(float(height - 1), y1 + expand_px),
    ]


class LabelStudioDatasetIndex:
    def __init__(self, annotations_dir: Path, image_search_root: Path):
        self.annotations_dir = annotations_dir
        self.image_search_root = image_search_root
        self._targets: list[TargetRecord] = []
        self._targets_by_key: dict[str, TargetRecord] = {}
        self._categories: list[str] = []

    @property
    def targets(self) -> list[TargetRecord]:
        return self._targets

    @property
    def categories(self) -> list[str]:
        return self._categories

    def get_target(self, key: str) -> TargetRecord:
        return self._targets_by_key[key]

    def build(self) -> None:
        image_index = self._build_image_index()
        targets: list[TargetRecord] = []
        seen_keys: set[str] = set()
        sort_index = 0

        for annotation_json in sorted(self.annotations_dir.glob("*.json")):
            payload = json.loads(annotation_json.read_text(encoding="utf-8"))
            image_filename = str(payload.get("image_filename", "")).strip()
            if not image_filename:
                continue
            image_path = self._resolve_image_path(image_filename, image_index)

            for result_index, result in enumerate(payload.get("results", [])):
                if result.get("type") != "rectanglelabels":
                    continue
                value = result.get("value") or {}
                labels = value.get("rectanglelabels") or []
                category_name = str(labels[0]).strip() if labels else "object"
                image_width = int(result.get("original_width") or payload.get("image_width") or 0)
                image_height = int(result.get("original_height") or payload.get("image_height") or 0)
                if image_width <= 0 or image_height <= 0:
                    continue

                x = float(value["x"]) * image_width / 100.0
                y = float(value["y"]) * image_height / 100.0
                w = float(value["width"]) * image_width / 100.0
                h = float(value["height"]) * image_height / 100.0
                annotation_id = str(result.get("id", f"result_{result_index}"))
                key = sanitize_component(
                    f"{annotation_json.stem}__{category_name}__{annotation_id}__{result_index}"
                )
                if key in seen_keys:
                    key = sanitize_component(f"{key}__dup_{result_index}")
                seen_keys.add(key)

                target = TargetRecord(
                    key=key,
                    annotation_file_name=annotation_json.name,
                    annotation_json_path=str(annotation_json.resolve()),
                    image_path=str(image_path),
                    image_file_name=image_filename,
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
                )
                sort_index += 1
                targets.append(target)

        targets.sort(key=lambda item: (item.category_name.lower(), item.image_file_name.lower(), item.sort_index))
        self._targets = targets
        self._targets_by_key = {item.key: item for item in targets}
        self._categories = sorted({item.category_name for item in targets}, key=str.lower)

    def _build_image_index(self) -> dict[str, list[Path]]:
        image_index: dict[str, list[Path]] = defaultdict(list)
        for path in self.image_search_root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if self.annotations_dir in path.parents:
                continue
            image_index[path.name].append(path.resolve())
        return image_index

    def _resolve_image_path(self, image_filename: str, image_index: dict[str, list[Path]]) -> Path:
        direct = Path(image_filename)
        if direct.is_file():
            return direct.resolve()

        basename = Path(image_filename).name
        candidates = image_index.get(basename, [])
        if not candidates:
            raise FileNotFoundError(
                f"Image '{image_filename}' from annotations was not found under {self.image_search_root}"
            )
        preferred = [
            candidate
            for candidate in candidates
            if not any(token in str(candidate).lower() for token in ("_vote", "overlay", "iteration", "contact_sheet"))
        ]
        if preferred:
            candidates = preferred
        if len(candidates) == 1:
            return candidates[0]

        exact_suffix = image_filename.replace("\\", "/").strip("/")
        for candidate in candidates:
            normalized = str(candidate).replace("\\", "/")
            if normalized.endswith(exact_suffix):
                return candidate
        return candidates[0]
