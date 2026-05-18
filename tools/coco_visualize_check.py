from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from pycocotools.coco import COCO


MASK_COLOR = np.asarray([48, 208, 127], dtype=np.float32)
BBOX_COLOR = (0, 194, 255)
ERROR_COLOR = (255, 106, 89)


def iter_coco_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(item for item in path.rglob("*.json") if item.is_file())


def find_image_path(images_root: Path | None, annotation_path: Path, file_name: str) -> Path | None:
    if not file_name:
        return None
    candidates: list[Path] = []
    raw = Path(file_name)
    if raw.is_absolute():
        candidates.append(raw)
    if images_root is not None:
        candidates.append(images_root / file_name)
        candidates.append(images_root / raw.name)
    candidates.append(annotation_path.parent / file_name)
    candidates.append(annotation_path.parent / raw.name)
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def canvas_for_image(annotation_path: Path, images_root: Path | None, image_info: dict[str, Any]) -> Image.Image:
    image_path = find_image_path(images_root, annotation_path, str(image_info.get("file_name") or ""))
    if image_path is not None:
        return Image.open(image_path).convert("RGB")
    width = int(image_info.get("width") or 1)
    height = int(image_info.get("height") or 1)
    return Image.new("RGB", (max(1, width), max(1, height)), (245, 247, 250))


def draw_mask(image: Image.Image, mask: np.ndarray, alpha: float = 0.45) -> None:
    mask_bool = np.asarray(mask).astype(bool)
    if mask_bool.shape[:2] != (image.height, image.width):
        return
    arr = np.asarray(image).astype(np.float32)
    arr[mask_bool] = arr[mask_bool] * (1.0 - alpha) + MASK_COLOR * alpha
    image.paste(Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)))


def draw_bbox(draw: ImageDraw.ImageDraw, bbox: Any, color: tuple[int, int, int], label: str) -> None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return
    x, y, w, h = [float(value) for value in bbox[:4]]
    xy = [x, y, x + w, y + h]
    draw.rectangle(xy, outline=color, width=3)
    draw.text((x + 3, y + 3), label, fill=color)


def visualize_one(annotation_path: Path, output_dir: Path, images_root: Path | None) -> dict[str, Any]:
    coco = COCO(str(annotation_path))
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = {
        "file": str(annotation_path),
        "images": len(coco.imgs),
        "annotations": len(coco.anns),
        "decoded_masks": 0,
        "no_mask": 0,
        "decode_errors": [],
        "outputs": [],
    }

    for image_id, image_info in sorted(coco.imgs.items(), key=lambda item: str(item[0])):
        image = canvas_for_image(annotation_path, images_root, image_info)
        draw = ImageDraw.Draw(image)
        for ann in coco.loadAnns(coco.getAnnIds(imgIds=[image_id])):
            ann_id = ann.get("id")
            segmentation = ann.get("segmentation")
            if not segmentation:
                stats["no_mask"] += 1
                draw_bbox(draw, ann.get("bbox"), BBOX_COLOR, f"id={ann_id} no-mask")
                continue
            try:
                mask = coco.annToMask(ann)
            except Exception as error:
                stats["decode_errors"].append(
                    {"annotation_id": ann_id, "error": f"{error.__class__.__name__}: {error}"}
                )
                draw_bbox(draw, ann.get("bbox"), ERROR_COLOR, f"id={ann_id} decode-error")
                continue
            stats["decoded_masks"] += 1
            draw_mask(image, mask)
            draw_bbox(draw, ann.get("bbox"), tuple(MASK_COLOR.astype(int).tolist()), f"id={ann_id} mask")

        suffix = image_id if len(coco.imgs) > 1 else "overlay"
        output_path = output_dir / f"{annotation_path.stem}__{suffix}.png"
        image.save(output_path)
        stats["outputs"].append(str(output_path))
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and visualize COCO annotations with pycocotools.")
    parser.add_argument("coco", type=Path, help="COCO JSON file or directory containing COCO JSON files.")
    parser.add_argument("--images-root", type=Path, default=None, help="Optional image root for real-image overlays.")
    parser.add_argument("--output-dir", type=Path, default=Path("coco_visual_check"), help="Overlay output directory.")
    parser.add_argument("--report-json", type=Path, default=None, help="Optional path for the JSON summary report.")
    args = parser.parse_args()

    reports = [
        visualize_one(path, args.output_dir, args.images_root)
        for path in iter_coco_paths(args.coco)
    ]
    summary = {
        "files": len(reports),
        "annotations": sum(item["annotations"] for item in reports),
        "decoded_masks": sum(item["decoded_masks"] for item in reports),
        "no_mask": sum(item["no_mask"] for item in reports),
        "decode_errors": sum(len(item["decode_errors"]) for item in reports),
        "reports": reports,
    }
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(text)
    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
