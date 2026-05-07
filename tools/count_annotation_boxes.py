#!/usr/bin/env python
"""Count bounding boxes by category in annotation JSON files.

Supports COCO-style JSON files with ``categories`` and ``annotations``.
Also supports Label Studio-style rectangle labels as a fallback.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _has_valid_bbox(annotation: dict[str, Any]) -> bool:
    bbox = annotation.get("bbox")
    return isinstance(bbox, list) and len(bbox) >= 4


def _category_name(category_id: Any, categories: dict[Any, str]) -> str:
    if category_id in categories:
        return categories[category_id]
    return f"unknown:{category_id}"


def count_coco(data: dict[str, Any]) -> tuple[Counter[str], dict[str, set[Any]]]:
    categories = {}
    for category in data.get("categories") or []:
        if not isinstance(category, dict):
            continue
        category_id = category.get("id")
        name = category.get("name")
        if category_id is not None and name:
            categories[category_id] = str(name)

    counts: Counter[str] = Counter()
    ids_by_name: dict[str, set[Any]] = defaultdict(set)
    for category_id, name in categories.items():
        ids_by_name[name].add(category_id)

    for annotation in data.get("annotations") or []:
        if not isinstance(annotation, dict) or not _has_valid_bbox(annotation):
            continue
        category_id = annotation.get("category_id")
        name = _category_name(category_id, categories)
        counts[name] += 1
        ids_by_name[name].add(category_id)
    return counts, ids_by_name


def count_label_studio(data: Any) -> tuple[Counter[str], dict[str, set[Any]]]:
    counts: Counter[str] = Counter()
    ids_by_name: dict[str, set[Any]] = defaultdict(set)
    items = data if isinstance(data, list) else [data]

    for item in items:
        if not isinstance(item, dict):
            continue
        annotations = item.get("annotations") or item.get("completions") or []
        if not annotations and "result" in item:
            annotations = [item]
        for annotation in annotations:
            if not isinstance(annotation, dict):
                continue
            results = annotation.get("result") or []
            for result in results:
                if not isinstance(result, dict):
                    continue
                value = result.get("value") or {}
                labels = value.get("rectanglelabels") or value.get("labels") or []
                has_box = all(key in value for key in ("x", "y", "width", "height"))
                if not has_box:
                    continue
                for label in labels:
                    counts[str(label)] += 1
                    ids_by_name[str(label)].add(str(label))
    return counts, ids_by_name


def count_file(path: Path) -> tuple[Counter[str], dict[str, set[Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if isinstance(data, dict) and "annotations" in data and "categories" in data:
        return count_coco(data)
    return count_label_studio(data)


def main() -> int:
    parser = argparse.ArgumentParser(description="Count annotation boxes by category.")
    parser.add_argument("annotations_root", help="Folder containing annotation JSON files.")
    args = parser.parse_args()

    root = Path(args.annotations_root).expanduser().resolve()
    json_paths = sorted(root.rglob("*.json"))
    total_counts: Counter[str] = Counter()
    ids_by_name: dict[str, set[Any]] = defaultdict(set)
    failed: list[tuple[Path, str]] = []

    for path in json_paths:
        try:
            counts, ids = count_file(path)
        except Exception as exc:  # noqa: BLE001 - keep batch reporting useful.
            failed.append((path, str(exc)))
            continue
        total_counts.update(counts)
        for name, values in ids.items():
            ids_by_name[name].update(values)

    total_boxes = sum(total_counts.values())
    print(f"Annotation root: {root}")
    print(f"JSON files: {len(json_paths)}")
    print(f"Total boxes: {total_boxes}")
    print()
    print("Category\tBoxes")
    all_names = set(total_counts) | set(ids_by_name)
    for name in sorted(all_names, key=lambda item: (-total_counts[item], item.lower())):
        print(f"{name}\t{total_counts[name]}")

    if failed:
        print()
        print("Failed files:")
        for path, error in failed:
            print(f"{path}\t{error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
