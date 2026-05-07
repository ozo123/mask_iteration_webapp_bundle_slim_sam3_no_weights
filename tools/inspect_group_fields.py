#!/usr/bin/env python
"""Inspect group-related fields in annotation JSON files."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(r"C:\Users\admin\Desktop\标注")


def scalar_repr(value: Any) -> str:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return repr(value)
    return f"<{type(value).__name__}>"


def truthy_group_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "none", "null", "no"}
    if isinstance(value, (list, dict)):
        return bool(value)
    return value is not None


def walk(value: Any, prefix: str = ""):
    if isinstance(value, dict):
        for key, child in value.items():
            key_path = f"{prefix}.{key}" if prefix else str(key)
            yield key_path, child
            yield from walk(child, key_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            key_path = f"{prefix}[]" if prefix else "[]"
            yield from walk(child, key_path)


group_field_values: dict[str, Counter[str]] = defaultdict(Counter)
annotation_group_values: dict[str, Counter[str]] = defaultdict(Counter)
annotation_keys = Counter()
nonzero_group_annotations = []
total_files = 0
total_annotations = 0

for path in sorted(ROOT.rglob("*.json")):
    total_files += 1
    data = json.loads(path.read_text(encoding="utf-8"))
    annotations = data.get("annotations") or []
    total_annotations += len(annotations)

    for key_path, value in walk(data):
        if "group" in key_path.lower():
            group_field_values[key_path][scalar_repr(value)] += 1

    for ann in annotations:
        if not isinstance(ann, dict):
            continue
        for key, value in ann.items():
            annotation_keys[key] += 1
            if "group" in str(key).lower():
                annotation_group_values[str(key)][scalar_repr(value)] += 1
                if truthy_group_value(value):
                    nonzero_group_annotations.append((str(path), str(key), value, ann.get("category_id"), ann.get("bbox")))

print(f"files\t{total_files}")
print(f"annotations\t{total_annotations}")
print()
print("annotation_keys")
for key, count in annotation_keys.most_common():
    print(f"{key}\t{count}")

print()
print("all_group_related_paths")
if not group_field_values:
    print("<none>")
for key_path in sorted(group_field_values):
    values = group_field_values[key_path]
    compact = ", ".join(f"{value}:{count}" for value, count in values.most_common(20))
    print(f"{key_path}\t{compact}")

print()
print("annotation_group_fields")
if not annotation_group_values:
    print("<none>")
for key in sorted(annotation_group_values):
    values = annotation_group_values[key]
    compact = ", ".join(f"{value}:{count}" for value, count in values.most_common(20))
    print(f"{key}\t{compact}")

print()
print(f"truthy_annotation_group_values\t{len(nonzero_group_annotations)}")
for path, key, value, category_id, bbox in nonzero_group_annotations[:50]:
    print(f"{path}\t{key}={value!r}\tcategory_id={category_id!r}\tbbox={bbox!r}")
