#!/usr/bin/env python
"""Analyze likely causes of count discrepancies in COCO-style annotations."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(r"C:\Users\admin\Desktop\标注")
DIFFS = {
    "book": 2353,
    "orange": 1377,
    "banana": 652,
    "cup": 270,
    "pens": 174,
    "bowl": 105,
    "bottle": 87,
    "switch": 85,
    "apple": 2,
}


def truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


total = Counter()
all_annotations = Counter()
invalid_bbox = Counter()
zero_size_bbox = Counter()
groupof = Counter()
occluded = Counter()
truncated = Counter()
zero_files = []
files_by_category = defaultdict(Counter)
groupof_files = defaultdict(list)

for path in sorted(ROOT.rglob("*.json")):
    data = json.loads(path.read_text(encoding="utf-8"))
    categories = {item.get("id"): item.get("name") for item in data.get("categories", [])}
    annotations = data.get("annotations", []) or []
    if not annotations:
        zero_files.append(str(path))
    group_name = path.relative_to(ROOT).parts[0] if len(path.relative_to(ROOT).parts) else ""
    for annotation in annotations:
        raw_name = categories.get(annotation.get("category_id"), f"unknown:{annotation.get('category_id')}")
        all_annotations[raw_name] += 1
        bbox = annotation.get("bbox")
        if not isinstance(bbox, list) or len(bbox) < 4:
            invalid_bbox[raw_name] += 1
            continue
        name = raw_name
        try:
            if float(bbox[2]) <= 0 or float(bbox[3]) <= 0:
                zero_size_bbox[name] += 1
        except (TypeError, ValueError):
            invalid_bbox[name] += 1
            continue
        total[name] += 1
        files_by_category[name][group_name] += 1
        if truthy(annotation.get("GroupOf")):
            groupof[name] += 1
            groupof_files[name].append(str(path))
        if truthy(annotation.get("Occluded")):
            occluded[name] += 1
        if truthy(annotation.get("Truncated")):
            truncated[name] += 1

print("category\tjson_count\tdiff\tgroupof\toccluded\ttruncated\tby_group")
for name, diff in DIFFS.items():
    by_group = ", ".join(f"{group}:{count}" for group, count in files_by_category[name].most_common())
    print(f"{name}\t{total[name]}\t{diff}\t{groupof[name]}\t{occluded[name]}\t{truncated[name]}\t{by_group}")

print()
print(f"zero_annotation_files\t{len(zero_files)}")
for path in zero_files:
    print(path)

print()
print("categories_with_groupof")
for name, count in groupof.most_common():
    print(f"{name}\t{count}")

print()
print(f"all_annotations\t{sum(all_annotations.values())}")
print(f"valid_bbox_annotations\t{sum(total.values())}")
print(f"invalid_bbox_annotations\t{sum(invalid_bbox.values())}")
print(f"zero_size_bbox_annotations\t{sum(zero_size_bbox.values())}")
for name, count in invalid_bbox.most_common():
    print(f"invalid_bbox\t{name}\t{count}")
for name, count in zero_size_bbox.most_common():
    print(f"zero_size_bbox\t{name}\t{count}")
