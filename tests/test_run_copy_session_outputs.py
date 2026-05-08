import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mask_iteration_webapp.models import HistoryRecord, SessionState, TargetRecord
from mask_iteration_webapp.service import (
    RUN_COCO_DIR,
    RUN_DELETE_DIR,
    RUN_KEEP_DIR,
    RUN_STATE_DIR,
    MaskIterationService,
    SessionStore,
    UploadedTargetStore,
)


class DummyInference:
    reference_box_expand_px = 10
    _runtime = {}

    def readiness(self):
        return {"dummy": True}

    def _load_runtime(self):
        return {"np": np, "mask_utils": None}

    def mask_from_rle(self, rle):
        height, width = rle["size"]
        mask = np.zeros((height, width), dtype=bool)
        for y, x in rle.get("points", []):
            mask[int(y), int(x)] = True
        return mask

    def _mask_to_rle(self, mask):
        ys, xs = np.where(np.asarray(mask).astype(bool))
        return {"size": [int(mask.shape[0]), int(mask.shape[1])], "points": [[int(y), int(x)] for y, x in zip(ys, xs)]}

    def _mask_to_xywh(self, mask):
        ys, xs = np.where(np.asarray(mask).astype(bool))
        if len(xs) == 0:
            return None
        return [float(xs.min()), float(ys.min()), float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)]


def _target(tmp_path):
    coco_path = tmp_path / "runs" / "copy_a" / "annotations" / RUN_KEEP_DIR / RUN_COCO_DIR / "ann.json"
    image_path = tmp_path / "runs" / "copy_a" / "images" / RUN_KEEP_DIR / "a.png"
    coco_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"not-used")
    coco_path.write_text(
        json.dumps(
            {
                "info": {"year": 2026, "version": "1.0", "description": "", "contributor": "", "url": "", "date_created": "2026"},
                "images": [{"id": 1, "height": 4, "width": 4, "file_name": "a.png"}],
                "categories": [{"id": 1, "name": "box", "supercategory": "box"}],
                "annotations": [{"id": 101, "image_id": 1, "category_id": 1, "segmentation": [], "area": 4, "bbox": [0, 0, 2, 2], "iscrowd": 0}],
            }
        ),
        encoding="utf-8",
    )
    return TargetRecord(
        key="target_a",
        annotation_file_name="ann.json",
        annotation_json_path=str(coco_path),
        image_path=str(image_path),
        image_file_name="a.png",
        annotation_id="101",
        source_annotation_id="101",
        result_index=0,
        category_name="box",
        category_id=1,
        image_width=4,
        image_height=4,
        bbox_xywh=[0, 0, 2, 2],
        bbox_xyxy=[0, 0, 2, 2],
        sort_index=0,
        import_id="copy_a",
        imported_at="2026-01-01T00:00:00+00:00",
    )


def _history(history_id, parent, point):
    return HistoryRecord(
        history_id=history_id,
        parent_history_id=parent,
        name=history_id,
        kind="iteration" if parent else "initial",
        created_at="2026-01-01T00:00:00+00:00",
        score=0.9,
        mask_rle={"size": [4, 4], "points": [point]},
        mask_area=1,
        mask_bbox_xywh=[point[1], point[0], 1, 1],
        prompt_box_xyxy=[0, 0, 3, 3],
    )


def _session(target, current_history_id="iter1"):
    return SessionState(
        schema_version=1,
        session_id=target.key,
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
        target=target,
        prompt_box_xyxy=[0, 0, 3, 3],
        system_prompt_points=[],
        working_points=[],
        line_strokes=[],
        locked_regions=[],
        text_prompt="",
        history=[_history("init", None, [0, 0]), _history("iter1", "init", [2, 2])],
        current_history_id=current_history_id,
    )


def test_coco_segmentation_and_state_use_current_history(tmp_path):
    target_store = UploadedTargetStore(tmp_path / "runs")
    service = MaskIterationService(target_store, SessionStore(tmp_path / "sessions"), DummyInference())
    session = _session(_target(tmp_path), current_history_id="init")

    service._save_session_outputs(session)

    coco = json.loads(Path(session.target.annotation_json_path).read_text(encoding="utf-8"))
    assert coco["annotations"][0]["segmentation"]["points"] == [[0, 0]]
    state_path = tmp_path / "runs" / "copy_a" / "annotations" / RUN_KEEP_DIR / RUN_STATE_DIR / "a.json"
    state_payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert state_payload["format"] == "mask_iteration_state"
    assert len(state_payload["sessions"]) == 1
    assert [item["history_id"] for item in state_payload["sessions"][0]["history"]] == ["init", "iter1"]
    assert state_payload["sessions"][0]["history"][1]["mask_rle"]["points"] == [[2, 2]]


def test_delete_history_allows_any_non_init_history(tmp_path):
    target_store = UploadedTargetStore(tmp_path / "runs")
    service = MaskIterationService(target_store, SessionStore(tmp_path / "sessions"), DummyInference())
    session = _session(_target(tmp_path), current_history_id="init")
    session.history.append(_history("iter2", "iter1", [3, 3]))
    service._save_session_outputs(session)

    payload = service.delete_history_items("target_a", ["iter1"])
    assert [item["history_id"] for item in payload["session"]["history"]] == ["init", "iter2"]

    payload = service.delete_history_items("target_a", ["iter2"])
    assert [item["history_id"] for item in payload["session"]["history"]] == ["init"]


def test_delete_current_history_falls_back_to_latest_remaining(tmp_path):
    target_store = UploadedTargetStore(tmp_path / "runs")
    service = MaskIterationService(target_store, SessionStore(tmp_path / "sessions"), DummyInference())
    session = _session(_target(tmp_path), current_history_id="iter2")
    session.history.append(_history("iter2", "iter1", [3, 3]))
    service._save_session_outputs(session)

    payload = service.delete_history_items("target_a", ["iter2"])

    assert [item["history_id"] for item in payload["session"]["history"]] == ["init", "iter1"]
    assert payload["session"]["current_history_id"] == "iter1"


def test_delete_target_writes_deleted_coco_and_state(tmp_path):
    target_store = UploadedTargetStore(tmp_path / "runs")
    service = MaskIterationService(target_store, SessionStore(tmp_path / "sessions"), DummyInference())
    session = _session(_target(tmp_path), current_history_id="iter1")
    service._save_session_outputs(session)

    service.delete_target("target_a")

    kept_coco_path = Path(session.target.annotation_json_path)
    kept_coco = json.loads(kept_coco_path.read_text(encoding="utf-8"))
    assert kept_coco["annotations"] == []

    deleted_coco_path = tmp_path / "runs" / "copy_a" / "annotations" / RUN_DELETE_DIR / RUN_COCO_DIR / "ann.json"
    deleted_coco = json.loads(deleted_coco_path.read_text(encoding="utf-8"))
    assert deleted_coco["annotations"][0]["id"] == 101
    assert deleted_coco["annotations"][0]["segmentation"]["points"] == [[2, 2]]

    kept_state_path = tmp_path / "runs" / "copy_a" / "annotations" / RUN_KEEP_DIR / RUN_STATE_DIR / "a.json"
    deleted_state_path = tmp_path / "runs" / "copy_a" / "annotations" / RUN_DELETE_DIR / RUN_STATE_DIR / "a.json"
    assert not kept_state_path.exists()
    deleted_state = json.loads(deleted_state_path.read_text(encoding="utf-8"))
    assert deleted_state["sessions"][0]["is_deleted"] is True


def test_open_session_restores_from_run_state_when_session_cache_is_missing(tmp_path):
    target_store = UploadedTargetStore(tmp_path / "runs")
    target = _target(tmp_path)
    target_store._targets_by_key[target.key] = target
    service = MaskIterationService(target_store, SessionStore(tmp_path / "sessions_a"), DummyInference())
    session = _session(target, current_history_id="iter1")
    service._save_session_outputs(session)

    restored_service = MaskIterationService(target_store, SessionStore(tmp_path / "sessions_b"), DummyInference())
    payload = restored_service.open_session("target_a")

    assert payload["session"]["current_history_id"] == "iter1"
    assert [item["history_id"] for item in payload["session"]["history"]] == ["init", "iter1"]
