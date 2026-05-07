import base64
import json
import sys
from io import BytesIO
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mask_iteration_webapp.service import RUN_COCO_DIR, RUN_KEEP_DIR, UploadedTargetStore


def _png_data_url(color="white"):
    image = Image.new("RGB", (10, 10), color)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _coco(annotation_id=101, image_name="a.png"):
    return {
        "images": [{"id": 1, "file_name": image_name, "width": 10, "height": 10}],
        "annotations": [{"id": annotation_id, "image_id": 1, "category_id": 1, "bbox": [1, 1, 4, 4]}],
        "categories": [{"id": 1, "name": "box"}],
    }


def test_same_image_folder_can_use_independent_annotation_states(tmp_path):
    store = UploadedTargetStore(tmp_path / "runs")

    state_a = store.import_bundle(
        image_file_name="a.png",
        image_data_url=_png_data_url("white"),
        annotation_file_name="ann.json",
        annotation_text=json.dumps(_coco(101)),
        image_set_id="same_images",
        annotation_state_id="state_a",
    )
    state_b = store.import_bundle(
        image_file_name="a.png",
        image_data_url=_png_data_url("white"),
        annotation_file_name="ann.json",
        annotation_text=json.dumps(_coco(201)),
        image_set_id="same_images",
        annotation_state_id="state_b",
    )

    state_a_annotation = tmp_path / "runs" / "state_a" / "annotations" / RUN_KEEP_DIR / RUN_COCO_DIR / "ann.json"
    state_b_annotation = tmp_path / "runs" / "state_b" / "annotations" / RUN_KEEP_DIR / RUN_COCO_DIR / "ann.json"

    assert state_a_annotation.exists()
    assert state_b_annotation.exists()
    assert json.loads(state_a_annotation.read_text())["annotations"][0]["id"] == 101
    assert json.loads(state_b_annotation.read_text())["annotations"][0]["id"] == 201
    assert state_a["targets"][0]["key"] != state_b["targets"][0]["key"]
    assert "state_a" in state_a["targets"][0]["key"]
    assert "state_b" in state_b["targets"][0]["key"]


def test_reimporting_same_annotation_state_uses_existing_working_copy(tmp_path):
    store = UploadedTargetStore(tmp_path / "runs")

    store.import_bundle(
        image_file_name="a.png",
        image_data_url=_png_data_url("white"),
        annotation_file_name="ann.json",
        annotation_text=json.dumps(_coco(101)),
        image_set_id="same_images",
        annotation_state_id="review_state",
    )
    annotation_path = tmp_path / "runs" / "review_state" / "annotations" / RUN_KEEP_DIR / RUN_COCO_DIR / "ann.json"
    edited = json.loads(annotation_path.read_text())
    edited["annotations"] = []
    annotation_path.write_text(json.dumps(edited), encoding="utf-8")

    reimported = store.import_bundle(
        image_file_name="a.png",
        image_data_url=_png_data_url("white"),
        annotation_file_name="ann.json",
        annotation_text=json.dumps(_coco(101)),
        image_set_id="same_images",
        annotation_state_id="review_state",
    )

    assert reimported["restored_from_working_copy"] is True
    assert reimported["targets"] == []
    assert json.loads(annotation_path.read_text())["annotations"] == []


def test_original_direct_import_still_works_without_explicit_state(tmp_path):
    store = UploadedTargetStore(tmp_path / "runs")

    imported = store.import_bundle(
        image_file_name="a.png",
        image_data_url=_png_data_url("white"),
        annotation_file_name="ann.json",
        annotation_text=json.dumps(_coco(101)),
        import_session_id="legacy_folder_work_dataset",
    )

    assert len(imported["targets"]) == 1
    assert (tmp_path / "runs" / "legacy_folder_work_dataset" / "images" / RUN_KEEP_DIR / "a.png").exists()
    assert (tmp_path / "runs" / "legacy_folder_work_dataset" / "annotations" / RUN_KEEP_DIR / RUN_COCO_DIR / "ann.json").exists()


def _state_payload(image_name="a.png", annotation_id="101"):
    target = {
        "key": "old_key",
        "annotation_file_name": "ann.json",
        "annotation_json_path": "old/ann.json",
        "image_path": "old/a.png",
        "image_file_name": image_name,
        "annotation_id": annotation_id,
        "source_annotation_id": annotation_id,
        "result_index": 0,
        "category_name": "box",
        "category_id": 1,
        "image_width": 10,
        "image_height": 10,
        "bbox_xywh": [1, 1, 4, 4],
        "bbox_xyxy": [1, 1, 5, 5],
        "sort_index": 0,
        "import_id": "old_copy",
        "imported_at": "2026-01-01T00:00:00+00:00",
    }
    session = {
        "schema_version": 1,
        "session_id": "old_key",
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
        "target": target,
        "prompt_box_xyxy": [0, 0, 6, 6],
        "system_prompt_points": [],
        "working_points": [{"point_id": "p1", "x": 2, "y": 2, "label": 1, "created_at": "2026-01-01T00:00:00+00:00", "source": "manual"}],
        "line_strokes": [],
        "locked_regions": [],
        "text_prompt": "restore me",
        "history": [
            {
                "history_id": "init",
                "parent_history_id": None,
                "name": "init",
                "kind": "initial",
                "created_at": "2026-01-01T00:00:00+00:00",
                "score": 0.5,
                "mask_rle": {"size": [10, 10], "counts": [100]},
                "mask_area": 0,
                "mask_bbox_xywh": None,
                "prompt_box_xyxy": [0, 0, 6, 6],
                "manual_points_snapshot": [],
                "line_strokes_snapshot": [],
                "locked_regions_snapshot": [],
                "system_prompt_points": [],
                "text_prompt": "",
                "used_mask_prompt": False,
            }
        ],
        "current_history_id": "init",
        "variant_name": "manual_clicks",
        "notes": "",
        "is_deleted": False,
        "deleted_at": None,
    }
    return {"schema_version": 2, "format": "mask_iteration_state", "session": session}


def test_state_annotation_import_restores_full_session_payload(tmp_path):
    store = UploadedTargetStore(tmp_path / "runs")

    imported = store.import_bundle(
        image_file_name="a.png",
        image_data_url=_png_data_url("white"),
        annotation_file_name="ann.state.json",
        annotation_text=json.dumps(_state_payload()),
        image_set_id="copy_a",
        annotation_state_id="copy_a",
    )

    assert len(imported["targets"]) == 1
    assert len(imported["restored_sessions"]) == 1
    restored_session = imported["restored_sessions"][0]
    assert restored_session["text_prompt"] == "restore me"
    assert restored_session["target"]["import_id"] == "copy_a"
    assert restored_session["session_id"] == imported["targets"][0]["key"]
    assert (tmp_path / "runs" / "copy_a" / "annotations" / RUN_KEEP_DIR / RUN_COCO_DIR / "ann.json").exists()


def test_saved_state_import_reuses_original_coco_file_name(tmp_path):
    store = UploadedTargetStore(tmp_path / "runs")
    payload = _state_payload()
    payload["coco_file_name"] = "ann.json"

    imported = store.import_bundle(
        image_file_name="a.png",
        image_data_url=_png_data_url("white"),
        annotation_file_name="ann__101.json",
        annotation_text=json.dumps(payload),
        image_set_id="copy_a",
        annotation_state_id="copy_a",
    )

    assert imported["targets"][0]["annotation_file_name"] == "ann.json"
    assert (tmp_path / "runs" / "copy_a" / "annotations" / RUN_KEEP_DIR / RUN_COCO_DIR / "ann.json").exists()
    assert not (tmp_path / "runs" / "copy_a" / "annotations" / RUN_KEEP_DIR / RUN_COCO_DIR / "ann__101.json").exists()
