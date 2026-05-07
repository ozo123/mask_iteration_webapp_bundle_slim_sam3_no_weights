import base64
import json
import sys
from io import BytesIO
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mask_iteration_webapp.service import UploadedTargetStore


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
    store = UploadedTargetStore(tmp_path / "work_dataset")

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

    state_a_annotation = tmp_path / "work_dataset" / "annotations" / "state_a" / "ann.json"
    state_b_annotation = tmp_path / "work_dataset" / "annotations" / "state_b" / "ann.json"

    assert state_a_annotation.exists()
    assert state_b_annotation.exists()
    assert json.loads(state_a_annotation.read_text())["annotations"][0]["id"] == 101
    assert json.loads(state_b_annotation.read_text())["annotations"][0]["id"] == 201
    assert state_a["targets"][0]["key"] != state_b["targets"][0]["key"]
    assert "state_a" in state_a["targets"][0]["key"]
    assert "state_b" in state_b["targets"][0]["key"]


def test_reimporting_same_annotation_state_uses_existing_working_copy(tmp_path):
    store = UploadedTargetStore(tmp_path / "work_dataset")

    store.import_bundle(
        image_file_name="a.png",
        image_data_url=_png_data_url("white"),
        annotation_file_name="ann.json",
        annotation_text=json.dumps(_coco(101)),
        image_set_id="same_images",
        annotation_state_id="review_state",
    )
    annotation_path = tmp_path / "work_dataset" / "annotations" / "review_state" / "ann.json"
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
    store = UploadedTargetStore(tmp_path / "work_dataset")

    imported = store.import_bundle(
        image_file_name="a.png",
        image_data_url=_png_data_url("white"),
        annotation_file_name="ann.json",
        annotation_text=json.dumps(_coco(101)),
        import_session_id="legacy_folder_work_dataset",
    )

    assert len(imported["targets"]) == 1
    assert (tmp_path / "work_dataset" / "images" / "legacy_folder_work_dataset" / "a.png").exists()
    assert (tmp_path / "work_dataset" / "annotations" / "legacy_folder_work_dataset" / "ann.json").exists()
