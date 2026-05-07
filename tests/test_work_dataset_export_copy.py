import base64
import json
import sys
from io import BytesIO
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mask_iteration_webapp.service import UploadedTargetStore, SessionStore, MaskIterationService


class DummyInference:
    reference_box_expand_px = 10
    def readiness(self):
        return {"dummy": True}


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


def test_export_work_dataset_copy_contains_current_images_annotations_and_pair_report(tmp_path):
    target_store = UploadedTargetStore(tmp_path / "work_dataset")
    session_store = SessionStore(tmp_path / "sessions")
    service = MaskIterationService(target_store, session_store, DummyInference())

    target_store.import_bundle(
        image_file_name="a.png",
        image_data_url=_png_data_url(),
        annotation_file_name="ann.json",
        annotation_text=json.dumps(_coco(101)),
        image_set_id="same_images",
        annotation_state_id="state_a",
    )

    export = service.export_work_dataset_copy(image_set_id="same_images", annotation_state_id="state_a", export_name="manual_name")
    export_root = Path(export["export_root"])

    assert export_root.name == "manual_name"
    assert (export_root / "images" / "a.png").exists()
    assert (export_root / "annotations" / "ann.json").exists()
    manifest = json.loads((export_root / "manifest.json").read_text())
    assert manifest["image_set_id"] == "same_images"
    assert manifest["annotation_state_id"] == "state_a"
    assert manifest["pairing"]["matched_count"] == 1
    assert manifest["pairing"]["missing_annotation_files"] == []
    assert manifest["pairing"]["missing_image_files"] == []
