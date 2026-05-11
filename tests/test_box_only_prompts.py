import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mask_iteration_webapp.models import SessionState, TargetRecord
from mask_iteration_webapp.service import Sam3InferenceService


class FakeModel:
    def __init__(self):
        self.calls = []

    def predict_inst(self, state, **kwargs):
        self.calls.append(kwargs)
        masks = np.asarray([[[True, False], [False, False]]], dtype=bool)
        scores = np.asarray([0.75], dtype=np.float32)
        logits = np.asarray([[[32.0, -32.0], [-32.0, -32.0]]], dtype=np.float32)
        return masks, scores, logits


class FakeProcessor:
    def set_image(self, image):
        return {"image": image}


class FakeTorch:
    class cuda:
        @staticmethod
        def is_available():
            return False


def _target(tmp_path):
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"not-used")
    return TargetRecord(
        key="target_a",
        annotation_file_name="ann.json",
        annotation_json_path=str(tmp_path / "ann.json"),
        image_path=str(image_path),
        image_file_name="image.png",
        annotation_id="1",
        source_annotation_id="1",
        result_index=0,
        category_name="object",
        category_id=1,
        image_width=10,
        image_height=10,
        bbox_xywh=[2, 2, 4, 4],
        bbox_xyxy=[2, 2, 6, 6],
        sort_index=0,
        import_id="copy_a",
        imported_at="2026-01-01T00:00:00+00:00",
    )


def _service(model):
    service = Sam3InferenceService(
        project_root=Path("."),
        sam3_repo_dir=Path("."),
        local_deps_dir=Path("."),
        checkpoint=None,
        device="cpu",
        reference_box_expand_px=0,
    )
    service._ensure_model = lambda: (model, FakeProcessor(), "cpu")
    service._load_runtime = lambda: {"np": np, "torch": FakeTorch, "mask_utils": None}
    service._load_image = lambda image_path: object()
    return service


def test_initial_prediction_uses_box_without_center_point(tmp_path):
    model = FakeModel()
    result = _service(model).predict_initial(_target(tmp_path))

    assert result["system_prompt_points"] == []
    assert len(model.calls) == 1
    assert model.calls[0]["point_coords"] is None
    assert model.calls[0]["point_labels"] is None
    assert model.calls[0]["box"].tolist() == [[2.0, 2.0, 6.0, 6.0]]


def test_iteration_without_manual_points_uses_box_without_center_fallback(tmp_path):
    model = FakeModel()
    service = _service(model)

    service.iterate(
        target=_target(tmp_path),
        prompt_box_xyxy=[2.0, 2.0, 6.0, 6.0],
        working_points=[],
        previous_logits=np.zeros((1, 2, 2), dtype=np.float32),
    )

    assert len(model.calls) == 1
    assert model.calls[0]["point_coords"] is None
    assert model.calls[0]["point_labels"] is None
    assert model.calls[0]["box"].tolist() == [[2.0, 2.0, 6.0, 6.0]]


def test_loading_legacy_session_drops_system_center_points(tmp_path):
    payload = {
        "schema_version": 1,
        "session_id": "target_a",
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
        "target": _target(tmp_path).to_dict(),
        "prompt_box_xyxy": [2.0, 2.0, 6.0, 6.0],
        "system_prompt_points": [
            {
                "point_id": "system_center_prompt",
                "x": 4.0,
                "y": 4.0,
                "label": 1,
                "created_at": "2026-01-01T00:00:00+00:00",
                "source": "system",
            }
        ],
        "working_points": [],
        "line_strokes": [],
        "locked_regions": [],
        "text_prompt": "",
        "history": [
            {
                "history_id": "init",
                "parent_history_id": None,
                "name": "init",
                "kind": "initial",
                "created_at": "2026-01-01T00:00:00+00:00",
                "score": 0.75,
                "mask_rle": {"size": [2, 2], "counts": [0, 1, 3]},
                "mask_area": 1,
                "mask_bbox_xywh": [0, 0, 1, 1],
                "prompt_box_xyxy": [2.0, 2.0, 6.0, 6.0],
                "system_prompt_points": [
                    {
                        "point_id": "system_center_prompt",
                        "x": 4.0,
                        "y": 4.0,
                        "label": 1,
                        "created_at": "2026-01-01T00:00:00+00:00",
                        "source": "system",
                    }
                ],
            }
        ],
        "current_history_id": "init",
    }

    session = SessionState.from_dict(payload)

    assert session.system_prompt_points == []
    assert session.history[0].system_prompt_points == []
