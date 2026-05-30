import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mask_iteration_webapp.models import (
    HistoryRecord,
    LockedRegionRecord,
    PointRecord,
    SessionState,
    StrokePointRecord,
    TargetRecord,
)
from mask_iteration_webapp.service import MaskIterationService, Sam3InferenceService, SessionStore, UploadedTargetStore


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


class CategoryPromptProcessor:
    def __init__(self):
        self.text_prompts = []

    def set_image(self, image):
        return {"image": image}

    def set_text_prompt(self, prompt, state):
        self.text_prompts.append(prompt)
        state["text_prompt"] = prompt

    def add_geometric_prompt(self, box, label, state):
        return {
            "masks": np.asarray([[[False, True], [False, True]]], dtype=bool),
            "boxes": np.asarray([[2.0, 2.0, 6.0, 6.0]], dtype=np.float32),
            "scores": np.asarray([0.9], dtype=np.float32),
        }


class FakeTorch:
    class cuda:
        @staticmethod
        def is_available():
            return False


class RecordingIterationInference:
    reference_box_expand_px = 0

    def __init__(self):
        self.iteration_points = None

    def _load_runtime(self):
        return {"np": np, "mask_utils": None}

    def mask_from_rle(self, rle):
        return np.asarray(rle["mask"], dtype=bool)

    def _mask_to_rle(self, mask):
        return {"mask": np.asarray(mask).astype(bool).tolist()}

    def _mask_to_xywh(self, mask):
        ys, xs = np.where(np.asarray(mask).astype(bool))
        if len(xs) == 0:
            return None
        return [float(xs.min()), float(ys.min()), float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)]

    def iterate(self, target, prompt_box_xyxy, working_points, previous_logits):
        self.iteration_points = [point.to_dict() for point in working_points]
        mask = np.asarray([[True, False], [False, False]], dtype=bool)
        return {
            "mask_rle": self._mask_to_rle(mask),
            "mask_area": 1,
            "mask_bbox_xywh": [0.0, 0.0, 1.0, 1.0],
            "score": 0.9,
            "logits": np.asarray([[[32.0, -32.0], [-32.0, -32.0]]], dtype=np.float32),
        }


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


def _service(model, processor=None):
    service = Sam3InferenceService(
        project_root=Path("."),
        sam3_repo_dir=Path("."),
        local_deps_dir=Path("."),
        checkpoint=None,
        device="cpu",
        reference_box_expand_px=0,
    )
    service._ensure_model = lambda: (model, processor or FakeProcessor(), "cpu")
    service._load_runtime = lambda: {"np": np, "torch": FakeTorch, "mask_utils": None}
    service._load_image = lambda image_path: object()
    return service


def test_initial_prediction_uses_category_prompt_points_with_box(tmp_path):
    model = FakeModel()
    processor = CategoryPromptProcessor()
    target = _target(tmp_path)
    target.category_name = "apple"
    result = _service(model, processor=processor).predict_initial(target)

    assert processor.text_prompts == ["apple"]
    assert [point.source for point in result["system_prompt_points"]] == ["category"]
    assert len(model.calls) == 1
    assert model.calls[0]["point_coords"].shape == (1, 2)
    assert model.calls[0]["point_labels"].tolist() == [1]
    assert model.calls[0]["box"].tolist() == [[2.0, 2.0, 6.0, 6.0]]


def test_initial_prediction_falls_back_to_box_only_when_category_prompt_unavailable(tmp_path):
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
            },
            {
                "point_id": "initial_category_prompt_0",
                "x": 4.5,
                "y": 4.5,
                "label": 1,
                "created_at": "2026-01-01T00:00:00+00:00",
                "source": "category",
            }
        ],
        "working_points": [
            {
                "point_id": "system_center_prompt",
                "x": 4.0,
                "y": 4.0,
                "label": 1,
                "created_at": "2026-01-01T00:00:00+00:00",
                "source": "system",
            },
            {
                "point_id": "initial_category_prompt_0",
                "x": 4.5,
                "y": 4.5,
                "label": 1,
                "created_at": "2026-01-01T00:00:00+00:00",
                "source": "category",
            },
            {
                "point_id": "manual_keep",
                "x": 5.0,
                "y": 5.0,
                "label": 1,
                "created_at": "2026-01-01T00:00:00+00:00",
                "source": "manual",
            },
        ],
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
                "manual_points_snapshot": [
                    {
                        "point_id": "system_center_prompt",
                        "x": 4.0,
                        "y": 4.0,
                        "label": 1,
                        "created_at": "2026-01-01T00:00:00+00:00",
                        "source": "system",
                    },
                    {
                        "point_id": "initial_category_prompt_0",
                        "x": 4.5,
                        "y": 4.5,
                        "label": 1,
                        "created_at": "2026-01-01T00:00:00+00:00",
                        "source": "category",
                    },
                    {
                        "point_id": "manual_keep",
                        "x": 5.0,
                        "y": 5.0,
                        "label": 1,
                        "created_at": "2026-01-01T00:00:00+00:00",
                        "source": "manual",
                    },
                ],
                "system_prompt_points": [
                    {
                        "point_id": "system_center_prompt",
                        "x": 4.0,
                        "y": 4.0,
                        "label": 1,
                        "created_at": "2026-01-01T00:00:00+00:00",
                        "source": "system",
                    },
                    {
                        "point_id": "initial_category_prompt_0",
                        "x": 4.5,
                        "y": 4.5,
                        "label": 1,
                        "created_at": "2026-01-01T00:00:00+00:00",
                        "source": "category",
                    }
                ],
            }
        ],
        "current_history_id": "init",
    }

    session = SessionState.from_dict(payload)

    assert [point.point_id for point in session.system_prompt_points] == ["initial_category_prompt_0"]
    assert [point.point_id for point in session.working_points] == ["manual_keep"]
    assert [point.point_id for point in session.history[0].manual_points_snapshot] == ["manual_keep"]
    assert [point.point_id for point in session.history[0].system_prompt_points] == ["initial_category_prompt_0"]


def _locked_region(region_id, label, points):
    return LockedRegionRecord(
        region_id=region_id,
        label=label,
        created_at="2026-01-01T00:00:00+00:00",
        points=[StrokePointRecord(x=x, y=y) for x, y in points],
    )


def test_legacy_locked_region_defaults_to_foreground_label():
    region = LockedRegionRecord.from_dict(
        {
            "region_id": "legacy_region",
            "created_at": "2026-01-01T00:00:00+00:00",
            "points": [{"x": 1, "y": 1}, {"x": 3, "y": 1}, {"x": 3, "y": 3}],
        }
    )

    assert region.label == 1


def test_locked_regions_apply_background_over_foreground(tmp_path):
    target = _target(tmp_path)
    target_store = UploadedTargetStore(tmp_path / "runs")
    session_store = SessionStore(tmp_path / "sessions")
    inference = RecordingIterationInference()
    service = MaskIterationService(target_store, session_store, inference)
    session = SessionState(
        schema_version=1,
        session_id=target.key,
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
        target=target,
        prompt_box_xyxy=[2.0, 2.0, 6.0, 6.0],
        system_prompt_points=[],
        working_points=[],
        line_strokes=[],
        locked_regions=[
            _locked_region("fg", 1, [(1, 1), (5, 1), (5, 5), (1, 5)]),
            _locked_region("bg", 0, [(3, 3), (7, 3), (7, 7), (3, 7)]),
        ],
        text_prompt="",
        history=[],
        current_history_id="init",
    )

    mask, logits = service._apply_locked_regions_to_mask_and_logits(
        session,
        np.zeros((10, 10), dtype=bool),
        np.zeros((1, 10, 10), dtype=np.float32),
    )

    assert bool(mask[2, 2]) is True
    assert bool(mask[4, 4]) is False
    assert float(logits[0, 2, 2]) == 32.0
    assert float(logits[0, 4, 4]) == -32.0


def test_locked_regions_background_wins_even_when_added_first(tmp_path):
    target = _target(tmp_path)
    target_store = UploadedTargetStore(tmp_path / "runs")
    session_store = SessionStore(tmp_path / "sessions")
    inference = RecordingIterationInference()
    service = MaskIterationService(target_store, session_store, inference)
    session = SessionState(
        schema_version=1,
        session_id=target.key,
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
        target=target,
        prompt_box_xyxy=[2.0, 2.0, 6.0, 6.0],
        system_prompt_points=[],
        working_points=[],
        line_strokes=[],
        locked_regions=[
            _locked_region("bg", 0, [(3, 3), (7, 3), (7, 7), (3, 7)]),
            _locked_region("fg", 1, [(1, 1), (5, 1), (5, 5), (1, 5)]),
        ],
        text_prompt="",
        history=[],
        current_history_id="init",
    )

    mask, logits = service._apply_locked_regions_to_mask_and_logits(
        session,
        np.zeros((10, 10), dtype=bool),
        np.zeros((1, 10, 10), dtype=np.float32),
    )

    assert bool(mask[2, 2]) is True
    assert bool(mask[4, 4]) is False
    assert float(logits[0, 2, 2]) == 32.0
    assert float(logits[0, 4, 4]) == -32.0


def test_update_locked_region_replaces_points_and_clamps_to_image_bounds(tmp_path):
    target = _target(tmp_path)
    target_store = UploadedTargetStore(tmp_path / "runs")
    session_store = SessionStore(tmp_path / "sessions")
    inference = RecordingIterationInference()
    service = MaskIterationService(target_store, session_store, inference)
    initial_mask = np.zeros((10, 10), dtype=bool)
    session = SessionState(
        schema_version=1,
        session_id=target.key,
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
        target=target,
        prompt_box_xyxy=[2.0, 2.0, 6.0, 6.0],
        system_prompt_points=[],
        working_points=[],
        line_strokes=[],
        locked_regions=[_locked_region("fg", 1, [(1, 1), (5, 1), (5, 5), (1, 5)])],
        text_prompt="",
        history=[
            HistoryRecord(
                history_id="init",
                parent_history_id=None,
                name="init",
                kind="initial",
                created_at="2026-01-01T00:00:00+00:00",
                score=0.75,
                mask_rle={"mask": initial_mask.tolist()},
                mask_area=0,
                mask_bbox_xywh=None,
                prompt_box_xyxy=[2.0, 2.0, 6.0, 6.0],
            )
        ],
        current_history_id="init",
    )
    session_store.save_session(session)

    payload = service.update_locked_region(
        target.key,
        "fg",
        [{"x": -5, "y": -7}, {"x": 9, "y": 0}, {"x": 9, "y": 9}, {"x": 0, "y": 9}],
    )

    updated = session_store.load_session(target.key)
    assert updated is not None
    assert payload["session"]["current_history_id"].startswith("region_edit_")
    assert updated.current_history().kind == "region_edit"
    assert updated.locked_regions[0].region_id == "fg"
    assert [(point.x, point.y) for point in updated.locked_regions[0].points] == [
        (0.0, 0.0),
        (9.0, 0.0),
        (9.0, 9.0),
        (0.0, 9.0),
    ]


def test_iteration_strips_generated_points_before_inference(tmp_path):
    target = _target(tmp_path)
    target_store = UploadedTargetStore(tmp_path / "runs")
    session_store = SessionStore(tmp_path / "sessions")
    inference = RecordingIterationInference()
    service = MaskIterationService(target_store, session_store, inference)
    session = SessionState(
        schema_version=1,
        session_id=target.key,
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
        target=target,
        prompt_box_xyxy=[2.0, 2.0, 6.0, 6.0],
        system_prompt_points=[],
        working_points=[
            PointRecord(
                point_id="system_center_prompt",
                x=4.0,
                y=4.0,
                label=1,
                created_at="2026-01-01T00:00:00+00:00",
                source="system",
            ),
            PointRecord(
                point_id="initial_category_prompt_0",
                x=4.5,
                y=4.5,
                label=1,
                created_at="2026-01-01T00:00:00+00:00",
                source="category",
            ),
            PointRecord(
                point_id="manual_keep",
                x=5.0,
                y=5.0,
                label=1,
                created_at="2026-01-01T00:00:00+00:00",
                source="manual",
            ),
        ],
        line_strokes=[],
        locked_regions=[],
        text_prompt="",
        history=[
            HistoryRecord(
                history_id="init",
                parent_history_id=None,
                name="init",
                kind="initial",
                created_at="2026-01-01T00:00:00+00:00",
                score=0.75,
                mask_rle={"mask": [[True, False], [False, False]]},
                mask_area=1,
                mask_bbox_xywh=[0.0, 0.0, 1.0, 1.0],
                prompt_box_xyxy=[2.0, 2.0, 6.0, 6.0],
            )
        ],
        current_history_id="init",
    )
    session_store.save_session(session)

    service.iterate(target.key)

    assert [point["point_id"] for point in inference.iteration_points] == ["manual_keep"]
    updated = session_store.load_session(target.key)
    assert updated is not None
    assert [point.point_id for point in updated.working_points] == ["manual_keep"]
    assert [point.point_id for point in updated.current_history().manual_points_snapshot] == ["manual_keep"]


def test_iteration_is_blocked_while_locked_regions_exist(tmp_path):
    target = _target(tmp_path)
    target_store = UploadedTargetStore(tmp_path / "runs")
    session_store = SessionStore(tmp_path / "sessions")
    inference = RecordingIterationInference()
    service = MaskIterationService(target_store, session_store, inference)
    session = SessionState(
        schema_version=1,
        session_id=target.key,
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
        target=target,
        prompt_box_xyxy=[2.0, 2.0, 6.0, 6.0],
        system_prompt_points=[],
        working_points=[],
        line_strokes=[],
        locked_regions=[_locked_region("fg", 1, [(1, 1), (5, 1), (5, 5), (1, 5)])],
        text_prompt="",
        history=[
            HistoryRecord(
                history_id="init",
                parent_history_id=None,
                name="init",
                kind="initial",
                created_at="2026-01-01T00:00:00+00:00",
                score=0.75,
                mask_rle={"mask": [[True, False], [False, False]]},
                mask_area=1,
                mask_bbox_xywh=[0.0, 0.0, 1.0, 1.0],
                prompt_box_xyxy=[2.0, 2.0, 6.0, 6.0],
            )
        ],
        current_history_id="init",
    )
    session_store.save_session(session)

    with pytest.raises(ValueError, match="closed locked regions"):
        service.iterate(target.key)

    assert inference.iteration_points is None
