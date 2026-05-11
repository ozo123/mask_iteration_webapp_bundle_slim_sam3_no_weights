from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def copy_points(points: list["PointRecord"]) -> list["PointRecord"]:
    return [PointRecord.from_dict(point.to_dict()) for point in points]


def drop_system_prompt_points(points: list["PointRecord"]) -> list["PointRecord"]:
    return [point for point in points if str(point.source or "").lower() != "system"]


def copy_line_strokes(strokes: list["LineStrokeRecord"]) -> list["LineStrokeRecord"]:
    return [LineStrokeRecord.from_dict(stroke.to_dict()) for stroke in strokes]


def copy_locked_regions(regions: list["LockedRegionRecord"]) -> list["LockedRegionRecord"]:
    return [LockedRegionRecord.from_dict(region.to_dict()) for region in regions]


@dataclass
class TargetRecord:
    key: str
    annotation_file_name: str
    annotation_json_path: str
    image_path: str
    image_file_name: str
    annotation_id: str
    source_annotation_id: str | None
    result_index: int
    category_name: str
    category_id: int | None
    image_width: int
    image_height: int
    bbox_xywh: list[float]
    bbox_xyxy: list[float]
    sort_index: int
    import_id: str | None = None
    imported_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TargetRecord":
        return cls(
            key=str(payload["key"]),
            annotation_file_name=str(payload["annotation_file_name"]),
            annotation_json_path=str(payload["annotation_json_path"]),
            image_path=str(payload["image_path"]),
            image_file_name=str(payload["image_file_name"]),
            annotation_id=str(payload["annotation_id"]),
            source_annotation_id=payload.get("source_annotation_id"),
            result_index=int(payload["result_index"]),
            category_name=str(payload["category_name"]),
            category_id=payload.get("category_id"),
            image_width=int(payload["image_width"]),
            image_height=int(payload["image_height"]),
            bbox_xywh=[float(value) for value in payload["bbox_xywh"]],
            bbox_xyxy=[float(value) for value in payload["bbox_xyxy"]],
            sort_index=int(payload["sort_index"]),
            import_id=payload.get("import_id"),
            imported_at=payload.get("imported_at"),
        )


@dataclass
class PointRecord:
    point_id: str
    x: float
    y: float
    label: int
    created_at: str
    source: str = "manual"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PointRecord":
        return cls(**payload)


@dataclass
class StrokePointRecord:
    x: float
    y: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StrokePointRecord":
        return cls(x=float(payload["x"]), y=float(payload["y"]))


@dataclass
class LineStrokeRecord:
    stroke_id: str
    label: int
    created_at: str
    source: str = "manual"
    points: list[StrokePointRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["points"] = [point.to_dict() for point in self.points]
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LineStrokeRecord":
        return cls(
            stroke_id=str(payload["stroke_id"]),
            label=1 if int(payload.get("label", 1)) == 1 else 0,
            created_at=str(payload.get("created_at") or utc_now_iso()),
            source=str(payload.get("source", "manual") or "manual"),
            points=[StrokePointRecord.from_dict(item) for item in payload.get("points", [])],
        )


@dataclass
class LockedRegionRecord:
    region_id: str
    created_at: str
    source: str = "manual"
    points: list[StrokePointRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["points"] = [point.to_dict() for point in self.points]
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LockedRegionRecord":
        return cls(
            region_id=str(payload["region_id"]),
            created_at=str(payload.get("created_at") or utc_now_iso()),
            source=str(payload.get("source", "manual") or "manual"),
            points=[StrokePointRecord.from_dict(item) for item in payload.get("points", [])],
        )


@dataclass
class HistoryRecord:
    history_id: str
    parent_history_id: str | None
    name: str
    kind: str
    created_at: str
    score: float | None
    mask_rle: dict[str, Any]
    mask_area: int
    mask_bbox_xywh: list[float] | None
    prompt_box_xyxy: list[float]
    manual_points_snapshot: list[PointRecord] = field(default_factory=list)
    line_strokes_snapshot: list[LineStrokeRecord] = field(default_factory=list)
    locked_regions_snapshot: list[LockedRegionRecord] = field(default_factory=list)
    system_prompt_points: list[PointRecord] = field(default_factory=list)
    text_prompt: str = ""
    used_mask_prompt: bool = False
    mask_logits_relpath: str | None = None
    mask_rle_relpath: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["manual_points_snapshot"] = [point.to_dict() for point in self.manual_points_snapshot]
        payload["line_strokes_snapshot"] = [stroke.to_dict() for stroke in self.line_strokes_snapshot]
        payload["locked_regions_snapshot"] = [region.to_dict() for region in self.locked_regions_snapshot]
        payload["system_prompt_points"] = [point.to_dict() for point in self.system_prompt_points]
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "HistoryRecord":
        return cls(
            history_id=str(payload["history_id"]),
            parent_history_id=payload.get("parent_history_id"),
            name=str(payload["name"]),
            kind=str(payload["kind"]),
            created_at=str(payload["created_at"]),
            score=payload.get("score"),
            mask_rle=payload.get("mask_rle") or {},
            mask_area=int(payload.get("mask_area", 0)),
            mask_bbox_xywh=payload.get("mask_bbox_xywh"),
            prompt_box_xyxy=[float(value) for value in payload["prompt_box_xyxy"]],
            manual_points_snapshot=[
                PointRecord.from_dict(item) for item in payload.get("manual_points_snapshot", [])
            ],
            line_strokes_snapshot=[
                LineStrokeRecord.from_dict(item) for item in payload.get("line_strokes_snapshot", [])
            ],
            locked_regions_snapshot=[
                LockedRegionRecord.from_dict(item) for item in payload.get("locked_regions_snapshot", [])
            ],
            system_prompt_points=drop_system_prompt_points(
                [PointRecord.from_dict(item) for item in payload.get("system_prompt_points", [])]
            ),
            text_prompt=str(payload.get("text_prompt", "") or ""),
            used_mask_prompt=bool(payload.get("used_mask_prompt", False)),
            mask_logits_relpath=payload.get("mask_logits_relpath"),
            mask_rle_relpath=payload.get("mask_rle_relpath"),
        )


@dataclass
class SessionState:
    schema_version: int
    session_id: str
    created_at: str
    updated_at: str
    target: TargetRecord
    prompt_box_xyxy: list[float]
    system_prompt_points: list[PointRecord]
    working_points: list[PointRecord]
    line_strokes: list[LineStrokeRecord]
    locked_regions: list[LockedRegionRecord]
    text_prompt: str
    history: list[HistoryRecord]
    current_history_id: str
    variant_name: str = "manual_clicks"
    notes: str = ""
    is_deleted: bool = False
    deleted_at: str | None = None

    def current_history(self) -> HistoryRecord:
        for item in self.history:
            if item.history_id == self.current_history_id:
                return item
        raise KeyError(f"Current history id not found: {self.current_history_id}")

    def history_by_id(self, history_id: str) -> HistoryRecord:
        for item in self.history:
            if item.history_id == history_id:
                return item
        raise KeyError(f"History id not found: {history_id}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "target": self.target.to_dict(),
            "prompt_box_xyxy": self.prompt_box_xyxy,
            "system_prompt_points": [point.to_dict() for point in self.system_prompt_points],
            "working_points": [point.to_dict() for point in self.working_points],
            "line_strokes": [stroke.to_dict() for stroke in self.line_strokes],
            "locked_regions": [region.to_dict() for region in self.locked_regions],
            "text_prompt": self.text_prompt,
            "history": [item.to_dict() for item in self.history],
            "current_history_id": self.current_history_id,
            "variant_name": self.variant_name,
            "notes": self.notes,
            "is_deleted": self.is_deleted,
            "deleted_at": self.deleted_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SessionState":
        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            session_id=str(payload["session_id"]),
            created_at=str(payload["created_at"]),
            updated_at=str(payload["updated_at"]),
            target=TargetRecord.from_dict(payload["target"]),
            prompt_box_xyxy=[float(value) for value in payload["prompt_box_xyxy"]],
            system_prompt_points=drop_system_prompt_points(
                [PointRecord.from_dict(item) for item in payload.get("system_prompt_points", [])]
            ),
            working_points=[PointRecord.from_dict(item) for item in payload.get("working_points", [])],
            line_strokes=[LineStrokeRecord.from_dict(item) for item in payload.get("line_strokes", [])],
            locked_regions=[LockedRegionRecord.from_dict(item) for item in payload.get("locked_regions", [])],
            text_prompt=str(payload.get("text_prompt", "") or ""),
            history=[HistoryRecord.from_dict(item) for item in payload.get("history", [])],
            current_history_id=str(payload["current_history_id"]),
            variant_name=str(payload.get("variant_name", "manual_clicks") or "manual_clicks"),
            notes=str(payload.get("notes", "")),
            is_deleted=bool(payload.get("is_deleted", False)),
            deleted_at=payload.get("deleted_at"),
        )
