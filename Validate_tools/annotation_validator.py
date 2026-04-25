#!/usr/bin/env python3
"""
数据集标注验证工具包
支持调用LLM API对目标检测标注进行质量验证

功能：
1. 单张/批量图片验证
2. 异步高并发验证
3. 支持多种API格式（OpenAI兼容格式）
4. 结果导出为JSON
"""

import os
import json
import base64
import asyncio
import aiohttp
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from openai import OpenAI
import re


@dataclass
class Annotation:
    """单个标注框数据类"""
    x: float  # 中心点 x (百分比)
    y: float  # 中心点 y (百分比)
    width: float  # 宽度 (百分比)
    height: float  # 高度 (百分比)
    label: str  # 类别标签
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "center": {"x": round(self.x, 2), "y": round(self.y, 2)},
            "size": {"width": round(self.width, 2), "height": round(self.height, 2)},
            "bbox": {
                "x1": round(self.x - self.width/2, 2),
                "y1": round(self.y - self.height/2, 2),
                "x2": round(self.x + self.width/2, 2),
                "y2": round(self.y + self.height/2, 2)
            }
        }


@dataclass
class ValidationResult:
    """单张图片验证结果数据类"""
    image_name: str
    total_annotations: int
    correct_annotations: List[Dict]
    incorrect_annotations: List[Dict]
    missing_annotations: List[Dict]
    comments: str
    score: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_name": self.image_name,
            "total_annotations": self.total_annotations,
            "correct_annotations": self.correct_annotations,
            "incorrect_annotations": self.incorrect_annotations,
            "missing_annotations": self.missing_annotations,
            "comments": self.comments,
            "score": self.score
        }


class AnnotationValidator:
    """
    标注验证器
    
    支持同步和异步两种验证模式：
    - 同步模式：适合少量图片验证，使用OpenAI客户端
    - 异步模式：适合批量验证，使用aiohttp实现高并发
    """
    
    def __init__(
        self, 
        api_key: str, 
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        model: str = "qwen-vl-max"
    ):
        """
        初始化验证器
        
        Args:
            api_key: API密钥
            base_url: API基础URL
            model: 模型名称
        """
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        
        # 初始化同步客户端
        self._sync_client = None
    
    @property
    def sync_client(self) -> OpenAI:
        """获取同步客户端（懒加载）"""
        if self._sync_client is None:
            self._sync_client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
        return self._sync_client
    
    @staticmethod
    def load_rules(rules_path: str) -> Dict[str, Any]:
        """加载类别规则定义"""
        with open(rules_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @staticmethod
    def load_annotation_document(anno_path: str) -> Dict[str, Any]:
        """加载标注文件原始JSON"""
        with open(anno_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def parse_annotation_payload(data: Dict[str, Any]) -> List[Annotation]:
        """
        加载标注文件
        
        支持格式：
        {
            "results": [
                {
                    "value": {
                        "x": 25.0,
                        "y": 66.5,
                        "width": 8.9,
                        "height": 14.0,
                        "rectanglelabels": ["chair"]
                    }
                }
            ]
        }
        """
        annotations = []
        for result in data.get('results', []):
            value = result.get('value', {})
            labels = value.get('rectanglelabels', [])
            if labels:
                anno = Annotation(
                    x=value.get('x', 0),
                    y=value.get('y', 0),
                    width=value.get('width', 0),
                    height=value.get('height', 0),
                    label=labels[0]
                )
                annotations.append(anno)
        
        return annotations

    @classmethod
    def load_annotation(cls, anno_path: str) -> List[Annotation]:
        """兼容旧接口：直接返回标注框列表"""
        return cls.parse_annotation_payload(cls.load_annotation_document(anno_path))
    
    @staticmethod
    def encode_image(image_path: str) -> str:
        """将图片编码为base64字符串"""
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')

    @staticmethod
    def _normalize_label_token(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip().strip('"').strip("'")
        if not text:
            return None
        return re.sub(r"\s+", " ", text).lower()

    @classmethod
    def _build_allowed_label_lookup(cls, rules: Dict[str, Any]) -> Dict[str, str]:
        lookup: Dict[str, str] = {}
        for label in rules.keys():
            token = cls._normalize_label_token(label)
            if token:
                lookup[token] = label
        return lookup

    @classmethod
    def _canonicalize_allowed_label(
        cls,
        value: Any,
        lookup: Dict[str, str],
    ) -> Optional[str]:
        token = cls._normalize_label_token(value)
        if token in {None, "null", "none", "n/a", "na", "unknown", "uncertain", "unsure", "not sure"}:
            return None
        return lookup.get(token)

    @classmethod
    def _sanitize_validation_response(
        cls,
        validation_data: Dict[str, Any],
        rules: Dict[str, Any],
    ) -> Dict[str, Any]:
        lookup = cls._build_allowed_label_lookup(rules)
        warnings: List[str] = []

        def sanitize_field(entry: Dict[str, Any], field_name: str) -> None:
            if not isinstance(entry, dict) or field_name not in entry:
                return
            raw_value = entry.get(field_name)
            canonical = cls._canonicalize_allowed_label(raw_value, lookup)
            if canonical is not None:
                entry[field_name] = canonical
                return
            token = cls._normalize_label_token(raw_value)
            if token is None:
                entry[field_name] = None
                return
            entry[f"{field_name}_raw"] = raw_value
            entry[field_name] = None
            warnings.append(f"{field_name}={raw_value}")

        def sanitize_text(text: Any) -> Any:
            if not isinstance(text, str) or not text.strip():
                return text

            def replace_quoted_label(match: re.Match[str]) -> str:
                candidate = match.group(1).strip()
                canonical = cls._canonicalize_allowed_label(candidate, lookup)
                if canonical is not None:
                    return canonical
                token = cls._normalize_label_token(candidate)
                if token and re.fullmatch(r"[A-Za-z0-9 _\\-/]{2,40}", candidate):
                    warnings.append(f"text_label={candidate}")
                    return "different allowed class"
                return candidate

            sanitized = re.sub(r"'([^'\n]{2,40})'", replace_quoted_label, text)
            sanitized = re.sub(r'"([^"\n]{2,40})"', replace_quoted_label, sanitized)
            return sanitized

        def sanitize_text_fields(entry: Dict[str, Any], field_names: List[str]) -> None:
            if not isinstance(entry, dict):
                return
            for field_name in field_names:
                if field_name in entry:
                    entry[field_name] = sanitize_text(entry.get(field_name))

        for item in validation_data.get('errors', []) or []:
            sanitize_field(item, 'label')
            sanitize_field(item, 'suggested_label')
            sanitize_text_fields(item, ['issue', 'suggestion', 'description'])
        for item in validation_data.get('missing', []) or []:
            sanitize_field(item, 'suggested_label')
            sanitize_text_fields(item, ['description', 'location'])
        for item in validation_data.get('correct_annotations', []) or []:
            sanitize_field(item, 'label')
            sanitize_field(item, 'suggested_label')
            sanitize_text_fields(item, ['issue', 'suggestion', 'description'])
        for item in validation_data.get('incorrect_annotations', []) or []:
            sanitize_field(item, 'label')
            sanitize_field(item, 'suggested_label')
            sanitize_text_fields(item, ['issue', 'suggestion', 'description'])
        for item in validation_data.get('missing_annotations', []) or []:
            sanitize_field(item, 'suggested_label')
            sanitize_text_fields(item, ['description', 'location'])

        for field_name in ['assessment', 'overall_assessment', 'notes', 'comments']:
            if field_name in validation_data:
                validation_data[field_name] = sanitize_text(validation_data.get(field_name))

        if warnings:
            note_text = "Taxonomy filter removed unsupported labels: " + ", ".join(warnings[:8])
            existing_notes = validation_data.get('notes')
            if isinstance(existing_notes, str) and existing_notes.strip():
                validation_data['notes'] = existing_notes.strip() + " | " + note_text
            else:
                validation_data['notes'] = note_text

        return validation_data

    def _legacy_build_prompt_unused(
        self, 
        annotations: List[Annotation], 
        rules: Dict[str, Any],
        strict_mode: bool = False,
        review_scope: str = ""
    ) -> str:
        """
        构建验证提示词
        
        Args:
            annotations: 标注列表
            rules: 类别规则
            strict_mode: 是否使用严格模式（更多报错）
        """
        # 构建类别规则文本
        rules_text = []
        for category, info in rules.items():
            definition = info.get('definition', '')
            distinction = info.get('distinction_rules', '')
            rules_text.append(
                f"- {category}: {definition}" + 
                (f" (区分规则: {distinction})" if distinction else "")
            )
        
        # 构建当前标注文本
        anno_text = []
        for i, anno in enumerate(annotations, 1):
            anno_text.append(
                f"{i}. {anno.label}: 中心点({anno.x:.1f}%, {anno.y:.1f}%), "
                f"大小({anno.width:.1f}%, {anno.height:.1f}%)"
            )
        
        allowed_categories = list(rules.keys())
        allowed_categories_text = chr(10).join(f"- {label}" for label in allowed_categories)
        taxonomy_guardrails = f"""
## Allowed Output Category Names ({len(allowed_categories)} exact strings):
{allowed_categories_text}

## Taxonomy Constraints:
- Every category name in the JSON output MUST be one of the {len(allowed_categories)} exact category names above.
- Never invent, translate, paraphrase, broaden, or substitute labels outside this list.
- Forbidden examples unless they exactly appear above: bag, purse, fruit, appliance, furniture, decoration.
- If no exact allowed replacement label is justified with high confidence, set suggested_label to null.
- Use suggested_label for alternative categories. Keep suggestion focused on fix instructions and do not introduce extra category names outside the allowed list.
"""

        single_box_mode = len(annotations) == 1
        if single_box_mode:
            anno = annotations[0]
            box_text = (
                f"1. {anno.label}: 中心点({anno.x:.2f}%, {anno.y:.2f}%), "
                f"大小({anno.width:.2f}%, {anno.height:.2f}%)"
            )
            normalized_review_scope = str(review_scope or "").strip().lower()
            if normalized_review_scope == "single_annotation_original_image_box":
                image_context = (
                    "当前图片是整张原图。\n"
                    "你只需要审核下面这一个当前展示的预测框。\n"
                    "除非其他物体能直接帮助你判断这个框是否正确，否则忽略图片中的其他目标和其他标注。"
                )
            elif normalized_review_scope == "single_annotation_crop_box2x":
                image_context = (
                    "当前图片不是整张原图，而是围绕当前目标框裁出的局部图。\n"
                    "这个局部图的范围大约是原框宽高各扩展到 2 倍后的区域。\n"
                    "下面给出的框坐标已经是相对于这张局部图重新计算后的坐标。\n"
                    "你只需要审核下面这一个当前展示的预测框。\n"
                    "除非其他物体能直接帮助你判断这个框是否正确，否则忽略局部图中的其他目标。"
                )
            else:
                image_context = (
                    "当前图片只包含当前单框审核所需的上下文。\n"
                    "你只需要审核下面这一个当前展示的预测框。"
                )

            single_box_output_format = """{
    "correct_annotations": [
        {
            "index": 1,
            "label": "必须是当前标注的56类之一",
            "is_correct": true,
            "issues": "如果有问题，说明原因；如果正确，简要说明判断依据",
            "suggested_label": null
        }
    ],
    "missing_annotations": [],
    "overall_assessment": "整体评估：只针对当前这个框，说明类别和框位置是否合理",
    "score": 85
}"""

            if strict_mode:
                return f"""你是一个专业的目标检测数据标注审核专家。请仔细审核这张图片中当前展示的单个标注框质量。

{image_context}

## 待检测的目标类别（共{len(rules)}类）：
{chr(10).join(rules_text)}

{taxonomy_guardrails}

## 当前人工标注的锚框（当前只审核1个）：
{box_text}

## 请完成以下审核任务：

### 1. 标注正确性检查
对当前这一个标注框，判断：
- 类别标签是否正确？锚框内的物体是否确实属于当前标注类别？
- 锚框位置是否准确？是否完整包含目标物体？是否有明显过多背景？
- 锚框尺寸是否合理？

### 2. 单框审核范围说明
- 本次请求只审核当前这一个框，不审核整张图全部标注。
- 不要检查整张图的漏标问题。
- missing_annotations 固定返回空列表。
- 如果你认为当前标签错误，suggested_label 必须只能从上面给出的 {len(allowed_categories)} 个类别中选择，或者返回 null。
- score 必须填写 0 到 100 之间的整数，表示当前这个框的审核质量分，不要照抄示例值。

### 3. 输出格式
请以 JSON 格式返回审核结果：
{single_box_output_format}

请只返回 JSON 格式结果，不要有其他文字说明。"""

            return f"""你是一个专业的目标检测数据标注审核专家。请仔细审核这张图片中当前展示的单个标注框质量。

{image_context}

## 待检测的目标类别（共{len(rules)}类）：
{chr(10).join(rules_text)}

{taxonomy_guardrails}

## 当前人工标注的锚框（当前只审核1个）：
{box_text}

## 请完成以下审核任务：

### 1. 标注正确性检查
对当前这一个标注框，判断：
- 类别标签是否正确？锚框内的物体是否确实属于当前标注类别？
- 锚框位置是否准确？是否完整包含目标物体？是否有明显过多背景？
- 锚框尺寸是否合理？

### 2. 宽松审核原则
- 当你拿不准时，优先认为这个框是正确的。
- 轻微的框边界偏差可以接受。
- 只有在你比较确定当前框有明显问题时，才把 is_correct 设为 false。
- 本次请求只审核当前这一个框，不审核整张图全部标注。
- 不要检查整张图的漏标问题。
- missing_annotations 固定返回空列表。
- 如果你认为当前标签错误，suggested_label 必须只能从上面给出的 {len(allowed_categories)} 个类别中选择，或者返回 null。
- score 必须填写 0 到 100 之间的整数，表示当前这个框的审核质量分，不要照抄示例值。

### 3. 输出格式
请以 JSON 格式返回审核结果：
{single_box_output_format}

请只返回 JSON 格式结果，不要有其他文字说明。"""

        if strict_mode:
            prompt = f"""You are a professional object detection annotation reviewer. Please review the annotation quality of this indoor scene image.

## Target Categories ({len(rules)} classes):
{chr(10).join(rules_text)}

{taxonomy_guardrails}

## Current Annotations ({len(annotations)} boxes):
{chr(10).join(anno_text) if anno_text else "No annotations"}

## Review Tasks:

### 1. Correctness Check
Check each annotation:
- Is the category correct?
- Does the bbox accurately contain the object?
- Is the bbox size reasonable?

### 2. Missing Check
Check for unannotated objects that should be labeled.

### 3. Output Format
Return results in this JSON format:
{{
    "summary": {{
        "total_annotations": {len(annotations)},
        "correct_count": 0,
        "error_count": 0,
        "missing_count": 0
    }},
    "errors": [
        {{
            "index": 1,
            "label": "original_category_from_the_56_allowed_classes",
            "issue": "description of the problem",
            "suggested_label": "one_of_the_56_allowed_classes_or_null",
            "suggestion": "how to fix"
        }}
    ],
    "missing": [
        {{
            "location": "where in the image",
            "description": "object description",
            "suggested_label": "one_of_the_56_allowed_classes_or_null"
        }}
    ],
    "assessment": "brief overall assessment",
    "score": 85
}}

Return ONLY the JSON, no other text."""
        else:
            # 宽松模式 - 减少过度纠错
            prompt = f"""You are a senior data annotation reviewer for an indoor UAV dataset. Your task is to review bounding box annotations with HIGH PRECISION - avoid false positives.

## CRITICAL REVIEW PRINCIPLES:
1. **WHEN IN DOUBT, CONSIDER IT CORRECT** - Only report errors if you are >90% confident
2. **Bounding box tolerance**: Small deviations (<20% size difference, <10% position shift) are ACCEPTABLE
3. **Office furniture ambiguity**: Chairs, dividers, partitions, and cabinets can look similar - only flag if CLEARLY wrong
4. **Partial occlusion is normal** - Don't flag boxes that reasonably cover partially hidden objects
5. **Scale variation is acceptable** - Objects at different distances naturally have different sizes

## Target Categories ({len(rules)} classes):
{chr(10).join(rules_text[:25])}
... (and {len(rules) - 25} more categories)

{taxonomy_guardrails}

## Current Annotations ({len(annotations)} boxes):
{chr(10).join(anno_text) if anno_text else "No annotations"}

## ERROR CLASSIFICATION (Only report HIGH confidence errors):

### Category Error (high confidence only)
- The object is CLEARLY a different category (e.g., a bottle labeled as "person")
- Office furniture exceptions: chair/couch/divider mislabeling requires clear visual evidence

### Position Error (severe only)
- Box is completely on background/wrong object
- Box misses >50% of the target object
- Box is 3x+ larger than the object (includes excessive background)

### Missing Annotation (obvious omissions only)
- Clearly visible, unambiguous objects NOT annotated
- Size threshold: ignore tiny objects (<2% of image area)
- Ambiguous regions: when uncertain, DO NOT report as missing

## CONFIDENCE LEVELS:
- **high**: You are >90% sure this is an error (obvious mislabeling, severe misplacement)
- **medium**: You are 70-90% sure (minor issues, debatable cases)
- **low**: You are <70% sure (uncertain, possibly correct)

**IMPORTANT**: Only include "high" confidence errors in your report. Skip medium/low confidence issues.

## Output Format (JSON only):
{{
    "summary": {{
        "total_annotations": {len(annotations)},
        "correct_count": 0,
        "error_count": 0,
        "missing_count": 0
    }},
    "errors": [
        {{
            "index": 1,
            "label": "original_category_from_the_56_allowed_classes",
            "issue": "Clear description of the error",
            "confidence": "high",
            "suggested_label": "one_of_the_56_allowed_classes_or_null",
            "suggestion": "How to fix"
        }}
    ],
    "missing": [
        {{
            "location": "Where in image",
            "description": "Object description",
            "suggested_label": "one_of_the_56_allowed_classes_or_null",
            "confidence": "high"
        }}
    ],
    "assessment": "Brief assessment focusing on major issues only",
    "score": 85,
    "notes": "Any ambiguous cases you chose not to flag"
}}

Remember: We want to catch REAL errors, not minor imperfections. When uncertain, assume the annotator is correct. Return ONLY the JSON."""
        
        return prompt
    
    @staticmethod
    def _is_single_box_review_scope(review_scope: str) -> bool:
        return str(review_scope or "").strip().lower().startswith("single_annotation")

    @staticmethod
    def _coerce_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        return str(value or "").strip().lower() in {"true", "yes", "y", "1"}

    @staticmethod
    def _coerce_score(value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            score = int(round(float(value)))
        except Exception:
            return None
        return max(0, min(100, score))

    def _build_single_box_result(
        self,
        image_name: str,
        annotations: List[Annotation],
        validation_data: Dict[str, Any],
        rules: Dict[str, Any],
    ) -> ValidationResult:
        lookup = self._build_allowed_label_lookup(rules)
        review_entry = validation_data.get("review") if isinstance(validation_data.get("review"), dict) else validation_data
        base_label = annotations[0].label if annotations else ""
        label_correct = self._coerce_bool(review_entry.get("label_correct", True))
        bbox_reasonable = self._coerce_bool(review_entry.get("bbox_reasonable", True))
        suggested_label = self._canonicalize_allowed_label(review_entry.get("suggested_label"), lookup)
        reason = str(review_entry.get("reason") or validation_data.get("reason") or "").strip()
        score = self._coerce_score(review_entry.get("score", validation_data.get("score")))
        detail = {
            "index": 1,
            "label": base_label,
            "is_correct": bool(label_correct and bbox_reasonable),
            "label_correct": label_correct,
            "bbox_reasonable": bbox_reasonable,
            "suggested_label": suggested_label,
            "issues": reason,
        }
        return ValidationResult(
            image_name=image_name,
            total_annotations=len(annotations),
            correct_annotations=[detail] if detail["is_correct"] else [],
            incorrect_annotations=[] if detail["is_correct"] else [detail],
            missing_annotations=[],
            comments=reason,
            score=score,
        )

    def build_prompt(
        self,
        annotations: List[Annotation],
        rules: Dict[str, Any],
        strict_mode: bool = False,
        review_scope: str = ""
    ) -> str:
        allowed_categories = list(rules.keys())
        allowed_categories_text = chr(10).join(f"- {label}" for label in allowed_categories)

        if len(annotations) == 1:
            anno = annotations[0]
            normalized_review_scope = str(review_scope or "").strip().lower()
            if normalized_review_scope == "single_annotation_original_image_box":
                image_context = "Image type: full original image."
            elif normalized_review_scope == "single_annotation_crop_box2x":
                image_context = "Image type: 2x local crop around the target box. Coordinates below are on this crop."
            else:
                image_context = "Image type: single target review image."
            mode_hint = (
                "Be conservative. If uncertain, set both booleans to true."
                if not strict_mode else
                "Be strict. Set false when you see a clear issue."
            )
            return f"""Review one detection box only.

{image_context}

Target:
label={anno.label}
center=({anno.x:.2f}%, {anno.y:.2f}%)
size=({anno.width:.2f}%, {anno.height:.2f}%)

Allowed labels:
{allowed_categories_text}

Rules:
- Judge only this one box.
- Decide only whether the label is correct and whether the box position/size is reasonable.
- Ignore all other objects. Do not report missing objects.
- suggested_label must be one allowed label or null.
- reason must be short.
- score must be an integer from 0 to 100.
- {mode_hint}

Return JSON only:
{{
  "review": {{
    "label": "{anno.label}",
    "label_correct": true,
    "bbox_reasonable": true,
    "suggested_label": null,
    "score": 90,
    "reason": "short reason"
  }}
}}"""

        rules_text = []
        for category, info in rules.items():
            definition = info.get('definition', '')
            distinction = info.get('distinction_rules', '')
            rules_text.append(
                f"- {category}: {definition}" +
                (f" ({distinction})" if distinction else "")
            )
        anno_text = []
        for i, anno in enumerate(annotations, 1):
            anno_text.append(
                f"{i}. {anno.label}: center=({anno.x:.1f}%, {anno.y:.1f}%), "
                f"size=({anno.width:.1f}%, {anno.height:.1f}%)"
            )
        prompt = f"""You are reviewing object detection annotations.

Allowed labels:
{allowed_categories_text}

Annotations:
{chr(10).join(anno_text) if anno_text else "No annotations"}

Return JSON only:
{{
  "summary": {{
    "total_annotations": {len(annotations)},
    "correct_count": 0,
    "error_count": 0,
    "missing_count": 0
  }},
  "errors": [],
  "missing": [],
  "assessment": "brief assessment",
  "score": 85
}}"""
        return prompt

    @staticmethod
    def parse_validation_response(content: str) -> Dict[str, Any]:
        """
        解析API返回的验证结果
        
        支持多种格式：
        - 纯JSON
        - Markdown代码块包裹的JSON
        - 包含额外文本的JSON
        """
        try:
            # 尝试直接解析
            return json.loads(content)
        except json.JSONDecodeError:
            # 尝试从markdown代码块中提取
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', content)
            if json_match:
                return json.loads(json_match.group(1))
            else:
                # 尝试寻找JSON对象
                json_match = re.search(r'\{[\s\S]*\}', content)
                if json_match:
                    return json.loads(json_match.group(0))
                else:
                    raise ValueError(f"无法解析API返回的JSON: {content[:500]}")
    
    def validate_single_sync(
        self, 
        image_path: str, 
        anno_path: str, 
        rules: Dict[str, Any],
        strict_mode: bool = False
    ) -> ValidationResult:
        """
        同步验证单张图片
        
        Args:
            image_path: 图片路径
            anno_path: 标注文件路径
            rules: 类别规则
            strict_mode: 是否使用严格模式
            
        Returns:
            ValidationResult对象
        """
        image_name = Path(image_path).name
        
        # 加载标注
        annotation_document = self.load_annotation_document(anno_path)
        annotations = self.parse_annotation_payload(annotation_document)
        review_scope = str(annotation_document.get("review_scope", "") or "")
        
        # 编码图片
        base64_image = self.encode_image(image_path)
        prompt = self.build_prompt(annotations, rules, strict_mode, review_scope=review_scope)
        single_box_mode = self._is_single_box_review_scope(review_scope)
        
        # 构建提示词
        prompt = self.build_prompt(annotations, rules, strict_mode, review_scope=review_scope)
        
        try:
            # 调用API
            response = self.sync_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=160 if single_box_mode else 4000,
                temperature=0.0 if single_box_mode else 0.2
            )
            
            content = response.choices[0].message.content
            validation_data = self._sanitize_validation_response(
                self.parse_validation_response(content),
                rules,
            )
            if single_box_mode and (
                isinstance(validation_data.get('review'), dict)
                or 'label_correct' in validation_data
                or 'bbox_reasonable' in validation_data
            ):
                return self._build_single_box_result(
                    image_name=image_name,
                    annotations=annotations,
                    validation_data=validation_data,
                    rules=rules,
                )
            
            # 适配两种输出格式
            if 'summary' in validation_data:
                errors = validation_data.get('errors', [])
                missing = validation_data.get('missing', [])
                if strict_mode:
                    selected_errors = errors
                    selected_missing = missing
                else:
                    selected_errors = [
                        e for e in errors 
                        if e.get('confidence', '').lower() == 'high'
                    ]
                    selected_missing = [
                        m for m in missing 
                        if m.get('confidence', '').lower() == 'high'
                    ]
                return ValidationResult(
                    image_name=image_name,
                    total_annotations=len(annotations),
                    correct_annotations=[],
                    incorrect_annotations=selected_errors,
                    missing_annotations=selected_missing,
                    comments=validation_data.get('assessment', ''),
                    score=validation_data.get('score')
                )
            else:
                # 严格模式格式
                return ValidationResult(
                    image_name=image_name,
                    total_annotations=len(annotations),
                    correct_annotations=validation_data.get('correct_annotations', []),
                    incorrect_annotations=[
                        a for a in validation_data.get('correct_annotations', [])
                        if not a.get('is_correct', True)
                    ],
                    missing_annotations=validation_data.get('missing_annotations', []),
                    comments=validation_data.get('overall_assessment', ''),
                    score=validation_data.get('score')
                )
                
        except Exception as e:
            print(f"验证 {image_name} 时出错: {str(e)}")
            return ValidationResult(
                image_name=image_name,
                total_annotations=len(annotations),
                correct_annotations=[],
                incorrect_annotations=[],
                missing_annotations=[],
                comments=f"验证失败: {str(e)}"
            )
    
    async def validate_single_async(
        self, 
        image_path: str, 
        anno_path: str, 
        rules: Dict[str, Any],
        strict_mode: bool = False
    ) -> ValidationResult:
        """
        异步验证单张图片
        
        使用aiohttp实现真正的异步HTTP请求，适合高并发批量验证
        """
        image_name = Path(image_path).name
        
        # 加载标注
        annotation_document = self.load_annotation_document(anno_path)
        annotations = self.parse_annotation_payload(annotation_document)
        review_scope = str(annotation_document.get("review_scope", "") or "")
        
        # 编码图片
        base64_image = self.encode_image(image_path)
        
        # 构建提示词
        prompt = self.build_prompt(annotations, rules, strict_mode, review_scope=review_scope)
        
        # 构建请求
        prompt = self.build_prompt(annotations, rules, strict_mode, review_scope=review_scope)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 160 if self._is_single_box_review_scope(review_scope) else 4000,
            "temperature": 0.0 if self._is_single_box_review_scope(review_scope) else 0.2
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"API错误: {response.status}, {error_text}")
                    
                    result = await response.json()
                    content = result['choices'][0]['message']['content']
                    validation_data = self._sanitize_validation_response(
                        self.parse_validation_response(content),
                        rules,
                    )
                    if self._is_single_box_review_scope(review_scope) and (
                        isinstance(validation_data.get('review'), dict)
                        or 'label_correct' in validation_data
                        or 'bbox_reasonable' in validation_data
                    ):
                        return self._build_single_box_result(
                            image_name=image_name,
                            annotations=annotations,
                            validation_data=validation_data,
                            rules=rules,
                        )
                    
                    # 统一输出格式
                    if 'summary' in validation_data:
                        errors = validation_data.get('errors', [])
                        missing = validation_data.get('missing', [])
                        if strict_mode:
                            selected_errors = errors
                            selected_missing = missing
                        else:
                            selected_errors = [
                                e for e in errors 
                                if e.get('confidence', '').lower() == 'high'
                            ]
                            selected_missing = [
                                m for m in missing 
                                if m.get('confidence', '').lower() == 'high'
                            ]
                        return ValidationResult(
                            image_name=image_name,
                            total_annotations=len(annotations),
                            correct_annotations=[],
                            incorrect_annotations=selected_errors,
                            missing_annotations=selected_missing,
                            comments=validation_data.get('assessment', ''),
                            score=validation_data.get('score')
                        )
                    else:
                        return ValidationResult(
                            image_name=image_name,
                            total_annotations=len(annotations),
                            correct_annotations=validation_data.get('correct_annotations', []),
                            incorrect_annotations=[
                                a for a in validation_data.get('correct_annotations', [])
                                if not a.get('is_correct', True)
                            ],
                            missing_annotations=validation_data.get('missing_annotations', []),
                            comments=validation_data.get('overall_assessment', ''),
                            score=validation_data.get('score')
                        )
                    
        except Exception as e:
            print(f"验证 {image_name} 时出错: {str(e)}")
            return ValidationResult(
                image_name=image_name,
                total_annotations=len(annotations),
                correct_annotations=[],
                incorrect_annotations=[],
                missing_annotations=[],
                comments=f"验证失败: {str(e)}"
            )
    
    async def validate_batch_async(
        self,
        image_dir: str,
        anno_dir: str,
        rules: Dict[str, Any],
        max_concurrent: int = 5,
        limit: Optional[int] = None,
        strict_mode: bool = False,
        progress_callback: Optional[callable] = None
    ) -> List[ValidationResult]:
        """
        批量异步验证
        
        Args:
            image_dir: 图片目录
            anno_dir: 标注目录
            rules: 类别规则
            max_concurrent: 最大并发数
            limit: 限制验证图片数量
            strict_mode: 是否使用严格模式
            progress_callback: 进度回调函数，接收(当前索引, 总数, 图片名)
            
        Returns:
            ValidationResult列表
        """
        image_dir = Path(image_dir)
        anno_dir = Path(anno_dir)
        
        # 获取所有图片文件
        image_files = sorted([
            f for f in image_dir.iterdir() 
            if f.suffix.lower() in ['.jpg', '.jpeg', '.png']
        ])
        
        if limit:
            image_files = image_files[:limit]
        
        total = len(image_files)
        print(f"准备验证 {total} 张图片...")
        
        # 创建信号量限制并发
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def validate_with_limit(idx: int, img_path: Path) -> ValidationResult:
            async with semaphore:
                anno_path = anno_dir / f"{img_path.stem}.json"
                if not anno_path.exists():
                    return ValidationResult(
                        image_name=img_path.name,
                        total_annotations=0,
                        correct_annotations=[],
                        incorrect_annotations=[],
                        missing_annotations=[],
                        comments="未找到标注文件"
                    )
                
                result = await self.validate_single_async(
                    str(img_path), str(anno_path), rules, strict_mode
                )
                
                if progress_callback:
                    progress_callback(idx + 1, total, img_path.name)
                else:
                    print(f"[OK] [{idx+1}/{total}] {img_path.name}")
                
                return result
        
        # 并发执行所有验证任务
        tasks = [
            validate_with_limit(i, img) 
            for i, img in enumerate(image_files)
        ]
        results = await asyncio.gather(*tasks)
        
        return list(results)
    
    def validate_batch(
        self,
        image_dir: str,
        anno_dir: str,
        rules_path: str,
        output_path: str = "validation_results.json",
        max_concurrent: int = 5,
        limit: Optional[int] = None,
        strict_mode: bool = False
    ) -> Dict[str, Any]:
        """
        批量验证并保存结果（同步入口）
        
        Args:
            image_dir: 图片目录
            anno_dir: 标注目录
            rules_path: 规则文件路径
            output_path: 输出JSON文件路径
            max_concurrent: 最大并发数
            limit: 限制验证图片数量
            strict_mode: 是否使用严格模式
            
        Returns:
            汇总结果字典
        """
        # 加载规则
        rules = self.load_rules(rules_path)
        print(f"已加载 {len(rules)} 个类别定义")
        
        # 执行验证
        results = asyncio.run(self.validate_batch_async(
            image_dir, anno_dir, rules, max_concurrent, limit, strict_mode
        ))
        
        # 构建输出数据
        output_data = {
            "summary": {
                "total_images": len(results),
                "total_annotations": sum(r.total_annotations for r in results),
                "images_with_errors": sum(
                    1 for r in results 
                    if r.incorrect_annotations or r.missing_annotations
                ),
                "images_with_errors_only": sum(
                    1 for r in results 
                    if r.incorrect_annotations
                ),
                "images_with_missing_only": sum(
                    1 for r in results 
                    if r.missing_annotations and not r.incorrect_annotations
                )
            },
            "results": [r.to_dict() for r in results]
        }
        
        # 保存结果
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n验证完成！结果已保存到: {output_path}")
        print(f"总图片数: {output_data['summary']['total_images']}")
        print(f"总标注数: {output_data['summary']['total_annotations']}")
        print(f"有问题图片数: {output_data['summary']['images_with_errors']}")
        
        return output_data


# 便捷函数接口
def validate_single(
    image_path: str,
    anno_path: str,
    rules_path: str,
    api_key: Optional[str] = None,
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    model: str = "qwen-vl-max",
    strict_mode: bool = False
) -> ValidationResult:
    """
    便捷函数：验证单张图片
    
    不需要创建验证器实例，直接调用即可
    """
    api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("请提供API Key或设置DASHSCOPE_API_KEY环境变量")
    
    validator = AnnotationValidator(api_key, base_url, model)
    rules = validator.load_rules(rules_path)
    return validator.validate_single_sync(image_path, anno_path, rules, strict_mode)


def validate_batch(
    image_dir: str,
    anno_dir: str,
    rules_path: str,
    output_path: str = "validation_results.json",
    api_key: Optional[str] = None,
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    model: str = "qwen-vl-max",
    max_concurrent: int = 5,
    limit: Optional[int] = None,
    strict_mode: bool = False
) -> Dict[str, Any]:
    """
    便捷函数：批量验证
    
    不需要创建验证器实例，直接调用即可
    """
    api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("请提供API Key或设置DASHSCOPE_API_KEY环境变量")
    
    validator = AnnotationValidator(api_key, base_url, model)
    return validator.validate_batch(
        image_dir, anno_dir, rules_path, output_path,
        max_concurrent, limit, strict_mode
    )
