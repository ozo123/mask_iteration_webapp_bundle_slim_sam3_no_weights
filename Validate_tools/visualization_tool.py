#!/usr/bin/env python3
"""
标注可视化工具包
支持将标注框绘制在图片上，并可标记验证结果中的错误

功能：
1. 单张/批量图片可视化
2. 错误标注高亮显示（红色）
3. 漏标区域提示
4. 支持中文标签（使用PIL）
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from PIL import Image, ImageDraw, ImageFont


# 为不同类别预定义颜色（BGR格式用于OpenCV）
COLORS = {
    'handbag': (255, 0, 0), 'backpack': (0, 255, 0), 'bottle': (0, 0, 255),
    'wine glass': (255, 255, 0), 'cup': (255, 0, 255), 'bowl': (0, 255, 255),
    'fork': (128, 0, 0), 'spoon': (0, 128, 0), 'knife': (0, 0, 128),
    'banana': (128, 128, 0), 'apple': (128, 0, 128), 'orange': (0, 128, 128),
    'dining table': (255, 128, 0), 'chair': (255, 0, 128), 'potted plant': (128, 255, 0),
    'couch': (0, 255, 128), 'bed': (255, 128, 128), 'refrigerator': (128, 255, 255),
    'microwave': (255, 255, 128), 'sink': (128, 128, 255), 'cell phone': (192, 0, 0),
    'remote': (0, 192, 0), 'laptop': (0, 0, 192), 'keyboard': (192, 192, 0),
    'mouse': (192, 0, 192), 'tv': (0, 192, 192), 'teddy bear': (255, 192, 0),
    'scissors': (255, 0, 192), 'vase': (192, 255, 0), 'book': (0, 255, 192),
    'clock': (192, 128, 0), 'cake': (192, 0, 128), 'pizza': (128, 192, 0),
    'sandwich': (0, 192, 128), 'carrot': (255, 128, 192), 'broccoli': (128, 255, 192),
    'hot dog': (192, 255, 128), 'donut': (255, 192, 128), 'hair drier': (128, 128, 192),
    'toothbrush': (192, 128, 255), 'umbrella': (128, 192, 255), 'toaster': (255, 128, 255),
    'oven': (255, 255, 192), 'person': (0, 0, 0),
    'cookies': (255, 64, 0), 'pillow': (64, 255, 0), 
    'framed picture(pcitures)': (0, 64, 255), 'picture': (0, 64, 255),
    'trash can': (255, 64, 255), 'pens': (64, 255, 255), 
    'Water dispenser': (255, 255, 64),
    'coffee maker': (128, 64, 0), 'pot': (128, 0, 64), 'switch': (0, 128, 64),
    'Power outlet': (64, 128, 0), 'cabinet': (128, 128, 64), 'cloth': (64, 128, 128),
}

DEFAULT_COLOR = (128, 128, 128)


class AnnotationVisualizer:
    """标注可视化器"""
    
    def __init__(self, use_pil_for_text: bool = True):
        """
        初始化可视化器
        
        Args:
            use_pil_for_text: 是否使用PIL绘制中文标签（效果更好）
        """
        self.use_pil_for_text = use_pil_for_text
        
    @staticmethod
    def get_color(label: str) -> Tuple[int, int, int]:
        """获取类别对应的颜色"""
        return COLORS.get(label.lower(), DEFAULT_COLOR)
    
    @staticmethod
    def load_annotation(anno_path: str) -> List[Dict[str, Any]]:
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
        with open(anno_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        annotations = []
        for result in data.get('results', []):
            value = result.get('value', {})
            labels = value.get('rectanglelabels', [])
            if labels:
                annotations.append({
                    'x': value.get('x', 0),
                    'y': value.get('y', 0),
                    'width': value.get('width', 0),
                    'height': value.get('height', 0),
                    'label': labels[0]
                })
        
        return annotations
    
    @staticmethod
    def load_validation_result(result_path: str) -> Dict[str, Any]:
        """加载验证结果文件"""
        with open(result_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _get_font(self, size: int = 20) -> Optional[ImageFont.FreeTypeFont]:
        """获取字体对象"""
        try:
            # 尝试使用系统字体
            font_paths = [
                "C:/Windows/Fonts/simhei.ttf",  # Windows 黑体
                "C:/Windows/Fonts/simsun.ttc",  # Windows 宋体
                "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",  # Linux
                "/System/Library/Fonts/PingFang.ttc",  # macOS
            ]
            for path in font_paths:
                if os.path.exists(path):
                    return ImageFont.truetype(path, size)
        except:
            pass
        return None
    
    def draw_annotations(
        self,
        image_path: str,
        annotations: List[Dict[str, Any]],
        output_path: Optional[str] = None,
        show_labels: bool = True,
        highlight_errors: Optional[List[Dict]] = None,
        line_thickness: int = 2
    ) -> np.ndarray:
        """
        在图片上绘制标注框
        
        Args:
            image_path: 原图路径
            annotations: 标注列表
            output_path: 输出路径（可选）
            show_labels: 是否显示标签
            highlight_errors: 需要高亮显示的错误标注列表
            line_thickness: 框线粗细
            
        Returns:
            绘制后的图像数组（BGR格式）
        """
        # 读取图片
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图片: {image_path}")
        
        height, width = img.shape[:2]
        
        # 错误标注索引集合
        error_indices = set()
        error_reasons = {}
        if highlight_errors:
            for err in highlight_errors:
                if 'index' in err:
                    idx = err['index']
                    error_indices.add(idx)
                    error_reasons[idx] = err.get('issue', err.get('issues', '错误'))
        
        # 绘制每个标注框
        for i, anno in enumerate(annotations, 1):
            # 百分比转换为像素坐标
            cx = int(anno['x'] / 100.0 * width)
            cy = int(anno['y'] / 100.0 * height)
            w = int(anno['width'] / 100.0 * width)
            h = int(anno['height'] / 100.0 * height)
            
            # 计算左上角和右下角坐标
            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)
            x2 = int(cx + w / 2)
            y2 = int(cy + h / 2)
            
            # 确保坐标在图片范围内
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width - 1, x2), min(height - 1, y2)
            
            label = anno['label']
            color = self.get_color(label)
            
            # 如果是错误标注，使用红色高亮
            is_error = i in error_indices
            if is_error:
                color = (0, 0, 255)  # BGR红色
                thickness = line_thickness + 1
            else:
                thickness = line_thickness
            
            # 绘制矩形框
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            
            if show_labels:
                # 准备标签文本
                if is_error:
                    label_text = f"{i}:{label} [错误]"
                else:
                    label_text = f"{i}:{label}"
                
                if self.use_pil_for_text:
                    img = self._draw_text_pil(img, label_text, x1, y1, color)
                else:
                    img = self._draw_text_cv2(img, label_text, x1, y1, color)
        
        # 保存图片
        if output_path:
            cv2.imwrite(output_path, img)
            print(f"已保存可视化结果: {output_path}")
        
        return img
    
    def _draw_text_cv2(
        self, 
        img: np.ndarray, 
        text: str, 
        x: int, 
        y: int, 
        color: Tuple[int, int, int]
    ) -> np.ndarray:
        """使用OpenCV绘制文本（不支持中文）"""
        # 计算文本大小
        (text_w, text_h), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
        )
        
        # 绘制标签背景
        label_y = y - 5 if y - 5 > text_h else y + text_h + 5
        cv2.rectangle(
            img, 
            (x, label_y - text_h - 4), 
            (x + text_w + 4, label_y + 4), 
            color, 
            -1
        )
        
        # 绘制标签文字
        cv2.putText(
            img, 
            text, 
            (x + 2, label_y), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (255, 255, 255), 
            2
        )
        
        return img
    
    def _draw_text_pil(
        self, 
        img: np.ndarray, 
        text: str, 
        x: int, 
        y: int, 
        color: Tuple[int, int, int],
        font_size: int = 20
    ) -> np.ndarray:
        """使用PIL绘制文本（支持中文）"""
        # 转换颜色从BGR到RGB
        rgb_color = (color[2], color[1], color[0])
        
        # 转换为PIL图像
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        
        # 获取字体
        font = self._get_font(font_size)
        if font is None:
            # 回退到CV2方式
            return self._draw_text_cv2(img, text, x, y, color)
        
        # 计算文本大小
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        
        # 确定标签位置
        label_y = y - text_h - 5 if y - text_h - 5 > text_h else y + text_h + 5
        
        # 绘制背景
        draw.rectangle(
            [(x, label_y - text_h - 4), (x + text_w + 8, label_y + 4)],
            fill=rgb_color
        )
        
        # 绘制文字
        draw.text((x + 4, label_y - text_h), text, fill=(255, 255, 255), font=font)
        
        # 转换回OpenCV格式
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    def visualize_validation_result(
        self,
        image_path: str,
        anno_path: str,
        validation_result: Union[Dict[str, Any], str],
        output_path: Optional[str] = None,
        show_missing: bool = True
    ) -> np.ndarray:
        """
        可视化验证结果，标记错误标注和漏标区域
        
        Args:
            image_path: 原图路径
            anno_path: 标注文件路径
            validation_result: 验证结果字典或JSON文件路径
            output_path: 输出路径
            show_missing: 是否显示漏标提示
            
        Returns:
            绘制后的图像数组
        """
        # 加载标注
        annotations = self.load_annotation(anno_path)
        
        # 加载验证结果
        if isinstance(validation_result, str):
            validation_result = self.load_validation_result(validation_result)
        
        # 读取图片
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图片: {image_path}")
        
        height, width = img.shape[:2]
        
        # 获取错误和漏标信息
        incorrect_annotations = validation_result.get('incorrect_annotations', [])
        missing_annotations = validation_result.get('missing_annotations', [])
        
        # 如果没有incorrect_annotations，尝试从errors字段解析
        if not incorrect_annotations:
            errors = validation_result.get('errors', [])
            incorrect_annotations = [
                {
                    'index': e.get('index'),
                    'label': e.get('label'),
                    'issues': e.get('issue', e.get('issues', '')),
                    'confidence': e.get('confidence', 'high')
                }
                for e in errors
            ]
        
        # 绘制所有标注框
        for i, anno in enumerate(annotations, 1):
            cx = int(anno['x'] / 100.0 * width)
            cy = int(anno['y'] / 100.0 * height)
            w = int(anno['width'] / 100.0 * width)
            h = int(anno['height'] / 100.0 * height)
            
            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)
            x2 = int(cx + w / 2)
            y2 = int(cy + h / 2)
            
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width - 1, x2), min(height - 1, y2)
            
            label = anno['label']
            
            # 检查是否是错误标注
            is_error = any(
                err.get('index') == i 
                for err in incorrect_annotations
            )
            
            if is_error:
                color = (0, 0, 255)  # 红色表示错误
                thickness = 3
                # 找到错误原因
                error_info = next(
                    (err for err in incorrect_annotations if err.get('index') == i),
                    {}
                )
                error_reason = error_info.get('issues', error_info.get('issue', '未知错误'))
                label_text = f"{i}:{label} [错误: {error_reason[:15]}]"
            else:
                color = self.get_color(label)
                thickness = 2
                label_text = f"{i}:{label}"
            
            # 绘制矩形框
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            
            # 绘制标签
            if self.use_pil_for_text:
                img = self._draw_text_pil(img, label_text, x1, y1, color, font_size=18)
            else:
                img = self._draw_text_cv2(img, label_text, x1, y1, color)
        
        # 显示漏标提示
        if show_missing and missing_annotations:
            for j, missing in enumerate(missing_annotations[:5], 1):  # 最多显示5个
                description = missing.get('description', '')
                location = missing.get('location', '')
                suggested_label = missing.get('suggested_label', 'unknown')
                
                # 在右侧添加漏标提示
                y_pos = 30 + (j - 1) * 30
                if y_pos < height - 30:
                    text = f"漏标{j}: {suggested_label}"
                    if self.use_pil_for_text:
                        img = self._draw_text_pil(
                            img, text, width - 200, y_pos, (0, 165, 255), font_size=18
                        )
                    else:
                        cv2.putText(
                            img, text, (width - 200, y_pos), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2
                        )
        
        # 添加整体评估信息
        comments = validation_result.get('comments', validation_result.get('assessment', ''))
        score = validation_result.get('score', None)
        
        # 在底部添加评估信息
        comment_text = f"评估: {comments[:60]}" if comments else ""
        if score is not None:
            comment_text += f" | 评分: {score}"
        
        if comment_text:
            # 绘制黑色背景条
            cv2.rectangle(img, (0, height - 50), (width, height), (0, 0, 0), -1)
            if self.use_pil_for_text:
                img = self._draw_text_pil(
                    img, comment_text, 10, height - 40, (128, 128, 128), font_size=20
                )
            else:
                cv2.putText(
                    img, comment_text, (10, height - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                )
        
        # 保存图片
        if output_path:
            cv2.imwrite(output_path, img)
            print(f"已保存可视化结果: {output_path}")
        
        return img
    
    def batch_visualize(
        self,
        image_dir: str,
        anno_dir: str,
        output_dir: str,
        validation_results_path: Optional[str] = None,
        only_with_errors: bool = False
    ) -> List[str]:
        """
        批量可视化标注
        
        Args:
            image_dir: 图片目录
            anno_dir: 标注目录
            output_dir: 输出目录
            validation_results_path: 验证结果文件路径（可选）
            only_with_errors: 只包含有问题的图片
            
        Returns:
            生成的文件路径列表
        """
        image_dir = Path(image_dir)
        anno_dir = Path(anno_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载验证结果（如果有）
        validation_results = {}
        if validation_results_path and os.path.exists(validation_results_path):
            with open(validation_results_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 支持两种格式：直接是结果列表或包含results字段
                if 'results' in data:
                    for result in data.get('results', []):
                        validation_results[result['image_name']] = result
                else:
                    # 可能是直接以图片名为键的字典
                    validation_results = data
            print(f"已加载 {len(validation_results)} 条验证结果")
        
        # 获取所有图片
        image_files = sorted([
            f for f in image_dir.iterdir() 
            if f.suffix.lower() in ['.jpg', '.jpeg', '.png']
        ])
        
        print(f"开始可视化 {len(image_files)} 张图片...")
        
        output_files = []
        for img_path in image_files:
            anno_path = anno_dir / f"{img_path.stem}.json"
            output_path = output_dir / f"{img_path.stem}_vis.jpg"
            
            if not anno_path.exists():
                print(f"跳过: 未找到标注文件 {anno_path}")
                continue
            
            try:
                # 如果有验证结果，使用验证结果可视化
                if img_path.name in validation_results:
                    result = validation_results[img_path.name]
                    
                    # 如果设置了只显示有问题的图片
                    if only_with_errors:
                        has_errors = (
                            result.get('incorrect_annotations', []) or 
                            result.get('errors', [])
                        )
                        if not has_errors:
                            continue
                    
                    self.visualize_validation_result(
                        str(img_path),
                        str(anno_path),
                        result,
                        str(output_path)
                    )
                else:
                    # 普通可视化
                    annotations = self.load_annotation(str(anno_path))
                    self.draw_annotations(str(img_path), annotations, str(output_path))
                
                output_files.append(str(output_path))
                print(f"[OK] {img_path.name}")
                
            except Exception as e:
                print(f"[ERR] {img_path.name}: {str(e)}")
        
        print(f"\n可视化完成！结果保存在: {output_dir}")
        print(f"共生成 {len(output_files)} 张可视化图片")
        
        return output_files


# 便捷函数接口
def visualize_single(
    image_path: str,
    anno_path: str,
    output_path: Optional[str] = None,
    validation_result: Optional[Union[Dict, str]] = None,
    use_pil: bool = True
) -> np.ndarray:
    """
    便捷函数：可视化单张图片
    
    Args:
        image_path: 图片路径
        anno_path: 标注文件路径
        output_path: 输出路径
        validation_result: 验证结果（可选）
        use_pil: 是否使用PIL绘制中文
        
    Returns:
        绘制后的图像数组
    """
    visualizer = AnnotationVisualizer(use_pil_for_text=use_pil)
    
    if validation_result:
        return visualizer.visualize_validation_result(
            image_path, anno_path, validation_result, output_path
        )
    else:
        annotations = visualizer.load_annotation(anno_path)
        return visualizer.draw_annotations(image_path, annotations, output_path)


def visualize_batch(
    image_dir: str,
    anno_dir: str,
    output_dir: str,
    validation_results_path: Optional[str] = None,
    only_with_errors: bool = False,
    use_pil: bool = True
) -> List[str]:
    """
    便捷函数：批量可视化
    
    Args:
        image_dir: 图片目录
        anno_dir: 标注目录
        output_dir: 输出目录
        validation_results_path: 验证结果文件路径（可选）
        only_with_errors: 只包含有问题的图片
        use_pil: 是否使用PIL绘制中文
        
    Returns:
        生成的文件路径列表
    """
    visualizer = AnnotationVisualizer(use_pil_for_text=use_pil)
    return visualizer.batch_visualize(
        image_dir, anno_dir, output_dir, 
        validation_results_path, only_with_errors
    )
