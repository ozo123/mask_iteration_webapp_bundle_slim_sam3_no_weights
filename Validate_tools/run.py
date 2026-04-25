#!/usr/bin/env python3
"""
数据集标注验证工具 - 主入口

功能：
1. 验证模式：调用LLM API验证标注质量
2. 可视化模式：将标注框绘制在图片上
3. 完整流程：验证 + 可视化

用法：
    python run.py validate     # 验证模式
    python run.py visualize    # 可视化模式
    python run.py full         # 完整流程（验证+可视化）
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional

from annotation_validator import AnnotationValidator, validate_single, validate_batch
from visualization_tool import AnnotationVisualizer, visualize_single, visualize_batch


# 默认配置路径
DEFAULT_IMAGE_DIR = r"E:\UAV\Test\Benchmark\20251101-1111"
DEFAULT_ANNO_DIR = r"E:\UAV\Test\Benchmark\per_image_annotations"
DEFAULT_RULES_PATH = r"E:\UAV\Test\Benchmark\rules.json"
DEFAULT_OUTPUT_DIR = "results"
DEFAULT_VALIDATION_OUTPUT = "validation_results.json"


def get_api_key(args_key: Optional[str] = None) -> str:
    """获取API Key"""
    api_key = args_key or os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        api_key = input("请输入 DashScope API Key: ").strip()
    if not api_key:
        print("错误: 未提供API Key")
        sys.exit(1)
    return api_key


def cmd_validate(args):
    """验证命令"""
    print("=" * 60)
    print("数据集标注验证")
    print("=" * 60)
    
    api_key = get_api_key(args.api_key)
    
    # 初始化验证器
    validator = AnnotationValidator(
        api_key=api_key,
        base_url=args.base_url,
        model=args.model
    )
    
    # 确定路径
    image_dir = args.image_dir or DEFAULT_IMAGE_DIR
    anno_dir = args.anno_dir or DEFAULT_ANNO_DIR
    rules_path = args.rules or DEFAULT_RULES_PATH
    output_path = args.output or DEFAULT_VALIDATION_OUTPUT
    
    # 检查路径
    if not os.path.exists(rules_path):
        print(f"错误: 规则文件不存在: {rules_path}")
        sys.exit(1)
    
    if args.single:
        # 单张验证
        image_name = args.single
        image_path = os.path.join(image_dir, image_name)
        anno_path = os.path.join(anno_dir, f"{Path(image_name).stem}.json")
        
        if not os.path.exists(image_path):
            print(f"错误: 图片不存在: {image_path}")
            sys.exit(1)
        if not os.path.exists(anno_path):
            print(f"错误: 标注文件不存在: {anno_path}")
            sys.exit(1)
        
        print(f"\n验证图片: {image_name}")
        
        rules = validator.load_rules(rules_path)
        result = validator.validate_single_sync(
            image_path, anno_path, rules, strict_mode=args.strict
        )
        
        # 打印结果
        print("\n" + "=" * 60)
        print(f"验证结果: {result.image_name}")
        print("=" * 60)
        print(f"总标注数: {result.total_annotations}")
        
        if result.incorrect_annotations:
            print(f"\n错误标注 ({len(result.incorrect_annotations)}个):")
            for anno in result.incorrect_annotations[:5]:  # 最多显示5个
                idx = anno.get('index', 'N/A')
                label = anno.get('label', 'N/A')
                issue = anno.get('issues', anno.get('issue', 'N/A'))
                print(f"  - 标注 #{idx}: {label}")
                print(f"    问题: {issue[:60]}...")
        else:
            print("\n错误标注: 无")
        
        if result.missing_annotations:
            print(f"\n漏标物体 ({len(result.missing_annotations)}个):")
            for missing in result.missing_annotations[:3]:  # 最多显示3个
                label = missing.get('suggested_label', 'N/A')
                desc = missing.get('description', 'N/A')
                print(f"  - {label}: {desc[:50]}...")
        else:
            print("\n漏标物体: 无")
        
        print(f"\n整体评估: {result.comments}")
        if result.score is not None:
            print(f"质量评分: {result.score}/100")
        
        # 保存结果
        import json
        single_output = f"{Path(image_name).stem}_validation.json"
        with open(single_output, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到: {single_output}")
        
    else:
        # 批量验证
        if not os.path.exists(image_dir):
            print(f"错误: 图片目录不存在: {image_dir}")
            sys.exit(1)
        if not os.path.exists(anno_dir):
            print(f"错误: 标注目录不存在: {anno_dir}")
            sys.exit(1)
        
        print(f"\n图片目录: {image_dir}")
        print(f"标注目录: {anno_dir}")
        print(f"并发数: {args.concurrent}")
        if args.limit:
            print(f"限制数量: {args.limit}张")
        print(f"严格模式: {'是' if args.strict else '否'}")
        print()
        
        validator.validate_batch(
            image_dir=image_dir,
            anno_dir=anno_dir,
            rules_path=rules_path,
            output_path=output_path,
            max_concurrent=args.concurrent,
            limit=args.limit,
            strict_mode=args.strict
        )


def cmd_visualize(args):
    """可视化命令"""
    print("=" * 60)
    print("标注可视化")
    print("=" * 60)
    
    # 确定路径
    image_dir = args.image_dir or DEFAULT_IMAGE_DIR
    anno_dir = args.anno_dir or DEFAULT_ANNO_DIR
    output_dir = args.output_dir or os.path.join(DEFAULT_OUTPUT_DIR, "visualization")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    visualizer = AnnotationVisualizer(use_pil_for_text=not args.no_pil)
    
    if args.single:
        # 单张可视化
        image_name = args.single
        image_path = os.path.join(image_dir, image_name)
        anno_path = os.path.join(anno_dir, f"{Path(image_name).stem}.json")
        output_path = os.path.join(output_dir, f"{Path(image_name).stem}_vis.jpg")
        
        if not os.path.exists(image_path):
            print(f"错误: 图片不存在: {image_path}")
            sys.exit(1)
        if not os.path.exists(anno_path):
            print(f"错误: 标注文件不存在: {anno_path}")
            sys.exit(1)
        
        print(f"\n可视化图片: {image_name}")
        
        if args.validation:
            # 带验证结果的可视化
            visualizer.visualize_validation_result(
                image_path, anno_path, args.validation, output_path
            )
        else:
            # 普通可视化
            annotations = visualizer.load_annotation(anno_path)
            visualizer.draw_annotations(image_path, annotations, output_path)
        
        print(f"结果已保存到: {output_path}")
        
    else:
        # 批量可视化
        if not os.path.exists(image_dir):
            print(f"错误: 图片目录不存在: {image_dir}")
            sys.exit(1)
        if not os.path.exists(anno_dir):
            print(f"错误: 标注目录不存在: {anno_dir}")
            sys.exit(1)
        
        print(f"\n图片目录: {image_dir}")
        print(f"标注目录: {anno_dir}")
        print(f"输出目录: {output_dir}")
        if args.validation:
            print(f"验证结果: {args.validation}")
        if args.only_errors:
            print("只显示有问题的图片")
        print()
        
        visualize_batch(
            image_dir=image_dir,
            anno_dir=anno_dir,
            output_dir=output_dir,
            validation_results_path=args.validation,
            only_with_errors=args.only_errors,
            use_pil=not args.no_pil
        )


def cmd_full(args):
    """完整流程命令（验证+可视化）"""
    print("=" * 60)
    print("完整流程: 验证 + 可视化")
    print("=" * 60)
    
    api_key = get_api_key(args.api_key)
    
    # 确定路径
    image_dir = args.image_dir or DEFAULT_IMAGE_DIR
    anno_dir = args.anno_dir or DEFAULT_ANNO_DIR
    rules_path = args.rules or DEFAULT_RULES_PATH
    output_dir = args.output_dir or DEFAULT_OUTPUT_DIR
    validation_output = os.path.join(output_dir, DEFAULT_VALIDATION_OUTPUT)
    vis_output_dir = os.path.join(output_dir, "visualization_with_errors")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 步骤1: 验证
    print("\n【步骤1/2】执行验证...")
    print("-" * 60)
    
    validator = AnnotationValidator(
        api_key=api_key,
        base_url=args.base_url,
        model=args.model
    )
    
    validation_results = validator.validate_batch(
        image_dir=image_dir,
        anno_dir=anno_dir,
        rules_path=rules_path,
        output_path=validation_output,
        max_concurrent=args.concurrent,
        limit=args.limit,
        strict_mode=args.strict
    )
    
    # 步骤2: 可视化（只显示有问题的图片）
    print("\n【步骤2/2】生成可视化结果...")
    print("-" * 60)
    
    visualize_batch(
        image_dir=image_dir,
        anno_dir=anno_dir,
        output_dir=vis_output_dir,
        validation_results_path=validation_output,
        only_with_errors=True,
        use_pil=True
    )
    
    # 打印汇总
    print("\n" + "=" * 60)
    print("处理完成！")
    print("=" * 60)
    print(f"验证结果: {validation_output}")
    print(f"可视化结果: {vis_output_dir}")
    print(f"总图片数: {validation_results['summary']['total_images']}")
    print(f"有问题图片数: {validation_results['summary']['images_with_errors']}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="数据集标注验证工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 验证单张图片
  python run.py validate -s image.jpg
  
  # 批量验证前10张图片
  python run.py validate -l 10 -c 3
  
  # 可视化单张图片
  python run.py visualize -s image.jpg
  
  # 可视化所有图片（带验证结果）
  python run.py visualize -v validation_results.json
  
  # 完整流程（验证+可视化有问题图片）
  python run.py full -l 20
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 验证命令
    validate_parser = subparsers.add_parser('validate', help='验证标注质量')
    validate_parser.add_argument('-s', '--single', help='验证单张图片')
    validate_parser.add_argument('-i', '--image-dir', help='图片目录')
    validate_parser.add_argument('-a', '--anno-dir', help='标注目录')
    validate_parser.add_argument('-r', '--rules', help='规则文件路径')
    validate_parser.add_argument('-o', '--output', help='输出JSON文件路径')
    validate_parser.add_argument('-c', '--concurrent', type=int, default=5, help='并发数（默认5）')
    validate_parser.add_argument('-l', '--limit', type=int, help='限制验证图片数量')
    validate_parser.add_argument('--strict', action='store_true', help='使用严格模式')
    validate_parser.add_argument('--api-key', help='API Key')
    validate_parser.add_argument('--base-url', default='https://dashscope.aliyuncs.com/compatible-mode/v1', help='API基础URL')
    validate_parser.add_argument('--model', default='qwen-vl-max', help='模型名称')
    validate_parser.set_defaults(func=cmd_validate)
    
    # 可视化命令
    visualize_parser = subparsers.add_parser('visualize', help='可视化标注')
    visualize_parser.add_argument('-s', '--single', help='可视化单张图片')
    visualize_parser.add_argument('-i', '--image-dir', help='图片目录')
    visualize_parser.add_argument('-a', '--anno-dir', help='标注目录')
    visualize_parser.add_argument('-o', '--output-dir', help='输出目录')
    visualize_parser.add_argument('-v', '--validation', help='验证结果JSON文件')
    visualize_parser.add_argument('--only-errors', action='store_true', help='只显示有问题的图片')
    visualize_parser.add_argument('--no-pil', action='store_true', help='不使用PIL（可能不支持中文）')
    visualize_parser.set_defaults(func=cmd_visualize)
    
    # 完整流程命令
    full_parser = subparsers.add_parser('full', help='完整流程（验证+可视化）')
    full_parser.add_argument('-i', '--image-dir', help='图片目录')
    full_parser.add_argument('-a', '--anno-dir', help='标注目录')
    full_parser.add_argument('-r', '--rules', help='规则文件路径')
    full_parser.add_argument('-o', '--output-dir', help='输出目录')
    full_parser.add_argument('-c', '--concurrent', type=int, default=5, help='并发数（默认5）')
    full_parser.add_argument('-l', '--limit', type=int, help='限制验证图片数量')
    full_parser.add_argument('--strict', action='store_true', help='使用严格模式')
    full_parser.add_argument('--api-key', help='API Key')
    full_parser.add_argument('--base-url', default='https://dashscope.aliyuncs.com/compatible-mode/v1', help='API基础URL')
    full_parser.add_argument('--model', default='qwen-vl-max', help='模型名称')
    full_parser.set_defaults(func=cmd_full)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == "__main__":
    main()
