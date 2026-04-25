# 数据集标注验证工具包

> 基于大模型API的室内无人机数据集目标检测标注质量验证工具

## 工具概述

本工具包用于验证目标检测标注的准确性和完整性，通过调用LLM（大语言模型）API自动检测：

- **错误标注**：类别错误、位置偏差、尺寸不合理
- **漏标物体**：图片中未标注的可见目标
- **标注质量评分**：0-100分的整体质量评估

### 工作流程

```
输入：原始图片 + JSON标注文件
  ↓
调用LLM API分析（支持并发）
  ↓
输出：
  1. validation_results.json - 详细的验证结果
  2. visualization/ - 带标注框的可视化图片（错误标红）
```

---

## 目录结构

```
Validate_tools/
├── annotation_validator.py    # 核心验证模块（请勿修改）
├── visualization_tool.py      # 可视化模块（请勿修改）
├── run.py                     # 主入口脚本
├── rules.json                 # 类别定义规则
├── requirements.txt           # Python依赖包
└── README.md                  # 本说明文档
```

### 数据目录（需自行准备）

```
原始数据文件/
├── 20251101-1111/                # 原始图片目录（.jpg）
├── per_image_annotations/        # 标注文件目录（.json）
│   ├── 00d1ce01-DJI0029_frame_013300.json
│   └── ...
└── results/                      # 输出结果目录
    ├── validation_results.json
    └── visualization/
```

---

## 环境准备

### 1. 安装Python依赖

```bash
cd Validate_tools
pip install -r requirements.txt
```

**依赖包说明：**
- `openai` - OpenAI格式API客户端
- `aiohttp` - 异步HTTP请求（支持并发）
- `opencv-python` - 图像处理
- `Pillow` - 图像绘制（支持中文）
- `numpy` - 数值计算

### 2. 配置API Key

本工具使用**DashScope（通义千问）**API，需要设置API Key。

**方式一：环境变量（推荐）**

```powershell
# Windows PowerShell
$env:DASHSCOPE_API_KEY="your-api-key-here"

# 或者 Windows CMD
set DASHSCOPE_API_KEY=your-api-key-here

# 验证是否设置成功
echo $env:DASHSCOPE_API_KEY
```

**方式二：运行时输入**

如果不设置环境变量，运行脚本时会提示输入。

> **获取API Key**: 访问 https://dashscope.aliyun.com/ 申请

---

## 快速开始

### 命令概览

```bash
python run.py [command] [options]

Commands:
  validate     验证标注质量
  visualize    可视化标注
  full         完整流程（验证 + 可视化）
```

---

## 使用详解

### 1. 验证模式（validate）

验证标注质量，输出JSON格式的详细结果。

#### 1.1 验证单张图片

```bash
python run.py validate -s image_name.jpg
```

**示例：**
```bash
python run.py validate -s 00d1ce01-DJI0029_frame_013300.jpg
```

输出：`00d1ce01-DJI0029_frame_013300_validation.json`

#### 1.2 批量验证

```bash
# 验证全部图片
python run.py validate

# 验证前20张图片，并发数为3
python run.py validate -l 20 -c 3

# 验证全部，并发数为5
python run.py validate -c 5
```

**常用参数：**

| 参数 | 简写 | 说明 | 示例 |
|------|------|------|------|
| `--single` | `-s` | 单张图片验证 | `-s image.jpg` |
| `--limit` | `-l` | 限制验证数量 | `-l 50` |
| `--concurrent` | `-c` | 并发数（默认5） | `-c 3` |
| `--strict` | | 严格模式（更多报错） | `--strict` |
| `--output` | `-o` | 输出JSON路径 | `-o results.json` |

**完整示例：**
```bash
python run.py validate \
  -i "E:\\UAV\\Test\\Benchmark\\20251101-1111" \
  -a "E:\\UAV\\Test\\Benchmark\\per_image_annotations" \
  -r ".\\rules.json" \
  -o "E:\\UAV\\Test\\Benchmark\\results\\validation_results.json" \
  -l 100 \
  -c 5
```

---

### 2. 可视化模式（visualize）

将标注框绘制在图片上，错误标注用**红色**高亮显示。

#### 2.1 可视化单张图片

```bash
python run.py visualize -s image_name.jpg
```

#### 2.2 批量可视化

```bash
# 可视化所有图片
python run.py visualize

# 结合验证结果可视化（错误标红）
python run.py visualize -v validation_results.json

# 只可视化有问题的图片
python run.py visualize -v validation_results.json --only-errors
```

**常用参数：**

| 参数 | 简写 | 说明 | 示例 |
|------|------|------|------|
| `--single` | `-s` | 单张图片 | `-s image.jpg` |
| `--validation` | `-v` | 验证结果JSON | `-v results.json` |
| `--only-errors` | | 只显示有问题图片 | `--only-errors` |
| `--output-dir` | `-o` | 输出目录 | `-o vis_output` |
| `--no-pil` | | 不使用PIL（可能中文乱码） | `--no-pil` |

---

### 3. 完整流程（full）

自动执行 **验证 → 可视化（有问题图片）**

```bash
# 验证50张图片并生成可视化
python run.py full -l 50 -c 5

# 指定输出目录
python run.py full -l 100 -o "E:\\results"
```

**输出结构：**
```
results/
├── validation_results.json      # 验证结果
└── visualization/               # 可视化图片（仅有问题图片）
    ├── image1_vis.jpg
    └── image2_vis.jpg
```

---

## 数据格式说明

### 输入 - 标注文件格式

每个图片对应一个同名 `.json` 文件：

```json
{
  "results": [
    {
      "value": {
        "x": 25.0,              // 中心点 X 坐标（百分比 0-100）
        "y": 66.5,              // 中心点 Y 坐标（百分比 0-100）
        "width": 8.9,           // 宽度（百分比 0-100）
        "height": 14.0,         // 高度（百分比 0-100）
        "rectanglelabels": ["chair"]  // 类别标签
      }
    },
    {
      "value": {
        "x": 45.2,
        "y": 30.1,
        "width": 12.5,
        "height": 20.3,
        "rectanglelabels": ["person"]
      }
    }
  ]
}
```

### 输出 - 验证结果格式

```json
{
  "summary": {
    "total_images": 100,           // 验证图片总数
    "total_annotations": 450,      // 总标注数
    "images_with_errors": 12,      // 有错误图片数
    "images_with_errors_only": 10, // 只有错误无漏标
    "images_with_missing_only": 2  // 只有漏标无错误
  },
  "results": [
    {
      "image_name": "xxx.jpg",
      "total_annotations": 5,
      "incorrect_annotations": [     // 错误标注列表
        {
          "index": 2,                // 标注序号（从1开始）
          "label": "chair",
          "issue": "错标为chair，实际应为couch",
          "confidence": "high",
          "suggestion": "修改为couch"
        }
      ],
      "missing_annotations": [       // 漏标列表
        {
          "location": "图片右下角",
          "description": "有一个红色杯子未被标注",
          "suggested_label": "cup",
          "confidence": "high"
        }
      ],
      "comments": "整体标注质量良好，有少量漏标",
      "score": 85                    // 质量评分 0-100
    }
  ]
}
```

---

## 编程接口（高级用法）

如需在Python代码中调用，可直接导入模块：

```python
from annotation_validator import AnnotationValidator, validate_single
from visualization_tool import visualize_single

# ========== 单张验证 ==========
result = validate_single(
    image_path="path/to/image.jpg",
    anno_path="path/to/annotation.json",
    rules_path="path/to/rules.json",
    api_key="your-api-key",  # 或设置环境变量 DASHSCOPE_API_KEY
    strict_mode=False
)

print(f"错误数: {len(result.incorrect_annotations)}")
print(f"漏标数: {len(result.missing_annotations)}")
print(f"评分: {result.score}")

# ========== 批量验证 ==========
validator = AnnotationValidator(api_key="your-api-key")

results = validator.validate_batch(
    image_dir="path/to/images",
    anno_dir="path/to/annotations",
    rules_path="path/to/rules.json",
    output_path="results.json",
    max_concurrent=5,
    limit=50
)

# ========== 可视化 ==========
visualize_single(
    image_path="path/to/image.jpg",
    anno_path="path/to/annotation.json",
    output_path="output.jpg",
    validation_result=result.to_dict()  # 可选，传入则标红错误
)
```

---

## 类别定义

本数据集支持56个类别，定义在 `rules.json` 中：

**主要类别包括：**
- **家具**: chair, couch, bed, dining table, cabinet, potted plant
- **电子设备**: tv, laptop, cell phone, keyboard, mouse, remote, microwave
- **厨房用品**: bottle, cup, bowl, sink, refrigerator, oven
- **食物**: banana, apple, orange, cake, pizza, sandwich
- **办公用品**: book, backpack, handbag, clock, scissors
- **其他**: person, teddy bear, vase, pillow, cloth, picture

**查看完整类别定义：**
```bash
cat rules.json | python -m json.tool
```

---

## 常见问题（FAQ）

### Q1: API调用失败或超时？

**可能原因及解决：**
- API Key错误 → 检查 `DASHSCOPE_API_KEY` 是否正确设置
- 网络问题 → 检查网络连接
- API限流 → 降低并发数 `-c 3` 或减少验证数量 `-l 10`
- 超时 → 单张图片分析需要10-30秒，批量验证请耐心等待

### Q2: 如何提高验证速度？

- 增加并发数 `-c`（但不要超过API限制，建议3-5）
- 使用 `--limit` 参数先验证少量图片测试
- 确保网络稳定

### Q3: 可视化图片中文显示乱码？

- 默认使用PIL绘制中文，如果仍有问题，请确保系统安装了中文字体
- Windows系统通常自带黑体（simhei.ttf）
- 如不需要中文，可使用 `--no-pil` 参数

### Q4: 可以更换其他API吗？

支持所有OpenAI兼容格式的API：
```bash
python run.py validate \
  --base-url "https://api.example.com/v1" \
  --model "your-model-name"
```

### Q5: 严格模式和普通模式区别？

- **普通模式**（默认）：只报告高置信度错误（>90%确信），适合快速筛选
- **严格模式**（--strict）：报告更多潜在问题，适合精细审核

---

## 性能参考

在标准办公网络环境下：

| 数量 | 并发数 | 预计时间 |
|------|--------|----------|
| 10张 | 3 | ~2分钟 |
| 50张 | 5 | ~8分钟 |
| 100张 | 5 | ~15分钟 |
| 500张 | 5 | ~70分钟 |

> 注：实际时间受网络状况和API响应速度影响

---

## 注意事项

1. **API费用**：本工具调用大模型API会产生费用，请根据预算控制验证数量
2. **隐私安全**：图片会上传到云端API处理，请勿包含敏感信息
3. **结果参考**：LLM判断可能存在误差，建议对标记为错误的样本进行人工复核
4. **路径问题**：Windows路径建议使用双反斜杠 `\\` 或原始字符串

---

## 版本记录

- **v1.0** (2025-04-13)
  - 整合原有分散脚本为统一工具包
  - 支持异步并发验证
  - 支持中文标签可视化
  - 统一的命令行入口

---

## 技术支持

如遇问题，请检查：
1. 本README文档
2. 运行 `--help` 查看命令帮助：`python run.py validate --help`
3. 检查日志输出中的错误信息

---

**祝使用愉快！**
