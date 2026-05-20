# SAM3 Mask Iteration Web App

这是一个用于图片标注检查和 mask 迭代的本地 Web 工具包。仓库已经包含 Web 后端、前端页面、SAM3 源码和 `Validate_tools`，但不包含 SAM3 权重，也不应该把历史运行数据打包进来。

## 需要保留的文件

核心运行文件只有这些：

```text
start_webapp.py                 # 启动入口
mask_iteration_webapp/          # Python 后端服务
web/mask_iteration_app/         # 前端页面
Validate_tools/                 # 标注验证工具
third_party/sam3/               # SAM3 源码，不含权重
requirements.txt                # Python 依赖
```

下面这些是运行时生成的文件，通常不用提交；确认不需要历史结果后可以清理：

```text
runs/                           # 上传副本、迭代记录、导出副本、验证结果
__pycache__/                    # Python 缓存
.pytest_cache/                  # pytest 缓存
third_party/sam3/checkpoints/sam3.pt
                                # 本地 SAM3 权重；运行要用，但不要提交或打包
```

`third_party/sam3` 里的多个 `__init__.py` 看起来像重复文件，但它们是 Python 包结构的一部分，不建议删除。

## 配置环境

只保留 `requirements.txt` 作为配环境入口。建议先创建你自己的 Python 虚拟环境，然后安装依赖：

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` 已经包含 `torch` 和 `torchvision`。如果你的机器需要特定 CUDA 版本，可以按 PyTorch 官方安装方式替换对应 wheel。

## 启动

```powershell
python start_webapp.py --checkpoint .\third_party\sam3\checkpoints\sam3.pt --device auto
```

端口被占用时加 `--port`：

```powershell
python start_webapp.py --port 8766
```

启动后打开：

```text
http://127.0.0.1:8765/merged.html
```

## 权重位置

默认权重路径是：

```text
third_party/sam3/checkpoints/sam3.pt
```

源码目录 `third_party/sam3` 已经在包里，不要把它替换成只包含 checkpoint 的空目录。

## 数据和副本目录

网页导入图片和 JSON 后，程序会写入 `runs/` 下的工作副本，不会直接改原始上传文件。默认启动参数里 `--imports-root` 指向 `runs/work_dataset`，但每个可继续导入、继续迭代、可导出的副本会按副本 ID 单独放在 `runs/<copy_id>/`。

运行时常见顶层结构如下：

```text
runs/
  <copy_id>/                    # 一份图片 + 标注的工作副本或导出副本
  <another_copy_id>/            # 另一轮导入、复核或导出的副本
  sessions/                     # 当前交互会话、提示点、历史 mask、logits、验证记录
  work_dataset/                 # 默认导入根目录，保留 manifests 等兼容数据
  manifests/                    # 旧版或兼容清单，程序启动时也会扫描
  server_logs/                  # 如果用后台脚本启动，可能会写入 stdout/stderr 日志
```

单个副本文件夹的结构示意图：

```text
runs/<copy_id>/
  manifest.json                 # 副本清单：副本 ID、源文件、目标列表、导出信息、路径映射
  quality_marked_targets.csv    # 标注错误框、难处理目标的质量标记列表，按操作产生
  images/
    keep/                       # 保留的图片工作副本
      <image_name>.jpg
    delete/                     # 标记整图删除后归档的图片
      <image_name>.jpg
  annotations/
    keep/
      coco/                     # 可交付 COCO JSON，保存当前 mask/segmentation
        <image_stem>.json
      state/                    # Web App 可恢复状态，包含 target、session、历史记录
        <image_stem>.json
    delete/
      coco/                     # 删除类目标对应的 COCO 输出，按实际操作产生
        <image_stem>.json
      state/                    # 删除类目标对应的可恢复状态
        <image_stem>.json
```

其中 `keep` 是当前保留队列，`delete` 是明确删除目标或删除整图后产生的归档目录；目录不会一开始全部存在，只有发生对应操作后才会创建。`annotations/*/coco/` 面向交付和复核，`annotations/*/state/` 面向 Web App 重新导入和恢复迭代上下文。

“标注错误框”和“难处理目标”属于质量标记，不等同于删除。后端会把它们记录到 `runs/<copy_id>/quality_marked_targets.csv`；难处理目标还会在 `runs/sessions/difficult_targets/` 下保存索引和一份标注快照。它们不会从当前目标队列里移除，也不会从当前 `annotations/keep/coco/` 标注副本里删掉。

“模糊图”按整图删除处理。后端会写入 `runs/sessions/blurry_images/` 和 `runs/sessions/image_deletions.json`，并把该图片相关目标从当前队列和 `annotations/keep/coco/` 标注副本中移除。

同一批图片建议保持同一个“图片集命名”；每一轮标注使用不同“标注状态命名”，例如 `原始`、`第2轮`、`复核版`。重新导入同名标注状态时，程序会优先读取已有工作副本，避免用原始 JSON 覆盖已迭代结果。

网页里的“导出当前副本”会从源副本复制出一个新的 `runs/<export_name>/` 目录，重写 `manifest.json` 里的 `copy_id`、`copy_root`、目标 key 和文件路径，用于交付或复核。导出的副本仍然可以再次通过页面导入，继续作为下一轮迭代的起点。

## 支持的标注格式

- Label Studio 风格的 `rectanglelabels` JSON
- COCO 单图 JSON，也就是包含 `images` / `annotations` / `categories` 的结构

COCO 标注会在工作副本里补充 `segmentation`、`area`、`iscrowd` 等字段，并保留原有文件名和其他字段。

## 常用维护命令

只清理缓存：

```powershell
Remove-Item -Recurse -Force .pytest_cache, __pycache__ -ErrorAction SilentlyContinue
Get-ChildItem -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force
```

清理历史运行数据前请先确认 `runs/` 里没有需要保留的上传副本、导出结果或验证结果：

```powershell
Remove-Item -Recurse -Force runs
```
