# 使用说明

这个精简包已经包含 webapp 和 SAM3 运行所需源码，但不包含 SAM3 权重文件，也不包含历史上传数据。

导入图片和标注后，程序会维护一份工作副本，而不是直接改原始上传文件。默认工作副本位置是：

```text
runs/work_dataset/
  images/
  annotations/
  manifests/
```

图片和标注 JSON 会尽量保持原文件名。mask 迭代、删除框、删除整张图等网站操作会实时写到这份工作副本里；COCO 格式标注会在原 annotation 对象中补充 `segmentation`、`area`、`iscrowd`，并保留原有文件名和其他字段。

最新版导入逻辑会把图片集和标注状态分开维护，方便同一批图片反复生成不同轮次的标注副本：

```text
runs/work_dataset/
  images/
    <图片集命名>/
      ...
  annotations/
    <标注状态命名>/
      ...
  manifests/
    <标注状态命名>.json
```

例如同一批原图可以只保存一份 `images/无人机原图/`，然后分别维护 `annotations/原始/`、`annotations/第2轮/`、`annotations/复核版/`。重新导入同一个“标注状态命名”时，如果工作副本里已经有同名 JSON，程序会优先读取这个副本继续做，不会用原始上传 JSON 覆盖它。直接导入原始图片文件夹和原始标注文件夹仍然可用；如果不手动填写命名，网页会默认使用所选文件夹名生成图片集和标注状态。

网页里的“导出当前副本”会把当前图片集和当前标注状态复制到：

```text
runs/exports/<导出副本命名>/
  images/
  annotations/
  manifest.json
```

`manifest.json` 会记录导出来源和图片/标注配对检查结果，便于后续复核或交付。

## 1. 安装依赖

推荐使用项目自带的 conda 初始化脚本：

```bash
./setup_conda.sh
```

脚本会创建或更新 `mask_iteration_sam3` 环境，安装 PyTorch、SAM3、Validate_tools 相关依赖，并把桌面上的 `sam3.pt` 放到项目默认 checkpoint 位置。

如果手动配置，先创建虚拟环境，然后安装这个包的依赖：

```bash
pip install -r requirements.txt
```

再单独安装适合你机器的 `torch`：

- NVIDIA 显卡：安装 CUDA 版 `torch`
- Apple Silicon / Mac：安装支持 `mps` 的 `torch`
- 没有 GPU：安装 CPU 版 `torch`

## 2. SAM3 权重去哪里下载

SAM3 源码已经在这个包里，只需要你自己下载权重文件，权重不在这个包里。

下载完成后，需要准备好 `sam3.pt`。

## 3. 权重下载之后放在哪里

把权重文件放到：

```text
third_party/sam3/checkpoints/sam3.pt
```

完整结构应该是：

```text
mask_iteration_webapp_bundle_slim_sam3_no_weights/
  start_webapp.py
  web/
  mask_iteration_webapp/
  third_party/
    sam3/
      sam3/
        __init__.py
        model/
        assets/
          bpe_simple_vocab_16e6.txt.gz
      checkpoints/
        sam3.pt
      pyproject.toml
```

注意：`third_party/sam3` 已经是 SAM3 源码目录，不要把它替换成只包含 checkpoint 的空目录。

## 4. 如何启动程序

### 进入项目目录

先在终端进入这个项目的根目录，也就是包含 `start_webapp.py` 的目录：

```powershell
cd path\to\mask_iteration_webapp_bundle_slim_sam3_no_weights
```

如果你使用 conda，可以先进入自己创建的环境，例如：

```powershell
conda activate your_env_name
```

### Windows / PowerShell

推荐先运行 Windows 初始化脚本：

```powershell
.\setup_conda.bat
```

之后启动：

```powershell
.\run_conda.bat
```

也可以手动启动。`start_webapp.py` 会在当前 Python 缺少 `torch` 时自动寻找 conda，并重启到 `mask_iteration_sam3` 环境：

```powershell
python start_webapp.py --sam3-repo-dir ".\third_party\sam3" --checkpoint ".\third_party\sam3\checkpoints\sam3.pt" --device auto --validate-tools-dir ".\Validate_tools"
```

Windows + NVIDIA 显卡时可以在环境建好后按你的 CUDA 版本替换安装对应的 PyTorch CUDA wheel；没有 NVIDIA/CUDA 时默认 pip 安装的 CPU 版也可以运行，只是会慢。

### macOS / Linux

如果已经运行过 `./setup_conda.sh`，直接启动：

```bash
./run_conda.sh
```

也可以手动启动：

```bash
python start_webapp.py --sam3-repo-dir ./third_party/sam3 --checkpoint ./third_party/sam3/checkpoints/sam3.pt --device auto --validate-tools-dir ./Validate_tools
```

如果是 Apple Silicon，也可以指定：

```bash
python start_webapp.py --sam3-repo-dir ./third_party/sam3 --checkpoint ./third_party/sam3/checkpoints/sam3.pt --device mps --validate-tools-dir ./Validate_tools
```

启动后打开：

```text
http://127.0.0.1:8765/merged.html
```

如果 `8765` 端口已经被占用，可以换一个端口，例如：

```bash
python start_webapp.py --sam3-repo-dir ./third_party/sam3 --checkpoint ./third_party/sam3/checkpoints/sam3.pt --device auto --validate-tools-dir ./Validate_tools --port 8766
```

然后打开：

```text
http://127.0.0.1:8766/merged.html
```

### 上传数据

网页里选择：

```text
图片文件夹：选择存放图片的文件夹
标注文件夹 / 标注副本状态：选择存放 JSON 标注文件的文件夹
```

可选填写：

- 图片集命名：同一批图片建议保持同一个名字，便于复用图片副本。
- 标注状态命名：每一轮标注建议使用不同名字，例如 `原始`、`第2轮`、`复核版`。
- 导出副本命名：点击“导出当前副本”时使用；不填写时程序会用标注状态和时间生成。

当前程序支持：

- Label Studio 风格的 `rectanglelabels` JSON
- COCO 单图 JSON，也就是 `images` / `annotations` / `categories` 结构

## 5. GitHub 同步

后续功能修改建议通过 GitHub 提交并推送，便于保留每次迭代记录。
