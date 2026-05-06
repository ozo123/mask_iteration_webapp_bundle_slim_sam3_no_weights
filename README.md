# 使用说明

这个精简包已经包含 webapp 和 SAM3 运行所需源码，但不包含 SAM3 权重文件，也不包含历史上传数据。

## 1. 安装依赖

先创建虚拟环境，然后安装这个包的依赖：

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

```powershell
python start_webapp.py --sam3-repo-dir ".\third_party\sam3" --checkpoint ".\third_party\sam3\checkpoints\sam3.pt" --device auto --validate-tools-dir ".\Validate_tools"
```

### macOS / Linux

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
标注文件夹：选择存放 JSON 标注文件的文件夹
```

当前程序支持：

- Label Studio 风格的 `rectanglelabels` JSON
- COCO 单图 JSON，也就是 `images` / `annotations` / `categories` 结构

## 5. GitHub 同步

后续功能修改建议通过 GitHub 提交并推送，便于保留每次迭代记录。
