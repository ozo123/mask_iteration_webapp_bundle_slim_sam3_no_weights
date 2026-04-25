# SAM3 权重放这里

这个精简包故意没有打包 SAM3 权重文件。

请把下载好的权重文件放成：

```text
third_party/sam3/checkpoints/sam3.pt
```

然后在包根目录运行：

```bash
python start_webapp.py --sam3-repo-dir ./third_party/sam3 --checkpoint ./third_party/sam3/checkpoints/sam3.pt --device auto
```
