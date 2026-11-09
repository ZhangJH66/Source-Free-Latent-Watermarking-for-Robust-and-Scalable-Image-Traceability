# 精简核心代码（Ours）

方法的推理核心：水印嵌入、反演提取、用户映射、完整候选池识别。


## 文件

| 文件 | 作用 |
|---|---|
| `main.py` | 单图命令行入口：embed / extract / trace |
| `watermark.py` | 精简调用封装、加载资源、图像预处理与扫描流程 |
| `model.py` | 原始水印编码器/解码器结构，整文件保留 |
| `inversion.py` | 原始扩散反演实现，整文件保留并保留来源注释 |
| `mapping.py` | 原始潜空间局部变换和置乱/逆置乱 |
| `codec.py` | 原始映射分配及绝对编码器嵌入逻辑 |
| `checkpoint.py` | 原始 checkpoint 加载函数 |
| `scan.py` | 原始精确批量候选评分，包括浮点边界的 batch-one 复核 |
| `decision.py` | 原始流式前缀池统计和唯一最大值判决 |
| `config.json` | 正式模型、消息和注册池路径及已验收哈希 |
| `requirements.txt` | 精简推理依赖，版本依据原 R4 环境 |
| `SOURCE_MANIFEST.json` | 原文件来源、哈希及抽取符号 |

## 准备资源

修改 `config.json` 中的本机资源路径，消息及哈希默认保留正式 R4 值：

1. 已有 SD2.1-base 本地模型目录，不会自动联网下载。
2. 正式 `wm_model.ckpt`，其同目录必须有 `wm_model_config.yaml`。
3. 原注册池 `positions.npy` 和 `transforms.npy`，候选顺序即数组行号。

需要安装依赖时可运行 `python -m pip install -r requirements.txt`

## 使用

在本目录中运行，输出文件必须使用新路径，不覆盖已有结果：

```bash
python main.py --help
python main.py embed --image input.png --user-id 500 --output protected.png --gpu 1
python main.py extract --image protected.png --user-id 500 --output extracted.json --gpu 1
python main.py trace --image protected.png --output traced.json --pools 10000 100000 1000000 --gpu 1
python -m unittest test_core -v
```


