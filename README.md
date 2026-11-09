# 精简核心代码（Ours）

这里只保留本方法的推理核心：水印嵌入、反演提取、用户映射、完整候选池识别。
不包含训练、攻击生成、基线、批量实验、监控、论文统计、图片、权重或缓存。
原实验目录和最终交付包保持不变；这不是一套重新跑过全部实验的替代实现。

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
| `test_core.py` | 无权重、无 GPU 的逻辑测试 |

## 准备资源

修改 `config.json` 中的本机资源路径，消息及哈希默认保留正式 R4 值：

1. 已有 SD2.1-base 本地模型目录，不会自动联网下载。
2. 正式 `wm_model.ckpt`，其同目录必须有 `wm_model_config.yaml`。
3. 原注册池 `positions.npy` 和 `transforms.npy`，候选顺序即数组行号。

默认路径指向原 Linux 服务器，不是 Windows 可直接使用的资产路径。
权重、SD 模型和注册池不复制进本目录；只加载自己信任的 checkpoint。
需要安装依赖时可运行 `python -m pip install -r requirements.txt`，但尚未验证全新环境安装。
训练原入口仍在 `E:/RW/train_V3.py`，未混入这份推理精简目录。

## 使用

在本目录中运行，输出文件必须使用新路径，不覆盖已有结果：

```bash
python main.py --help
python main.py embed --image input.png --user-id 500 --output protected.png --gpu 1
python main.py extract --image protected.png --user-id 500 --output extracted.json --gpu 1
python main.py trace --image protected.png --output traced.json --pools 10000 100000 1000000 --gpu 1
python -m unittest test_core -v
```

`--gpu` 是原机器上的物理卡号，只允许 1/2/3，运行前会设置可见卡；不要在已重映射 GPU 的外层环境中误解卡号。
图像按原 R4 路径转换为 RGB、ToTensor 后 resize 为 512×512。
嵌入使用 VAE mode、绝对 encoder 和逆映射；提取保留空 prompt、guidance=1、15/3 步反演。
编码/解码分数精度、64 位消息、B256/K5、batch256、TF32 关闭均不主动改动。

`extract` 是已知映射的检测，不等于开放集追踪。
`trace` 完整遍历最大候选池，同步记录真实较小池前缀；不候选抽样、不近似检索、不提前停止。
唯一全池最高分至少 57/64 才接受，最高分并列则拒识；`top1_user_id` 在拒识时只是闭集第一名，不代表接受用户。
映射分配器使用数值 seed，不宣称密码学密钥安全。

## 整理与验证边界

7 个算法模块来自已验收发布候选的哈希绑定源文件。整文件复制或按 AST 原样抽取定义，仅隔离必要 import。
`watermark.py`、`main.py` 是新加的薄封装，不是新的科学实验或新模型。
本次只进行源哈希/AST 一致性、CPU 映射/评分/判决测试、语法检查和命令行检查；
不声称该封装已在正式权重上重新完成 GPU 端到端验收，也不触发任何旧评测任务。
机械整理脚本在 `E:/RW/tools/build_core_code.py`；没有公开上传。
