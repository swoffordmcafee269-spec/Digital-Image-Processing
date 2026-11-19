# hw2 · 图像增强（RAW 色彩恢复）

数字图像处理实习一。将相机 RAW 线性数据还原为观感正确的彩色图像，处理重点在色彩恢复与色调映射。

## 两套实现

| 目录 | 定位 |
|---|---|
| `图像增强/` | 迭代主线 v1–v5。`src/` 为基础实现，`v5/code/` 为第五轮改进版（含 `scripts/` 实验脚本与 `src/` 网络结构） |
| `图像增强2/` | 重构版 `canon_enhance` 包。模块化程度更高，带单元测试、验收检查、EMA、批量推理 |

两套代码**不是重复**，`train.py` 等文件 MD5 不同，是不同轮次的不同实现。若要复现最终结论，以 `图像增强2/` 为准；`图像增强/v5/` 保留了改进过程的中间版本。

## 结果指标

已在仓库内的 CSV 结果表（无需重跑）：

- `图像增强2/源代码.../results/baseline_metrics.csv`、`final_metrics.csv`
- `图像增强/v5/.backup_csv/{Canon,Sony}/{baseline,results}.csv`、`logs/train_log.csv`

## 运行

```powershell
cd 图像增强2
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-lock.txt

python -m unittest discover -s tests -v     # 自检
python -m canon_enhance.prepare             # 读取 1000 对数据、建 GT 缓存、划分场景分组
```

训练与续训命令见 `图像增强2/部署文档.md`。

## 数据准备（必须，数据未上传）

- 输入为 1000 对 RAW / JPEG，放在**外部只读目录**，路径由 `图像增强2/configs/canon.json` 指定
- 预留约 **60 GB** 磁盘：GT 解码缓存约 25 GB，加上中间 PNG、模型与日志
- 程序会校验权重与数据清单指纹，数据、场景划分、色彩标定、训练与推理配置必须配套
- 随机种子固定，但 GPU 内核允许性能优化，不承诺跨设备逐位一致

## 环境

PyTorch 2.11.0+cu128 / torchvision 0.26.0，实测于 RTX 5060 Laptop 8 GB。精确版本见 `图像增强2/requirements-lock.txt`。

## 本分支未包含

| 内容 | 体积 | 原因 |
|---|---|---|
| `图像增强2/artifacts/` | 55.9 GB | 运行产物与检查点 |
| `图像增强/data/` | 34.8 GB | 原始数据集 |
| `图像增强/outputs/` | 14.5 GB | 训练输出 |
| 训练检查点 `*.pt` | 单个 188–252 MB | **超过 GitHub 单文件 100 MiB 硬限制，无法上传** |
| `图像增强2/源代码（含部署文档）及结果数据表格/` | — | 与父目录逐字节重复 |

检查点未上传意味着无法直接推理，但训练流程、配置与结果指标完整，按 `部署文档.md` 可从数据重新训练。
