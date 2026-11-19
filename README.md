# hw1 · 文档图像几何校正

数字图像处理实习二。对 200 张带畸变的文档照片做几何校正（去畸变 / 拉平），并与其他方法对照评价。

## 方法构成

| 目录 | 内容 |
|---|---|
| `几何校正workbuddy2/baseline/` | 方案 A：基于 mask 的传统校正——mask 轮廓 → 四角点检测 → 排序 → 透视变换拉平 |
| `几何校正workbuddy2/uvdoc/` | UVDoc 深度学习去畸变（含预训练权重 `UVDoc/model/best_model.pkl`，30.7 MB） |
| `几何校正workbuddy/DocGeoNet/` | DocGeoNet 对照实现（含预训练权重 `model_pretrained/DocGeoNet.pth`，90.8 MB） |
| `几何校正workbuddy2/common/` | 图像读写工具 `imgio.py`，统一处理中文路径 |
| `几何校正workbuddy2/eval/` | 评价脚本（见下）与结果表 |

## 评价脚本

按编号顺序执行：

| 脚本 | 产出 |
|---|---|
| `eval/03_visual_compare.py` | 方法间可视化对比图 |
| `eval/04_line_regularity.py` | 直线规则性指标 → `eval/metric_line_regularity.csv` |
| `eval/05_ocr_eval.py` | OCR 可读性指标 → `eval/metric_ocr.csv` |
| `eval/06_summary_stats.py` | 汇总统计 → `eval/summary_stats.csv` |
| `baseline/01_baseline.py` | 基线校正结果 → `eval/baseline_log.csv` |

结果表已在仓库内，无需重跑即可查看指标：`summary_stats.csv` 记录各指标相对原图的均值 / 中位数 / 标准差（样本数 200），并标注方向（如 curvature 越小越好、line_var 越大越好）。

## 数据准备（必须，数据未上传）

`GeoCorrection.zip`（约 790 MB）解压到：

```
几何校正workbuddy2/GeoCorrection/_extract/GeoCorrection/
├── images/   200 张畸变文档图（.jpg）
└── mask/     200 张同名二值掩码（.png，文档区域为 255）
```

脚本通过 `ROOT/GeoCorrection/_extract/GeoCorrection/images` 定位输入，路径不对会直接报错。

## 环境

Python 3.13 实测。`requirements-lock.txt` 为从实际 venv 精确导出的 32 个包（含版本号）。

```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-lock.txt
```

无 GPU 可用 CPU 版 torch，UVDoc 推理会自动回退，仅速度变慢。

## 本分支未包含

- `GeoCorrection/*.zip`（790 MB，超单文件限制）与解压后的 `images/` `mask/`（约 1.6 GB）
- `venv/`（4.5 GB，已导出为 `requirements-lock.txt`）
- `_作业题目.txt`（含提交邮箱，不上传）
- 与 `几何校正workbuddy2/` 逐字节重复的 `源代码及结果数据表格/` 打包副本

更详细的安装与运行说明见 `几何校正workbuddy2/部署文档.md`，结论见 `总结文档.md`。
