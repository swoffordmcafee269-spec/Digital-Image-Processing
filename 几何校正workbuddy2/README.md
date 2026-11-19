# 数字图像处理实习二 · 文档图像几何校正

> 将存在透视变形、页面弯曲、倾斜的文档照片，恢复为平整、规整、便于阅读与 OCR 识别的图像。

![Python](https://img.shields.io/badge/Python-3.13-blue) ![Task](https://img.shields.io/badge/Task-Document%20Dewarping-orange) ![Method](https://img.shields.io/badge/Method-传统%20%2B%20SOTA-green)

---

## 目录

- [任务背景](#任务背景)
- [方案概览](#方案概览)
- [效果展示](#效果展示)
- [目录结构](#目录结构)
- [快速开始](#快速开始)
- [输出与评估](#输出与评估)
- [常见问题](#常见问题)
- [致谢与参考](#致谢与参考)

---

## 任务背景

文档图像几何校正是文档分析与 OCR 预处理的重要环节。真实拍摄的文档照片往往存在三类畸变：

- **透视变形（Perspective）**：手机/相机以非正对角度拍摄导致的梯形畸变
- **页面弯曲（Curved）**：摊开的书本沿书脊产生的曲面弯曲
- **倾斜（Rotate）**：纸张摆放不正带来的旋转

本实习基于 **GeoCorrection** 数据集（自建文档场景数据集，共 **200 张**，每张畸变文档图配有同名二值 **mask** 标注文档区域），设计并实现几何校正算法，将畸变文档恢复为平整的正常页面。

---

## 方案概览

采用「传统 baseline + 深度学习 SOTA」双方案对比：

| 方案 | 方法 | 原理 | 特点 |
|---|---|---|---|
| **A（baseline）** | OpenCV 透视校正 | mask 提取轮廓 → 四角点排序 → `getPerspectiveTransform` 单应变换 | 零训练、速度快，但无法矫正书页弯曲 |
| **C（主方法）** | UVDoc（SIGGRAPH Asia 2023） | 神经网络预测形变网格（backward mapping）→ `grid_sample` 全分辨率重映射 | 同时矫正透视+弯曲+倾斜，SOTA 效果 |

> mask 直接给出文档精确轮廓，省去分割步骤：方案 A 用其检测四角点，方案 C 用其裁剪文档 ROI。

---

## 效果展示

下图从 200 张结果中抽样展示三联对比（原图 | Baseline | UVDoc）：

![抽样对比总览](results/compare/_grid_sample.jpg)

### 关键量化结果（200 张均值）

**OCR 识别效果**（rapidocr，识别字符越多、置信度越高越好）：

| 指标 | 原图 | Baseline | UVDoc |
|---|---|---|---|
| 识别字符数 chars（↑） | 565.1 | 617.0 | **798.9** |
| 文本行数 lines（↑） | 37.6 | 42.0 | **47.0** |
| 平均置信度 conf（↑） | 0.7533 | 0.7606 | **0.7769** |

UVDoc 校正后识别字符数比原图提升 **41.4%**，明显优于 baseline，证明弯曲矫正对 OCR 前置处理的显著价值。

---

## 目录结构

```
├── baseline/
│   └── 01_baseline.py          # 方案A：传统透视校正
├── uvdoc/
│   ├── 02_uvdoc_infer.py       # 方案C：UVDoc GPU 推理
│   └── UVDoc/                  # UVDoc 官方代码 + best_model.pkl 预训练权重
├── eval/
│   ├── 03_visual_compare.py    # 三联对比图
│   ├── 04_line_regularity.py   # 文字行规整度量化
│   ├── 05_ocr_eval.py          # OCR 对比评估
│   └── 06_summary_stats.py     # 汇总统计
├── common/
│   └── imgio.py                # 中文路径图像读写工具
├── results/
│   ├── baseline/               # 方案A 校正结果
│   ├── uvdoc/                  # 方案C 校正结果
│   └── compare/                # 三联对比图 + 抽样总览
├── README.md                   # 本文件
├── 部署文档.md                  # 详细环境安装与运行说明
└── 总结文档.md                  # 完整实验结论与报告写作要点
```

---

## 快速开始

### 环境要求

- Python 3.10+（本项目使用 3.13）
- NVIDIA GPU（可选，UVDoc 推理建议使用；无 GPU 自动回退 CPU）
- 磁盘空间：约 5 GB（含 PyTorch GPU 版）

### 安装依赖

```bash
# 1) 创建并激活虚拟环境
python -m venv venv
venv\Scripts\activate        # Windows；Linux/macOS 用 source venv/bin/activate

# 2) 安装 PyTorch（GPU 版，CUDA 12.8；无 GPU 用 CPU 版）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 3) 安装其余依赖
pip install opencv-python numpy matplotlib tqdm rapidocr-onnxruntime
```

### 数据准备

解压 `GeoCorrection.zip` 到 `GeoCorrection/_extract/GeoCorrection/`，得到 `images/`（200 张 .jpg）和 `mask/`（200 张同名 .png）。

### 运行

```bash
python baseline/01_baseline.py    # ① 方案A：透视校正（~1 分钟）
python uvdoc/02_uvdoc_infer.py    # ② 方案C：UVDoc 推理（~2 分钟，GPU）
python eval/03_visual_compare.py  # ③ 三联对比图（~1 分钟）
python eval/04_line_regularity.py # ④ 文字行规整度（~1 分钟）
python eval/05_ocr_eval.py        # ⑤ OCR 评估（~60-90 分钟）
python eval/06_summary_stats.py   # ⑥ 汇总统计
```

> 详细的环境配置、UVDoc 模型获取、输出说明见 [部署文档.md](部署文档.md)。

---

## 输出与评估

| 路径 | 内容 |
|---|---|
| `results/baseline/` | 方案 A 校正结果（200 张） |
| `results/uvdoc/` | 方案 C（UVDoc）校正结果（200 张） |
| `results/compare/` | 每张三联对比图；`_grid_sample.jpg` 为抽样总览 |
| `eval/metric_line_regularity.csv` | 文字行规整度（curvature 越小越好，line_var 越大越好） |
| `eval/metric_ocr.csv` | OCR 字符数/行数/置信度（原图 vs baseline vs uvdoc） |
| `eval/summary_stats.csv` | 汇总统计表 |
| `eval/*_log.csv` | 各步骤运行日志 |

评估从三方面展开：**校正前后可视化对比**、**文字行规整程度量化**、**OCR 识别效果**。完整实验结论与报告写作要点见 [总结文档.md](总结文档.md)。

---

## 常见问题

- **`cv2.imread` 返回 `None`**：路径含中文或空格所致。本项目已在 `common/imgio.py` 用 `np.fromfile + imdecode` 统一处理。
- **`torch.cuda.is_available()` 为 `False`**：RTX 50 系显卡必须使用 cu128 wheel（torch ≥ 2.7）。
- **OCR 速度慢**：`eval/05_ocr_eval.py` 中的 `MAX_SIDE` 控制输入尺寸，可调小以加速（精度略降）。

---

## 致谢与参考

- 深度学习主方法基于 **UVDoc**：*Floor Verhoeven, Tanguy Magne, Olga Sorkine-Hornung. "UVDoc: Neural Grid-based Document Unwarping". SIGGRAPH Asia 2023.*
- 官方代码与预训练模型：<https://github.com/tanguymagne/UVDoc>
- OCR 评估使用 [RapidOCR](https://github.com/RapidAI/RapidOCR)（本地推理，无需联网）。
