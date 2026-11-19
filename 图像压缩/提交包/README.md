# 数字图像处理实习三 · 图像压缩

> 基于 Kodak 数据集，实现「自研 JPEG 风格 DCT 编码器」与「AVIF」两种有损压缩算法，并从压缩率、PSNR、SSIM、编解码耗时四个维度对比分析。

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![Task](https://img.shields.io/badge/Task-Image%20Compression-orange) ![Method](https://img.shields.io/badge/Method-DCT%20%2B%20AVIF-green)

---

## 目录

- [任务与数据](#任务与数据)
- [方案概览](#方案概览)
- [目录结构](#目录结构)
- [环境与依赖](#环境与依赖)
- [数据准备（必读）](#数据准备必读)
- [运行步骤](#运行步骤)
- [结果摘要](#结果摘要)
- [指标口径](#指标口径)
- [复现校验](#复现校验)
- [文档导航](#文档导航)

---

## 任务与数据

**任务**：完成图像读取、压缩、解压缩，并对压缩前后的图像质量与压缩效果做对比分析。

**数据**：Kodak Lossless True Color Image Suite，24 张 24bit 真彩色 RGB 图像，尺寸为 768×512 或 512×768。

**评价指标**：压缩率、压缩后 PSNR、压缩后 SSIM、压缩/解压缩时间。

> 本包为提交版，**不含数据集图片**（数据集由课程统一提供），需按下文第 [数据准备](#数据准备必读) 节放置。

---

## 方案概览

| 方案 | 类型 | 实现要点 |
|---|---|---|
| **自研 JPEG 风格 DCT 编码器** | 传统变换编码，从零实现 | RGB→YCbCr → 8×8 分块 DCT → JPEG 标准量化表按质量因子缩放 → Zigzag 扫描 → DC 差分 + AC 游程(RLE) → 规范哈夫曼熵编码 |
| **AVIF** | 先进编码标准（AV1 帧内） | 经 libavif（pillow-avif-plugin）调用，含更精细的块划分、多种变换核与环路滤波 |

两个编码器统一接口，因此评测脚本可完全复用：

```python
encode(img: np.uint8[H, W, 3], quality=50) -> bytes   # 压缩
decode(data: bytes) -> np.uint8[H, W, 3]              # 解压缩
```

有损压缩的关键在于**量化**：DCT 本身正交可逆，量化是整条链路上唯一的信息损失点。自研编码器完整实现了这条「变换 + 量化 + 熵编码」链路，未使用任何深度学习方法。

---

## 目录结构

```
提交包/
├── README.md               # 本文件
├── 部署文档.md              # 环境安装、运行步骤、输出说明、FAQ
├── 总结文档.md              # 实验结果汇总与报告写作要点
├── compressors/
│   ├── __init__.py
│   ├── jpeg_dct.py         # 自研 JPEG 编解码器（核心，含熵编码实现）
│   └── avif.py             # AVIF 编解码封装
├── eval.py                 # 指标计算：bpp、压缩率、PSNR、SSIM、计时
├── run_benchmark.py        # 批量评测入口（24 张图 × 10 配置 → CSV）
├── plot.py                 # 绘制 RD 曲线与三联对比图
├── generate_summary.py     # 由 metrics.csv 生成总结表格
└── results/
    ├── metrics.csv         # 240 行明细（10 配置 × 24 张图）
    └── summary.csv         # 10 个配置的 24 张均值
```

---

## 环境与依赖

纯 Python，无需 GPU、无需编译扩展。依赖安装命令、各包用途与精确版本号见 **[部署文档.md](部署文档.md) 第 2–3 节**。

一句话版本：

```bash
pip install numpy scipy Pillow pillow-avif-plugin scikit-image matplotlib
```

---

## 数据准备（必读）

`run_benchmark.py` 与 `plot.py` 中的 `DATA_DIR` 均为**相对路径**：

```python
DATA_DIR = "Kodak-Lossless-True-Color-Image-Suite-master/Kodak-Lossless-True-Color-Image-Suite-master/PhotoCD_PCD0992"
```

因此需把 24 张 PNG 原样放到**本包根目录**下的这一相对位置，形成如下结构（`PhotoCD_PCD0992` 目录里应有 `01.png` ~ `24.png`）：

```
提交包/
├── README.md
├── run_benchmark.py
├── ...
└── Kodak-Lossless-True-Color-Image-Suite-master/
    └── Kodak-Lossless-True-Color-Image-Suite-master/
        └── PhotoCD_PCD0992/
            ├── 01.png
            ├── 02.png
            ├── ...
            └── 24.png
```

两种处理方式，任选其一：

1. **推荐**：按上述结构放置数据，源码无需任何改动；
2. 若数据放在别处，修改 `run_benchmark.py` 第 21 行与 `plot.py` 第 26 行的 `DATA_DIR` 为你的实际路径（两处都要改）。

> 未放数据会报 `FileNotFoundError`。另外 `plot.py` 需要读取原图来绘制三联对比图，因此**即使只想看曲线，也请先完成本节**。

---

## 运行步骤

在本包根目录下依次执行：

```bash
python run_benchmark.py     # ① 批量评测 → results/metrics.csv、results/summary.csv（约 2 分钟）
python plot.py              # ② RD 曲线 + 三联对比图 → results/*.png（约 1 分钟）
python generate_summary.py  # ③ 由 metrics.csv 重新生成总结表格
```

只看自研算法、不跑 AVIF 时：`python run_benchmark.py --jpeg-only`

- ①③ 的输出 `results/*.csv` 已随包提供，用于校验复现结果是否一致。
- ② 输出的 5 张图（`rd_psnr.png`、`rd_ssim.png`、`compare_01/05/23.png`）未随包附带，运行后即生成；`总结文档.md` 中对这些图的引用即指此。

---

## 结果摘要

24 张图均值（与 `results/summary.csv` 一致，允许 ±2% 的环境浮点差异）：

| 方案 | bpp | PSNR(dB) | SSIM | 压缩率 | 编码(ms) | 解码(ms) |
|---|---|---|---|---|---|---|
| jpeg-dct-q10 | 0.311 | 26.94 | 0.7461 | 83.9 | 178.2 | 91.9 |
| jpeg-dct-q30 | 0.724 | 30.91 | 0.8691 | 36.1 | 211.2 | 171.8 |
| jpeg-dct-q50 | 1.028 | 32.69 | 0.9050 | 25.3 | 231.3 | 230.8 |
| jpeg-dct-q75 | 1.595 | 35.22 | 0.9388 | 16.2 | 277.5 | 342.8 |
| jpeg-dct-q90 | 2.779 | 39.21 | 0.9686 | 9.2 | 368.5 | 568.7 |
| avif-q10 | 0.115 | 26.45 | 0.7067 | 221.8 | 20.3 | 9.6 |
| avif-q30 | 0.261 | 29.29 | 0.8173 | 100.1 | 22.8 | 10.0 |
| avif-q50 | 0.622 | 33.07 | 0.9107 | 42.1 | 28.9 | 10.8 |
| avif-q70 | 1.263 | 37.20 | 0.9560 | 20.9 | 35.3 | 12.1 |
| avif-q90 | 2.491 | 41.09 | 0.9808 | 10.2 | 45.3 | 13.7 |

**率失真结论**

- 随质量因子 Q 增大，两种算法的 bpp 与 PSNR/SSIM 均单调上升，符合率失真基本规律。
- PSNR 相当（AVIF 33.1 dB vs 自研 JPEG 32.7 dB）时，AVIF bpp=0.622 vs JPEG bpp=1.028，**节省约 40% 码率**。
- PSNR 相当（AVIF 26.4 dB vs 自研 JPEG 26.9 dB）时，AVIF bpp=0.115 vs JPEG bpp=0.311，**节省约 63% 码率**。
- AVIF 编解码更快，是因为其底层为 C 库（libavif），而自研编码器为纯 Python 实现；这一差异主要来自实现语言与工程优化，而非算法复杂度本身。

---

## 指标口径

| 指标 | 定义 |
|---|---|
| `bpp` | 压缩后总比特数 ÷ 像素数（`H × W`），即每像素比特数 |
| `compression_ratio` | 原始像素量 `H × W × 3` 字节 ÷ 压缩后字节数 |
| `psnr` | `10 · log10(255² / MSE)`，MSE 对 RGB 三通道取平均 |
| `ssim` | `skimage.metrics.structural_similarity`，对 RGB 三通道取平均 |
| `t_enc` / `t_dec` | `time.perf_counter` 单次计时，取 24 张均值 |

> 压缩率分母采用原始像素量 `H×W×3`（而非 PNG 文件体积），避免与无损 PNG 的存储开销混淆。

---

## 复现校验

按以下 4 条核对，即可确认复现成功：

1. `results/metrics.csv` 含 **240 行**数据（10 配置 × 24 图）+ 1 行表头，与本包提供的文件行数一致。
2. `results/summary.csv` 数值与上方「结果摘要」表一致（允许 ±2% 浮动）。
3. `results/rd_psnr.png` 中 AVIF 曲线整体位于自研 JPEG 曲线**左上方**（相同 PSNR 下 bpp 更小）。
4. `results/compare_01.png` 等三联图（原图 / 自研 JPEG / AVIF）肉眼质量接近，局部放大区可看出细节差异。

---

## 文档导航

| 文档 | 用途 |
|---|---|
| [部署文档.md](部署文档.md) | 环境要求、依赖清单与版本、数据准备、输出说明、6 条 FAQ、复现校验 |
| [总结文档.md](总结文档.md) | 实现方案、结果汇总、率失真结论、可直接引用的结论话术、报告写作要点 |

---

## 说明

本项目全部算法均为经典信号处理与编码方法：自研 DCT 编码器基于 `numpy` + `scipy.fft` 实现，AVIF 基于 AV1 帧内编码（libavif）。**未使用任何神经网络或深度学习模型**。

数据集：Kodak Lossless True Color Image Suite（Eastman Kodak Company 公开发布，供图像压缩研究使用）。
