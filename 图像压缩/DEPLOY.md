# 图像压缩实习作业 · 部署文档

## 1. 项目简介

本项目基于 **Kodak Lossless True Color Image Suite**（24 张 24bit 真彩色 RGB 图像，768×512 或 512×768），实现了两种图像压缩算法并完成对比分析：

| 算法 | 类型 | 说明 |
|------|------|------|
| 自研 JPEG 风格 DCT 编码器 | 传统变换编码（从零实现） | RGB→YCbCr → 8×8 分块 DCT → 量化 → Zigzag 扫描 → DC 差分 + AC 游程(RLE) + 规范哈夫曼熵编码 |
| AVIF | 先进编码标准（AV1 帧内） | 基于 libavif（pillow-avif-plugin），率失真性能显著优于 JPEG |

**核心结论**：在相同 PSNR（约 33 dB）下，AVIF 的码率（bpp≈0.62）比自研 JPEG（bpp≈1.03）**节省约 40%**；低码率段（约 26.5 dB）节省可达约 63%。

评价指标覆盖作业全部要求：压缩率、压缩后 PSNR、压缩后 SSIM、压缩/解压缩时间。

## 2. 环境要求

| 项 | 要求 |
|----|------|
| 操作系统 | Windows 10/11（本项目在 Windows 11 上验证） |
| Python | ≥ 3.10（本项目用 3.14.7 验证通过） |
| 硬件 | 无特殊要求，CPU 即可，无需 GPU |

> 项目代码为纯 Python + 第三方库，无自编译扩展，跨平台（Linux/macOS）也可运行，只需替换路径写法。

## 3. 依赖清单与安装

在项目根目录下执行：

```bash
pip install numpy scipy Pillow pillow-avif-plugin scikit-image matplotlib
```

各依赖用途：

| 包 | 用途 |
|----|------|
| `numpy` | 数组运算、图像数据表示 |
| `scipy` | 8×8 分块 DCT / IDCT（`scipy.fft.dctn`） |
| `Pillow` | 图像读写 |
| `pillow-avif-plugin` | 注册 AVIF 格式，提供 AV1 编解码 |
| `scikit-image` | 计算 SSIM（`skimage.metrics.structural_similarity`） |
| `matplotlib` | 绘制率失真曲线与对比图 |

> 如需固定版本（本项目验证环境）：`numpy 2.5.2`、`scipy 1.18.1`、`Pillow 12.3.0`、`pillow-avif-plugin 1.6.0`、`scikit-image 0.26.0`、`matplotlib 3.11.2`。

## 4. 数据集准备

Kodak 数据集需按以下目录结构放置（项目默认读取路径）：

```
项目根目录/
└── Kodak-Lossless-True-Color-Image-Suite-master/
    └── Kodak-Lossless-True-Color-Image-Suite-master/
        └── PhotoCD_PCD0992/
            ├── 01.png
            ├── 02.png
            ├── ...
            └── 24.png
```

共 24 张 PNG。若你的数据路径不同，请修改 `run_benchmark.py` 和 `plot.py` 顶部的 `DATA_DIR` 常量。

## 5. 项目结构

```
图像压缩/
├── Kodak-Lossless-True-Color-Image-Suite-master/   # 数据集（见第 4 节）
├── compressors/
│   ├── __init__.py
│   ├── jpeg_dct.py      # 自研 JPEG 编解码器（encode/decode 接口）
│   └── avif.py          # AVIF 编解码封装（encode/decode 接口）
├── eval.py              # 指标计算：bpp、压缩率、PSNR、SSIM、计时
├── run_benchmark.py     # 批量评测入口（24 张图 → results/*.csv）
├── plot.py              # 画 RD 曲线 + 压缩前后对比图
├── generate_summary.py  # 由结果生成 summary.md
├── summary.md           # 结果总结（供写实习报告引用）
└── results/             # 输出：metrics.csv / summary.csv / 各图片
```

两个压缩算法统一接口：

```python
encode(img: np.uint8[H, W, 3], quality=50) -> bytes   # 压缩
decode(data: bytes) -> np.uint8[H, W, 3]              # 解压缩
```

## 6. 运行步骤

在项目根目录下，依次执行三个命令：

### 6.1 批量评测

```bash
python run_benchmark.py
```

- 对 24 张图，跑「自研 JPEG（Q=10/30/50/75/90）+ AVIF（Q=10/30/50/70/90）」共 10 个配置。
- 输出 `results/metrics.csv`（每张图每配置一行明细）与 `results/summary.csv`（各配置 24 张均值）。
- 耗时约 2 分钟。

> 只想跑自研 JPEG（跳过 AVIF）时：`python run_benchmark.py --jpeg-only`

### 6.2 绘图

```bash
python plot.py
```

- 输出 `results/rd_psnr.png`（PSNR-bpp 率失真曲线）、`results/rd_ssim.png`（SSIM-bpp 曲线）。
- 输出 `results/compare_01.png`、`compare_05.png`、`compare_23.png`（原图 / 自研JPEG / AVIF 三联对比，含局部放大）。

### 6.3 生成总结文档

```bash
python generate_summary.py
```

- 读取 `results/metrics.csv`，生成 `summary.md`（结果汇总表 + 率失真结论 + 可引用话术）。

## 7. 预期结果

运行完成后，`results/summary.csv` 应接近下表（24 张图均值，数值允许有 ±2% 的浮点/环境差异）：

| 方案 | avg_bpp | avg_psnr | avg_ssim | avg_ratio | 编码(ms) | 解码(ms) |
|------|---------|----------|----------|-----------|----------|----------|
| jpeg-dct-q10 | 0.311 | 26.94 | 0.746 | 83.9 | 178 | 92 |
| jpeg-dct-q50 | 1.028 | 32.69 | 0.905 | 25.3 | 231 | 231 |
| jpeg-dct-q90 | 2.779 | 39.21 | 0.969 | 9.1 | 369 | 569 |
| avif-q10 | 0.115 | 26.45 | 0.707 | 221.8 | 20 | 10 |
| avif-q50 | 0.622 | 33.07 | 0.911 | 42.1 | 29 | 11 |
| avif-q90 | 2.491 | 41.09 | 0.981 | 10.2 | 45 | 14 |

关键规律：

1. 随质量因子 Q 增大，两种算法的 bpp 与 PSNR/SSIM 均单调上升，符合率失真基本规律。
2. AVIF 在相同 PSNR 下码率显著低于自研 JPEG（约 40%~63% 节省）。
3. AVIF 编解码由底层 C 库（libavif）完成，速度快于纯 Python 实现的自研 DCT（速度差异主要来自实现语言，报告撰写时应如实说明）。

## 8. 常见问题（FAQ）

**Q1：报 `ModuleNotFoundError: No module named 'pillow_avif'`**
未安装 AVIF 插件。执行 `pip install pillow-avif-plugin`。

**Q2：报 `ModuleNotFoundError: No module named 'skimage'`**
未安装 scikit-image。执行 `pip install scikit-image`。

**Q3：报 `FileNotFoundError` 找不到 `01.png`**
数据集路径不对。确认数据在 `Kodak-Lossless-True-Color-Image-Suite-master/Kodak-Lossless-True-Color-Image-Suite-master/PhotoCD_PCD0992/` 下，或修改 `DATA_DIR` 常量。

**Q4：生成的图里中文显示成方块（□□□）**
matplotlib 默认字体不含中文。`plot.py` 已配置 `Microsoft YaHei`，若仍异常，请确认系统已安装该字体；或改用 `SimHei`（黑体）。

**Q5：跑得很慢 / 卡住**
正常现象：24 张图 × 10 配置共 240 次编解码 + SSIM 计算，约 1~2 分钟。若超过 5 分钟，可先 `--jpeg-only` 验证自研算法，再单独排查 AVIF。

**Q6：`plot.py` / `generate_summary.py` 文件头注释写「在 venv_cai 运行」**
这是早期版本残留的过时注释，可忽略。当前所有脚本均在主环境直接 `python xxx.py` 运行，本部署文档的命令为准。

## 9. 复现校验

按以下顺序核对，即可确认复现成功：

1. `results/metrics.csv` 存在，且包含 `10 配置 × 24 图 = 240` 行数据（+ 表头）。
2. `results/rd_psnr.png` 中 AVIF 曲线整体位于自研 JPEG 曲线左上方（相同 PSNR 下 bpp 更小）。
3. `results/compare_*.png` 中三张图（原图 / 自研JPEG / AVIF）肉眼质量接近，局部放大区能看出细节差异。
4. `summary.md` 生成成功，结果表与 `results/summary.csv` 数值一致。
