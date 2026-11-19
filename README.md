# hw3 · 图像压缩

数字图像处理实习三。在 Kodak 无损真彩图库（24 张 24 bit RGB，768×512 或 512×768）上实现两种压缩方案并做率失真对比。

| 方案 | 类型 | 说明 |
|---|---|---|
| 自研 JPEG 风格 DCT 编码器 | 传统变换编码，从零实现 | RGB→YCbCr → 8×8 分块 DCT → 量化 → Zigzag → DC 差分 + AC 游程 → 规范哈夫曼熵编码 |
| AVIF | 先进编码标准 | 基于 libavif（`pillow-avif-plugin`） |

**结论**：相同 PSNR（约 33 dB）下，AVIF 码率 bpp≈0.62，自研 JPEG bpp≈1.03，节省约 40%；低码率段（约 26.5 dB）节省可达约 63%。指标覆盖压缩率、PSNR、SSIM、压缩/解压耗时。

## 运行

```bash
pip install -r requirements.txt
python run_benchmark.py      # 跑全部 24 张图，输出 results/metrics.csv
python plot.py               # 率失真曲线与对比图
python generate_summary.py   # 汇总为 results/summary.csv
```

结果表已在仓库内：`results/metrics.csv`（逐图逐 bpp 点）、`results/summary.csv`（汇总）。

## 数据准备（必须，数据未上传）

需要 Kodak Lossless True Color Image Suite 的 24 张 PNG（`kodim01`–`kodim24`）。图片本体未上传，从官方或镜像获取后按脚本约定放入数据目录即可。

## 环境

纯 Python + 第三方库，无自编译扩展，CPU 即可。Python ≥ 3.10（实测 3.14.7），Windows 11 验证，Linux/macOS 替换路径写法后可运行。

## 目录

```
├── compressors/      自研 DCT 编码器与 AVIF 封装
├── run_benchmark.py  基准测试入口
├── eval.py           单次评价
├── plot.py           绘图
├── generate_summary.py
├── results/          结果表（metrics.csv / summary.csv）
├── 提交包/            提交时的文档集合（总结文档 / 部署文档）
├── DEPLOY.md         部署文档
├── summary.md        结论摘要
└── 历史对话内容.md
```

## 本分支未包含

- Kodak 图片本体（24 张 PNG，仅数据，非代码）
- 与父目录逐字节重复的 `提交包/{compressors,results,eval.py,...}` 副本（仅保留父目录没有的三份文档）

完整说明见 `DEPLOY.md` 与 `提交包/部署文档.md`。
