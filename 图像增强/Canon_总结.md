# 图像增强实习 · Canon 子集 · 方法与结果总结

> 本文件为 Canon 子集（路线 B，轻量残差 UNet）的方法、训练参数与结果结论汇总，可直接引用进实习报告。与 Sony 子集采用完全相同的模型与训练配置，仅数据集与输出目录不同。

## 一、任务与数据集

- 任务：极暗 RAW → 正常曝光 sRGB 的图像增强（低光增强 + 强去噪 + 曝光/色彩恢复）
- 数据集：SIED（See-in-the-Extremely-Dark，ICCV 2025，课程内称 BIED）的 **Canon_0.0001** 子集
- 光照范围：0.0001–0.001 lux（比 Sony 子集 0.001–0.01 lux 更极端，暗约 10 倍）
- 规模：1000 对「低光 RAW ↔ GT sRGB」，随机划分训练 900 / 验证 100，测试全量 1000
- RAW：`.raw` 二进制，uint16，3840×2160，黑电平 512 / 白电平 16383（实测确认，与 Sony 一致）

## 二、方法（路线 B：监督式轻量 UNet）

- 预处理：黑电平扣除 → 归一化 → RGGB 四通道打包（H/2 × W/2 × 4）
- 网络：残差 UNet，4 层下采样，InstanceNorm，输入 4ch packed RAW → 输出 12ch → `PixelShuffle` → 3ch RGB（分辨率 2×，对齐 GT）
- 损失：Charbonnier L1（权重 1.0）+ SSIM（权重 0.15）
- 改进点：混合精度 AMP、cosine 预热学习率、梯度裁剪、按验证 PSNR 选最优模型、任意尺寸推理

## 三、训练参数

| 参数 | 值 |
|---|---|
| 模型通道数 base_channels | 32 |
| 下采样层数 num_levels | 4 |
| 归一化 norm | Instance |
| 训练轮数 epochs | 50 |
| patch 尺寸 | 512 |
| batch 大小 | 8 |
| 学习率 lr | 1e-4 |
| 学习率策略 | cosine + 2 epoch 预热 |
| 优化器 | AdamW |
| 混合精度 AMP | 开启 |
| 梯度裁剪 grad_clip | 1.0 |
| 随机种子 seed | 42 |
| 相机黑/白电平 | 512 / 16383 |

## 四、结果（1000 张测试图全量均值）

| 指标 | 路线 B（UNet） | 传统基线（CLAHE） |
|---|---|---|
| **PSNR ↑** | **18.84 dB** | 9.29 dB |
| **SSIM ↑** | **0.695** | 0.489 |
| **LPIPS ↓** | **0.554** | 0.781 |

## 五、结论

- 基于 RAW 域的监督式 UNet 能有效完成 0.0001 lux 极暗场景的图像增强，输出由几乎全黑恢复为清晰可辨的彩色图像。
- 相比传统方法（CLAHE），深度方法在 PSNR 上**领先 +9.55 dB**、SSIM 与 LPIPS 均明显更优。
- Canon 子集（0.0001 lux）比 Sony 子集（0.001 lux）光照暗约 10 倍、信噪比更低，因此 PSNR（18.84）略低于 Sony（20.45），符合任务难度差异。

## 六、复现命令

```bash
python scripts/make_filelist.py --dataset Canon   # 生成配对清单
python src/train.py --config configs/config_canon.yaml      # 训练
python src/evaluate.py --config configs/config_canon.yaml   # 全量评估
python src/visualize.py --config configs/config_canon.yaml  # 可视化对比图
```

## 七、产物清单

| 文件 | 说明 |
|---|---|
| `configs/config_canon.yaml` | Canon 训练参数 |
| `outputs/checkpoints_canon/best.pt` | Canon 最优模型（base=32） |
| `outputs/results/Canon/results.csv` | 逐张指标 + 均值 |
| `outputs/results/Canon/baseline.csv` | 传统基线指标 |
| `outputs/results/Canon/comparison.png` | 输入/路线B/基线/GT 对比图 |
| `outputs/logs_canon/train_log.csv` | 训练曲线 |
