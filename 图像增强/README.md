# 数字图像处理实习 · 图像增强（极暗 RAW → 正常曝光 sRGB）

基于 **SIED 数据集**（ICCV 2025《Learning to See in the Extremely Dark》，课程内也称 **BIED**），
完成从极暗 RAW 图像到正常曝光 sRGB 图像的增强任务。本项目采用**路线 B：轻量 UNet 监督学习**为主，
并附带**路线 A：传统方法基线**（Gamma / CLAHE / Retinex / 直方图拉伸）用于报告对比。

- 数据集：`Sony_0.001`（0.001–0.01 lux）与 `Canon_0.0001`（0.0001–0.001 lux）两个子集，各 1000 对低光 RAW ↔ GT
- 评价指标：**PSNR / SSIM / LPIPS**（全量 1000 张）
- 任务本质：低光增强 + 强去噪 + 曝光/色彩恢复（三合一）

---

## 一、目录结构

```
图像增强/
├── README.md                  # 本文档
├── requirements.txt           # Python 依赖
├── .gitignore
├── configs/
│   └── config.yaml            # 全局配置（数据 / 模型 / 训练 / 评估）
├── data/                      # 数据目录（下载后放入，见 data/README.txt）
│   ├── Sony/                  # Sony 子集
│   └── Canon/                 # Canon 子集
├── src/
│   ├── config.py              # 配置加载（默认值 + YAML + 命令行覆盖）
│   ├── data/
│   │   ├── raw_utils.py       # 黑电平扣除 / RGGB 打包 / demosaic / ISP
│   │   ├── transforms.py      # 随机裁剪 / 翻转 / 旋转
│   │   └── dataset.py         # 配对数据集（训练增强 / 评估整图）
│   ├── models/
│   │   ├── unet.py            # 残差 UNet（4ch→12ch→pixel_shuffle→3ch）
│   │   └── losses.py          # Charbonnier L1 + SSIM（可选感知损失）
│   ├── baselines/
│   │   └── traditional.py     # 路线 A 传统基线
│   ├── utils/
│   │   ├── metrics.py         # PSNR / SSIM / LPIPS
│   │   └── logger.py          # 日志 CSV / 检查点 / 设备
│   ├── train.py               # 训练入口
│   ├── evaluate.py            # 全量评估入口
│   ├── inference.py           # 单张 / 批量推理
│   └── visualize.py           # 对比图可视化
├── scripts/
│   ├── check_env.py           # 环境自检（CUDA / GPU / 依赖）
│   └── make_filelist.py       # 扫描数据目录生成 filelist
└── outputs/                   # 训练产物（checkpoint / 日志 / 结果图）
```

---

## 二、环境依赖

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 安装匹配 CUDA 版本的 PyTorch（重要）

`requirements.txt` 中的 `torch` 默认装 CPU 版或最新 CUDA 版，请按你显卡驱动对应的 CUDA 版本安装：

```bash
# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

安装后运行自检：

```bash
python scripts/check_env.py
```

应看到 `CUDA 可用: True` 和你的 GPU 型号/显存。若为 False，检查驱动与 torch 版本是否匹配。

---

## 三、各模块功能说明

| 模块 | 功能 |
|---|---|
| `config.py` | 加载 YAML 配置，支持 `--set key=value` 覆盖，提供默认值合并 |
| `data/raw_utils.py` | RAW 读取（`.raw` 二进制 + `rawpy`）、黑电平扣除、RGGB 四通道打包、简单 demosaic、白平衡、gamma |
| `data/transforms.py` | 训练数据增强：随机裁剪 patch、随机水平/垂直翻转、随机 90° 旋转 |
| `data/dataset.py` | `PairedDataset`（训练，带增强）、`EvalDataset`（整图评估），自动对齐 GT 分辨率 |
| `models/unet.py` | 残差 UNet：输入 4ch packed RAW，输出 12ch 经 `PixelShuffle` 得到 3ch 全分辨率 RGB |
| `models/losses.py` | Charbonnier L1、SSIM 损失、VGG 感知损失（可选） |
| `baselines/traditional.py` | 传统基线：Gamma / CLAHE / Retinex(MSR) / 直方图拉伸 |
| `utils/metrics.py` | PSNR / SSIM / LPIPS 统一封装（输入 numpy (H,W,3) ∈ [0,1]） |
| `utils/logger.py` | CSV 训练日志、checkpoint 保存/加载、设备自动检测 |
| `train.py` | 训练：AMP 混合精度、梯度裁剪、cosine 预热学习率、按验证 PSNR 存最优模型、断点续训 |
| `evaluate.py` | 全量评估：逐张计算指标 → CSV + 均值（可选同时跑传统基线） |
| `inference.py` | 对单张或目录 RAW 做增强，输出 sRGB PNG |
| `visualize.py` | 生成「输入|路线B|基线|GT」对比图 |
| `scripts/make_filelist.py` | 扫描数据目录，自动配对 RAW↔GT，生成 train/val/test filelist |
| `scripts/check_env.py` | 环境自检 |

---

## 四、运行方式

### 第 1 步：接入数据

下载好数据集后放入 `data/`（见 `data/README.txt`），并确认 `configs/config.yaml` 中：

```yaml
data:
  root: "data"
  dataset: "Sony"          # 先做 Sony，再做 Canon 改成 "Canon"
  raw_subdir: ""           # 留空 = 递归扫描整个子集目录（兼容多层嵌套）
  gt_subdir: ""            # 同上
  raw_ext: "raw"           # 或 arw/cr2/dng
  gt_ext: "jpg"            # 或 png
  black_level: 512         # Sony 参数；Canon 需确认后修改
  white_level: 16383
```

### 第 2 步：生成 filelist

```bash
python scripts/make_filelist.py
```

会在 `data/Sony/` 下生成 `train.txt`、`val.txt`、`test.txt`。

### 第 3 步：训练

```bash
python src/train.py
```

常用覆盖参数：

```bash
# 显存不足时（多个 --set 需逐个写出）
python src/train.py --set train.patch_size=256 --set train.batch_size=4
# 换 Canon 子集
python src/train.py --set data.dataset=Canon
# 断点续训
python src/train.py --set train.resume=outputs/checkpoints/last.pt
```

训练产物：`outputs/checkpoints/best.pt`（最优）、`last.pt`（最新）、`outputs/logs/train_log.csv`。

### 第 4 步：全量评估

```bash
python src/evaluate.py
```

输出：`outputs/results/Sony/results.csv`（逐张 + 均值）、`baseline.csv`（传统基线）、
`images/`（增强结果图）。终端会打印模型与基线的均值对比。

### 第 5 步：可视化对比图

```bash
python src/visualize.py --set eval.num_samples=4
```

输出 `outputs/results/Sony/comparison.png`，可直接放入报告。

### 单张/批量推理（不评估）

```bash
python src/inference.py data/Sony/short/xxx.raw --out outputs/enhanced
```

---

## 五、后续接入 BIED 数据集的注意事项

1. **BIED 即 SIED**：课程网盘与老师仓库 README 中的 "BIED" 就是 SIED 数据集的重新打包版，
   数据配对关系（低光 RAW ↔ GT）一致，本项目直接适用。
2. **目录结构**：`make_filelist.py` 会**递归扫描** `data/Sony`、`data/Canon` 下的全部文件并按文件名主干配对，
   因此不要求精确目录名。推荐结构为 `data/Sony/Sony_0.001/{0.001, gt}`、`data/Canon/Canon_0.0001/{0.0001, gt}`
   （`0.001`/`0.0001` 放低光 RAW，`gt` 放 GT 参考图）。`raw_subdir`/`gt_subdir` 留空即递归扫描；
   也可指定子目录名来缩小扫描范围。
3. **文件名配对**：RAW 与 GT 需**文件名主干相同**（如 `0001.raw` ↔ `0001.jpg`），
   脚本会自动剥离 `_gt` / `_ref` 等后缀。配对失败的文件会在脚本输出中列出，请核对。
4. **相机参数**：`black_level=512`、`white_level=16383` 是 Sony 14bit 的参数。
   Canon 子集的参数可能不同，**下载后务必确认并修改 `config.yaml`**，否则增强结果会偏色/过曝。
   若 RAW 是 `.arw/.cr2` 等格式，`rawpy` 会自动读取真实黑电平/白电平，无需手动设置。
5. **分辨率**：代码按 `raw_width=3840 / raw_height=2160` 读取 `.raw` 二进制；
   若实际尺寸不同需修改。GT 分辨率与模型输出不一致时会自动缩放对齐，无需担心。
6. **显存不足**：默认 `patch_size=512, batch_size=8`（适合 ≥8GB 显存）。若 OOM，
   调小 `train.patch_size`（256）与 `train.batch_size`（4），或关闭 `train.amp`。
7. **训练时长**：1000 张、patch 512、50 epoch 在 8GB 显卡上约数小时；先用小 epoch 数
   （如 `--set train.epochs=10`）验证流程跑通，再放开训练。

---

## 六、路线说明（对应报告）

- **路线 A（传统基线）**：`baselines/traditional.py`，用于对比，突出深度学习方法的优势。
- **路线 B（本项目主线）**：轻量残差 UNet，RAW 域直接学习 RAW→sRGB 映射。
  相比朴素实现，做了以下改进：Charbonnier+SSIM 多损失、InstanceNorm、残差块、
  AMP 混合精度、cosine 预热学习率、按验证 PSNR 选最优模型、任意尺寸推理（反射填充）。
- **路线 C（可选对比）**：官方扩散模型仓库 https://github.com/JianghaiSCU/SIED ，
  可用其预训练模型跑 `inference.py` 做 SOTA 对比（报告中标为基线即可）。
