# Canon 极暗 RAW 图像增强

在 `D:\BaiduNetdiskDownload\BIED` 上进行只读实验；项目的缓存、模型、日志与结果只写入本目录。

## 项目状态

已完成训练、消融、验证集选模、1000 张最终增强图和传统基线的出图评价、全量核验与报告。`artifacts/pipeline_status.json` 为 `all/completed`，实际权重的独立推理与分块一致性检查也已通过。

选定模型为 adaptive 残差 U-Net，第 24000 步，使用 EMA 权重。下表来自最终保存并回读的 3840×2160 RGB PNG：

| 统计范围 | 图像数 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|---:|
| 全量课程统计（包含训练图） | 1000 | 23.7416 | 0.7940 | 0.3964 |
| 固定验证集 | 100 | 21.3734 | 0.7909 | 0.4153 |
| 按自动场景组隔离的测试集 | 100 | 25.3181 | 0.8065 | 0.3728 |

全量与测试集均达到第一阶段目标（22 dB / 0.78 / 0.40），尚未同时达到较高目标（24 dB / 0.82 / 0.35）。测试集比验证集得分高，反映场景分布差异；不能将该结果外推为所有新场景的保证。自动分组并非官方场景标签。

训练和阶段验证合计约 1.38 小时；最终 1000 张出图与评价约 19.91 分钟，平均纯推理 0.246 秒/张。部分图片仍有曝光偏差、残余彩噪和细节变软，详见报告的低分样例。

旧项目仅作对照，不修改或复用其训练权重。独立测试图像不参与训练、校准或模型选择。

## 评价约定

- 3840×2160、RGB、8-bit 无损 PNG。
- PSNR/SSIM 在保存并重新读取的 PNG 上计算，RGB，`data_range=1`。
- SSIM：7×7 均匀窗口，样本协方差，与旧项目的 scikit-image 默认实现保持一致。
- LPIPS：AlexNet v0.1，原分辨率，RGB 映射至 [-1,1]。
- 逐张计算再平均；全量 1000 张统计与场景隔离测试集统计分别报告。
- 增强接口不接受 GT；GT 仅用于训练、训练集相机色彩标定和单独评价。

## 运行

```powershell
python -m canon_enhance.pipeline
```

完整流程为：基础测试 → 数据准备 → 共享主训练 → 色度权重消融与 EMA → 强噪声自适应候选 → 验证集选模 → 必要时有限感知损失实验 → 1000 张 PNG 与指标 → 传统基线 → 核验 → 报告。单阶段及断点续训命令见 `部署文档.md`。

```powershell
python -m canon_enhance.status
```

## 项目目录与文件说明

```text
.
├── .gitignore                         # 忽略缓存、权重、结果图等大型或可再生成文件
├── README.md                          # 项目结果概览、指标口径、运行入口和目录说明
├── 部署文档.md                        # 环境安装、训练、推理、评分和核验的详细命令
├── WORK_LOG.md                        # 实施过程、实验选择和阶段结果记录
├── requirements.txt                   # Python 依赖包列表
├── requirements-lock.txt              # 复现实验使用的固定依赖版本
├── configs/
│   └── canon.json                     # 数据路径、RAW 尺寸、模型、训练及评价参数
├── tests/
│   └── test_core.py                   # RAW 处理、模型、指标、推理和路径保护等测试
├── canon_enhance/
│   ├── __init__.py                    # Python 包标记
│   ├── common.py                      # 配置、路径保护、清单指纹、保存和通用工具
│   ├── data.py                        # RAW/JPEG 读取、Bayer 打包、上下文与训练数据集
│   ├── prepare.py                     # 数据审计、缓存、场景分组、划分和颜色标定
│   ├── model.py                       # RAW 到彩色 sRGB 的残差 U-Net
│   ├── ema.py                         # 指数移动平均权重的更新与恢复
│   ├── train.py                       # 网络训练、验证、检查点保存和断点续训
│   ├── inference.py                   # 单张 RAW 推理接口及命令行入口
│   ├── batch.py                       # 对 RAW 目录批量增强，不需要 GT
│   ├── baseline.py                    # 传统 RAW 去噪和训练集颜色标定基线
│   ├── metrics.py                     # PSNR、SSIM、LPIPS 与颜色诊断指标
│   ├── evaluate.py                    # 生成 PNG、回读评分并汇总各数据划分
│   ├── score.py                       # 对已有 PNG 单独评分，无需加载增强模型
│   ├── select.py                      # 根据固定验证集结果比较并选择模型
│   ├── verify.py                      # 核验输出数量、尺寸、指标及数据划分
│   ├── acceptance.py                  # 检查交付权重的独立推理和分块一致性
│   ├── report.py                      # 汇总指标、对比图和实习报告
│   ├── pipeline.py                    # 按顺序运行测试、训练、评价、核验和报告
│   └── status.py                      # 查看训练和流水线阶段状态
└── artifacts/                         # 实验生成物；缓存和结果无需随源码重复提交
    ├── pipeline_status.json           # 整条自动流水线的完成状态
    ├── environment.json               # 本机 Python、PyTorch 和 GPU 环境信息
    ├── selection.json                 # 候选模型验证指标与最终选模记录
    ├── old_reference.json             # 旧项目对照结果的来源记录
    ├── pretrained_check.json          # 外部预训练模型核验记录
    ├── ssim_validation.json            # GPU SSIM 与参考实现的对照结果
    ├── cache/gt/{1..1000}.npy          # artifacts/cache/gt 下由 GT 图片生成的训练读取缓存，可重新生成
    ├── checkpoints/                   # 训练候选权重；selected.pt 是最终推理权重
    ├── data/                           # 审计、清单、划分、上下文缓存和分组预览
    ├── logs/                           # 训练曲线、验证预览、流水线日志和评分记录
    ├── previews/                       # 开发阶段及低分样例预览图
    ├── reports/                        # 实习报告、对比图、局部图和验证曲线
    ├── results/
    │   ├── final/                      # 网络模型的 1000 张预测图和评价结果
    │   └── baseline/                   # 传统基线的预测图和评价结果
    ├── smoke/                          # 独立评分入口的少量样例和复核结果
    └── test_tmp/                       # 自动测试运行时使用的临时目录
```

### `artifacts/` 常用文件

- `artifacts/data/manifest.json`：1000 对文件的配对关系、训练/验证/测试划分、分组、RAW 归一化和训练集颜色标定。`train.txt`、`val.txt`、`test.txt` 是各划分编号，`all.txt` 是全量编号；`audit.json` 记录数据检查，`contexts.npy` 是训练上下文缓存，`contact_*.jpg` 用于检查分组或划分。
- `checkpoints/selected.pt`：最终选定的独立推理权重。各实验子目录中的 `best_psnr.pt`、`best_lpips.pt` 和 `last.pt` 分别保存对应最佳指标模型及最近训练状态；`pilot/` 是早期流程诊断权重，不用于最终结果，`prior_blur_*/` 是验证过但未选中的平滑参数对照。
- `artifacts/results/{final,baseline}/images/{1..1000}.png`：每个编号对应一张 3840×2160 RGB 增强结果。
- `artifacts/results/{final,baseline}/metrics.csv`：逐图 PSNR、SSIM、LPIPS 及颜色诊断；`summary.json` 保存全量和各划分均值，`run.json` 记录运行配置和模型身份，`status.json` 记录进度，`verification.json` 保存完整性检查结果。`final/acceptance.json` 另记录实际权重的独立推理、分块一致性和指标复算检查。
- `reports/实习报告.md`：实测方法、结果、耗时和误差分析；`comparison.jpg`、`detail_crops.png`、`validation_curves.png` 分别展示整体对比、原尺寸局部和训练验证曲线。`experiment_summary.json` 和 `representative_cases.json` 保存报告用汇总与样例信息；`fixed_color_references.jpg` 展示颜色参考。
- `logs/`：`pipeline_*.log` 等记录各流水线阶段输出；实验子目录中的 `run_config.json`、`status.json`、`train.jsonl`、`validation.jsonl` 和 `val_*/` 记录训练参数、进度、损失、验证指标及预览。
- `smoke/score_input/` 和 `smoke/score_cpu/`：独立 PNG 评分入口的三张样图、评分表、汇总和交叉复核记录。

项目配置中的原始 RAW 和 GT 数据目录位于 `configs/canon.json` 指定的外部路径；本项目只读取原始数据。复现时先按“运行”章节安装依赖并检查配置，再运行测试和流水线。

## 结果位置

- `artifacts/checkpoints/selected.pt`：验证集选定的独立推理权重。
- `artifacts/results/final/images`：1000 张彩色增强 PNG。
- `artifacts/results/final/metrics.csv`：逐图指标；`summary.json`：分组均值。
- `artifacts/results/final/verification.json`：输出数量、尺寸、源文件检查。
- `artifacts/results/final/acceptance.json`：实际交付权重的无 GT 推理、分块一致性和指标复算。
- `artifacts/reports/实习报告.md` 和 `comparison.jpg`：实测报告与对比。
- `artifacts/data/manifest.json`：固定划分、来源、归一化和颜色标定。
- `artifacts/selection.json`：同验证集消融和选模依据。

只有实际生成且通过核验的文件才表示相应阶段完成。临时 pilot 来自早期流程诊断，不用于最终结果。官方增强预训练权重未能联网核验，本项目最终训练路线不声称复现其论文成绩。

交付检查：9 项基础测试通过；最终图和传统基线均为 1000 张 RGB PNG、1000 条有效指标；2000 个原始 RAW/JPEG 的大小和修改时间未变。实际权重检查中，RAW-only 与提交图最大相差 1 个 uint8 灰度级，分块与整图最大相差 1 级，三张指标复算与主 CSV 一致。检查详情见上述 JSON 和 `artifacts/logs/`。
