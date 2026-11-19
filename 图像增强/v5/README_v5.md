# v5 工作区（色彩恢复改进版）

**本目录是 v5 独立工作区。项目根下的 `src/`、`configs/`、`outputs/` 全部只读，一个字节都不写入。**

## 与 v1 的隔离

| 保护措施 | 文件 |
|---|---|
| 执行前全量哈希快照（1042 个文件：`outputs/**` 的 CSV/PNG/PT/LOG + `src/**` + `configs/**`） | `v5/.baseline_sha256.txt` |
| 图片清单（2000 张，按大小校验） | `v5/.baseline_images_manifest.txt` |
| CSV/日志/对比图备份（17 个文件，11 MB，不含图片） | `v5/.backup_csv/` |
| 运行期断言（S1 代码来源 / S2 输出路径前缀） | `v5/run.py` |
| 事后校验（S3 逐文件 sha256 比对） | `v5/run.py verify` |

**回退**：删除整个 `v5/` 目录 = 完全回到当前状态，零残留。

## 代码为什么在 `v5/code/src/`

因为不能改项目根的 `src/`。所以整份复制过来，只改副本。`v5/run.py` 会强制断言 `src` 实际从 `v5/code/src/` 加载 —— 如果哪天忘了设路径而静默加载到旧代码，启动器会直接拒绝执行。

## 用法（项目根目录下）

```bash
set PY=D:\Python文件夹\python.exe

%PY% v5\run.py probe                  # L0 探测（Bayer 4 排列 + WB 普查 + 口径自检）
%PY% v5\run.py smoke                  # 冒烟（回退等价性等 4 条）
%PY% v5\run.py train sony             # 训练 Sony（约 73 min）
%PY% v5\run.py train canon            # 训练 Canon（约 73 min）
%PY% v5\run.py eval sony              # 全量评估 Sony（约 90 min）
%PY% v5\run.py eval canon             # 全量评估 Canon（约 90 min）
%PY% v5\run.py verify                 # 哈希复校：确认 v1 零改动
```

## 目录说明

| 目录 | 内容 |
|---|---|
| `code/src` | 源码副本（唯一允许修改的代码） |
| `code/scripts` | 脚本副本 + 新增探测脚本 |
| `configs/` | `config_v5.yaml`（Sony）、`config_v5_canon.yaml`（Canon） |
| `cache/` | WB 增益缓存 `wb_gains_v5_{sony,canon}.npy` |
| `probe/` | L0 探测产物（Bayer 对照 CSV/图、WB 统计） |
| `logs/` | `sony/`、`canon/` 训练日志 |
| `checkpoints/` | `sony/`、`canon/`，`best.pt` 内含 `code_sha` 便于溯源 |
| `results/` | `sony/`、`canon/`，各含 `results.csv` + `images/*.jpg`（JPG q95，省 20 GB） |
| `metrics/` | 三口径指标（PSNR/SSIM/LPIPS + 饱和度 + 色度 + 锐度） |
| `reports/` | 对比表、6 宫格、结论 |

## 出图格式为什么改 JPG

实测 PNG 12.39 MB/张 → 1000 张 12 GB；JPG q95 1.75 MB/张 → 1000 张 1.8 GB。指标计算全部在内存 float 上做，与存储格式无关，不影响任何数值。
