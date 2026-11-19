# -*- coding: utf-8 -*-
"""环境自检脚本：打印 Python / CUDA / GPU / 依赖库状态，不安装任何东西。

用法：
    python scripts/check_env.py
"""
import sys
import importlib

REQUIRED = [
    "torch",
    "torchvision",
    "numpy",
    "cv2",
    "skimage",
    "lpips",
    "rawpy",
    "PIL",
    "matplotlib",
    "tqdm",
    "yaml",
]

print("=" * 60)
print("Python 版本:", sys.version.split()[0], "|", sys.executable)
print("=" * 60)

# 依赖检查
print("\n[依赖库]")
missing = []
for name in REQUIRED:
    try:
        mod = importlib.import_module(name)
        version = getattr(mod, "__version__", "?")
        print(f"  [OK]   {name:<14} {version}")
    except Exception:
        print(f"  [MISS] {name:<14} 未安装")
        missing.append(name)

# PyTorch / CUDA 检查
print("\n[CUDA / GPU]")
try:
    import torch
    print("  torch 版本      :", torch.__version__)
    print("  CUDA 可用       :", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("  CUDA 版本       :", torch.version.cuda)
        print("  GPU 数量        :", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU[{i}]         : {props.name}, "
                  f"显存 {props.total_memory / 1024**3:.1f} GB")
except Exception as e:
    print("  torch 未安装或加载失败:", e)

print("\n" + "=" * 60)
if missing:
    print("缺失依赖:", ", ".join(missing))
    print("请执行: pip install -r requirements.txt")
    print("torch 请按 CUDA 版本安装，见 README.md")
else:
    print("所有依赖均已安装。")
print("=" * 60)
