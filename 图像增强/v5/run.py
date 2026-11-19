# -*- coding: utf-8 -*-
"""v5 统一启动器：在动任何代码之前先把护栏焊死。

用法（项目根目录）：
    D:\\Python文件夹\\python.exe v5/run.py probe            # L0 探测
    D:\\Python文件夹\\python.exe v5/run.py train sony       # 训练 Sony
    D:\\Python文件夹\\python.exe v5/run.py train canon      # 训练 Canon
    D:\\Python文件夹\\python.exe v5/run.py eval sony        # 全量评估
    D:\\Python文件夹\\python.exe v5/run.py eval canon
    D:\\Python文件夹\\python.exe v5/run.py verify          # 哈希复校（确认没碰 v1）
    D:\\Python文件夹\\python.exe v5/run.py smoke           # 回退等价性冒烟

三道断言（任一失败即拒绝执行）：
  S1  src 必须从 v5/code/src 加载（不是项目根的 src），否则静默用旧代码
  S2  配置里所有输出路径必须落在 v5/ 下，否则会覆盖 v1 结果
  S3  （verify 子命令）outputs/src/configs 的 sha256 必须与快照一致
"""
import hashlib
import os
import subprocess
import sys

V5_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.dirname(V5_ROOT)
CODE_ROOT = os.path.join(V5_ROOT, "code")
PYTHON = sys.executable

# v1 的基准指标（用于口径自检）
V1_BASELINE = {"Sony": 20.4460, "Canon": 18.8427}


def _die(msg):
    print("\n" + "=" * 70)
    print("[护栏拦截] " + msg)
    print("=" * 70)
    sys.exit(2)


def assert_code_path():
    """S1：确认 src 解析到 v5/code/src，而不是项目根的 src。"""
    sys.path.insert(0, CODE_ROOT)
    # 清掉可能的缓存模块
    for k in [k for k in sys.modules if k == "src" or k.startswith("src.")]:
        del sys.modules[k]
    import src  # noqa: F401

    actual = os.path.abspath(src.__file__)
    expect_prefix = os.path.abspath(os.path.join(CODE_ROOT, "src")) + os.sep
    if not actual.startswith(expect_prefix):
        _die(f"S1 失败：src 从错误位置加载\n  实际: {actual}\n  期望前缀: {expect_prefix}\n"
             f"  原因：v5/code/src 未在 sys.path 首位，或环境里有同名包")
    print(f"[S1 OK] src -> {actual}")


def assert_output_paths(cfg, dataset):
    """S2：所有输出路径必须落在 v5/ 下。"""
    v5_abs = os.path.abspath(V5_ROOT) + os.sep
    checks = {
        "train.log_dir": cfg.train.log_dir,
        "train.ckpt_dir": cfg.train.ckpt_dir,
        "eval.output_dir": cfg.eval.output_dir,
    }
    wb_cache = cfg.data.get("wb_cache", "") or ""
    if wb_cache:
        checks["data.wb_cache"] = wb_cache
    isp = cfg.get("isp", None)
    if isp is not None and isp.get("gains_cache"):
        checks["isp.gains_cache"] = isp["gains_cache"]

    bad = []
    for k, v in checks.items():
        full = os.path.abspath(os.path.join(PROJ_ROOT, v))
        # 目录型缓存（如 v5/cache）本身也算在 v5 下，前缀判断已覆盖
        if not (full + os.sep).startswith(v5_abs) and not full.startswith(v5_abs):
            bad.append(f"  {k} = {v}\n     -> {full}  （不在 v5/ 下）")
    if bad:
        _die("S2 失败：以下路径会写到 v5/ 之外，可能导致覆盖 v1 结果\n" + "\n".join(bad))
    print(f"[S2 OK] {dataset}: {len(checks)} 个输出路径全部落在 v5/ 内")


# 数据集名 → 配置文件名。大小写不敏感查找，未命中直接报错，
# 绝不静默回退到某个默认值（否则 `train sony` 会误跑 Canon 配置）。
DATASET_CFG = {"sony": "config_v5.yaml", "canon": "config_v5_canon.yaml"}


def resolve_dataset(tok):
    """把命令行里的数据集名规范化成 'Sony' / 'Canon'。"""
    key = (tok or "sony").strip().lower()
    if key not in DATASET_CFG:
        _die(f"未知数据集 {tok!r}。可选：sony / canon（大小写不敏感）")
    return key.capitalize(), DATASET_CFG[key]


def load_cfg(dataset):
    from src.config import load_config
    _, name = resolve_dataset(dataset)
    path = os.path.join(V5_ROOT, "configs", name)
    if not os.path.exists(path):
        _die(f"配置文件不存在: {path}")
    return load_config(path)


def code_sha():
    """v5/code/src 的合并哈希，写进 checkpoint 用于溯源。"""
    h = hashlib.sha256()
    base = os.path.join(CODE_ROOT, "src")
    for root, _, files in os.walk(base):
        for f in sorted(files):
            if not f.endswith(".py"):
                continue
            full = os.path.join(root, f)
            rel = os.path.relpath(full, base)
            h.update(rel.encode("utf-8"))
            with open(full, "rb") as fh:
                h.update(fh.read())
    return h.hexdigest()[:16]


def run(script, args, dataset=None):
    env = dict(os.environ)
    env["PYTHONPATH"] = CODE_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    env["V5_CODE_SHA"] = code_sha()
    cmd = [PYTHON, os.path.join(CODE_ROOT, "src", script)] + args
    print(f"[run] {' '.join(cmd)}")
    print(f"[run] code_sha={code_sha()}")
    return subprocess.call(cmd, cwd=PROJ_ROOT, env=env)


def announce(cfg, ds_name):
    """执行前把实际生效的关键配置打出来 —— 防止「跑错配置」这类静默事故。"""
    print(f"[cfg] dataset={cfg.data.dataset}  config={CFG_FILE_OF(ds_name)}")
    print(f"[cfg] use_wb={cfg.data.get('use_wb')}  "
          f"chroma_weight={cfg.train.get('chroma_weight')}  "
          f"ckpt_dir={cfg.train.ckpt_dir}")


def CFG_FILE_OF(ds_name):
    return DATASET_CFG[ds_name.lower()]


def cmd_probe(argv):
    return run(os.path.join("..", "scripts", "probe_v5_wb.py"), argv)


def cmd_train(argv):
    ds_name, cfg_file = resolve_dataset(argv[0] if argv else "sony")
    cfg = load_cfg(ds_name)
    assert_output_paths(cfg, ds_name)
    announce(cfg, ds_name)
    return run("train.py", ["--config", os.path.join(V5_ROOT, "configs", cfg_file)])


def cmd_eval(argv):
    ds_name, cfg_file = resolve_dataset(argv[0] if argv else "sony")
    cfg = load_cfg(ds_name)
    assert_output_paths(cfg, ds_name)
    announce(cfg, ds_name)
    return run("evaluate.py", ["--config", os.path.join(V5_ROOT, "configs", cfg_file)])


def cmd_verify(argv):
    """S3：确认 outputs/src/configs 一个字节都没动。"""
    snap = os.path.join(V5_ROOT, ".baseline_sha256.txt")
    if not os.path.exists(snap):
        _die("找不到快照文件 v5/.baseline_sha256.txt")
    bad, missing, ok = [], [], 0
    with open(snap, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            digest, path = line.split("  ", 1)
            if not os.path.exists(path):
                missing.append(path)
                continue
            h = hashlib.sha256()
            with open(path, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            if h.hexdigest() != digest:
                bad.append(path)
            else:
                ok += 1
    print(f"\n[S3] 一致 {ok} 个 / 变更 {len(bad)} 个 / 缺失 {len(missing)} 个")
    if bad:
        print("\n!! 以下文件被改动：")
        for p in bad:
            print("   -", p)
    if missing:
        print("\n!! 以下文件缺失：")
        for p in missing:
            print("   -", p)
    if bad or missing:
        _die("S3 失败：v1 产物被改动！")
    print("[S3 OK] v1 全部产物与源码零改动")


def cmd_smoke(argv):
    cfg = load_cfg("sony")
    assert_output_paths(cfg, "sony")
    return run(os.path.join("..", "scripts", "smoke_v5.py"), argv)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)
    cmd = sys.argv[1]
    argv = sys.argv[2:]
    assert_code_path()

    table = {
        "probe": cmd_probe,
        "train": cmd_train,
        "eval": cmd_eval,
        "verify": cmd_verify,
        "smoke": cmd_smoke,
    }
    if cmd not in table:
        print(__doc__)
        sys.exit(1)
    sys.exit(table[cmd](argv) or 0)


if __name__ == "__main__":
    main()
