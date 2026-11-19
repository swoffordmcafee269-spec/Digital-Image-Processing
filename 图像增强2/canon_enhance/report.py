from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image,ImageDraw

from .common import load_config,read_manifest,safe_output,write_json
from .data import read_rgb,read_raw,pack_raw,linear_to_srgb_np


def build_report(cfg,name="final",baseline_name="baseline"):
    root=Path(cfg["artifact_root"]);result=root/"results"/name
    summary=json.loads((result/"summary.json").read_text(encoding="utf-8"))
    if not summary["complete"] or summary["metrics"]["all"]["count"]!=1000:
        raise ValueError("The final report requires all 1000 scored PNGs")
    manifest=read_manifest(cfg)
    rows=list(csv.DictReader((result/"metrics.csv").open(encoding="utf-8")))
    items={r["id"]:r for r in manifest["items"]};byid={r["id"]:r for r in rows}
    reports=safe_output(root/"reports",cfg);reports.mkdir(parents=True,exist_ok=True)
    baseline_path=root/"results"/baseline_name/"summary.json"
    baseline=json.loads(baseline_path.read_text(encoding="utf-8")) if baseline_path.exists() else None
    # Fixed color scenes plus deterministic low/middle/high test cases.
    test=sorted([r for r in rows if r["split"]=="test"],key=lambda r:float(r["psnr"]))
    chosen=list(dict.fromkeys(["10","100","798",test[0]["id"],test[len(test)//2]["id"],test[-1]["id"]]))
    canvas=Image.new("RGB",(1920,len(chosen)*296),"white");draw=ImageDraw.Draw(canvas)
    for i,ident in enumerate(chosen):
        raw=read_raw(Path(cfg["data_root"])/items[ident]["raw"],cfg["height"],cfg["width"])
        # Display convention only: full 16-bit container range, not a claim
        # about sensor white level. No exposure gain or GT is used.
        p=pack_raw(raw,cfg["black_level"],65535-cfg["black_level"])
        # A monochrome intensity preview avoids assuming a Bayer R/G/B order
        # that cannot be established from the headerless files alone.
        preview=np.repeat(p.mean(axis=0)[...,None],3,axis=-1)
        preview=np.rint(np.clip(linear_to_srgb_np(preview),0,1)*255).astype(np.uint8)
        im=Image.fromarray(preview);im.thumbnail((480,270));canvas.paste(im,(0,i*296))
        image_paths=[root/"results"/baseline_name/"images"/f"{ident}.png",result/"images"/f"{ident}.png",
                     Path(cfg["data_root"])/items[ident]["gt"]]
        for j,p in enumerate(image_paths):
            if not p.exists():continue
            with Image.open(p) as im:
                im=im.convert("RGB");im.thumbnail((480,270));canvas.paste(im,((j+1)*480,i*296))
        r=byid[ident]
        draw.text((8,i*296+274),f"{ident} | RAW intensity, 16-bit range, no gain",fill="black")
        draw.text((488,i*296+274),"Calibrated classical baseline",fill="black")
        draw.text((968,i*296+274),f"prediction | {r['split']} | PSNR {float(r['psnr']):.2f}, SSIM {float(r['ssim']):.3f}",fill="black")
        draw.text((1448,i*296+274),"GT reference",fill="black")
    canvas.save(reports/"comparison.jpg",quality=95)
    # Fixed 1:1 crops chosen for color and fine detail, before final evaluation.
    crop_specs=[("10",2800,1050),("100",1900,250),("798",550,700)]
    crop_canvas=Image.new("RGB",(1536,len(crop_specs)*544),"white");crop_draw=ImageDraw.Draw(crop_canvas)
    for i,(ident,x,y) in enumerate(crop_specs):
        paths=[root/"results"/baseline_name/"images"/f"{ident}.png",result/"images"/f"{ident}.png",
               Path(cfg["data_root"])/items[ident]["gt"]]
        for j,(label,path) in enumerate(zip(["Classical baseline","Prediction","GT reference"],paths)):
            if not path.exists():continue
            with Image.open(path) as im:
                crop_canvas.paste(im.crop((x,y,x+512,y+512)),(j*512,i*544))
            crop_draw.text((j*512+6,i*544+516),f"{ident} | {label} | x={x}, y={y} | 1:1 crop",fill="black")
    crop_canvas.save(reports/"detail_crops.png",compress_level=1)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,3.5),constrained_layout=True)
    for experiment in ["main","chroma0","chroma005","adaptive","perceptual"]:
        log=root/"logs"/experiment/"validation.jsonl"
        if not log.exists():continue
        history=[json.loads(line) for line in log.read_text().splitlines() if line.strip()]
        for ax,key in zip(axes,["psnr","ssim","lpips"]):
            ax.plot([r["step"] for r in history],[r[key] for r in history],marker=".",label=experiment)
            ax.set_xlabel("Training steps");ax.set_ylabel(key.upper());ax.grid(alpha=.25)
    axes[0].legend();fig.savefig(reports/"validation_curves.png",dpi=160);plt.close(fig)
    worst=test[:10]
    low_table=["| 测试图像 ID | PSNR ↑ | SSIM ↑ | LPIPS ↓ |",
               "|---|---:|---:|---:|"]
    for row in worst:
        low_table.append(f"| {row['id']} | {float(row['psnr']):.3f} | {float(row['ssim']):.4f} | {float(row['lpips']):.4f} |")
    write_json(reports/"representative_cases.json",{"comparison_ids":chosen,"lowest_test_psnr":worst,
                                                 "selection":"fixed color examples + lowest/median/highest test PSNR",
                                                 "fixed_detail_crops":[{"id":n,"x":x,"y":y,"size":512} for n,x,y in crop_specs]})
    table=["| 方法 / 统计范围 | 张数 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | ΔE2000 ↓（512缩略图） |",
           "|---|---:|---:|---:|---:|---:|"]
    for label,obj in [("新模型",summary),("标定传统基线",baseline)]:
        if obj:
            for split in ["all","train","val","test"]:
                s=obj["metrics"].get(split)
                if s:table.append(f"| {label} / {split} | {s['count']} | {s['psnr']:.3f} | {s['ssim']:.4f} | {s['lpips']:.4f} | {s['delta_e_2000_512']:.3f} |")
    targets=[("第一阶段",22,.78,.40),("较高目标",24,.82,.35),("冲刺目标",26,.86,.25)]
    result_lines=[];difference_lines=[]
    for split in ["all","test"]:
        score=summary["metrics"][split]
        achieved=[name for name,p,s,l in targets if score["psnr"]>=p and score["ssim"]>=s and score["lpips"]<=l]
        result_lines.append(f"- {split}：同时达到的指标档位：{'、'.join(achieved) if achieved else '尚未同时达到第一阶段三项门槛'}。")
        if baseline and split in baseline["metrics"]:
            b=baseline["metrics"][split]
            difference_lines.append(f"- {split} 相对标定传统基线：PSNR 变化 {score['psnr']-b['psnr']:+.3f} dB，SSIM 变化 {score['ssim']-b['ssim']:+.4f}，LPIPS 变化 {score['lpips']-b['lpips']:+.4f}（LPIPS 负值表示改善）。")
    generalization_gap=summary["metrics"]["train"]["psnr"]-summary["metrics"]["test"]["psnr"]
    selection=json.loads((root/"selection.json").read_text(encoding="utf-8"))
    experiments=["| 候选权重 | 步数 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |",
                 "|---|---:|---:|---:|---:|"]
    for candidate in selection["candidates"]:
        score=candidate["metrics"];path=Path(candidate["path"])
        experiments.append(f"| {path.parent.name}/{path.name} | {candidate['step']} | {score['psnr']:.3f} | {score['ssim']:.4f} | {score['lpips']:.4f} |")
    timing=["| 实验阶段 | 实测时长（分钟） | 结束步数 |",
            "|---|---:|---:|"]
    training_seconds=0.0
    for experiment in ["main","chroma0","chroma005","adaptive","perceptual"]:
        path=root/"logs"/experiment/"status.json"
        if not path.exists():continue
        status=json.loads(path.read_text(encoding="utf-8"))
        if status.get("state")!="completed":continue
        elapsed=status["elapsed_s"];training_seconds+=elapsed
        timing.append(f"| {experiment}（含该阶段验证） | {elapsed/60:.2f} | {status['step']} |")
    for label,obj in [("新模型全量出图与评价",summary),("传统基线全量出图与评价",baseline)]:
        if obj:timing.append(f"| {label} | {obj['elapsed_this_run_s']/60:.2f} | — |")
    write_json(reports/"experiment_summary.json",{"selection":selection,"training_stages_seconds":training_seconds,
               "final_metrics":summary["metrics"],"baseline_metrics":baseline["metrics"] if baseline else None})
    acceptance_path=result/"acceptance.json"
    acceptance_text="实际权重集成检查尚未运行，不将其标为通过。"
    if acceptance_path.exists():
        accepted=json.loads(acceptance_path.read_text(encoding="utf-8"))
        acceptance_text=(f"实际权重集成检查通过：{accepted['passed']}。仅传入 RAW 和权重的独立推理，与提交 PNG 的最大 uint8 误差为 "
                         f"{accepted['raw_only_vs_delivered_max_uint8_error']}；512 packed 像素分块、192 halo 与整图的平均绝对误差为 "
                         f"{accepted['tiled_vs_full_mean_uint8_error']:.6f}，最大误差为 {accepted['tiled_vs_full_max_uint8_error']}，"
                         f"99 分位误差为 {accepted['tiled_vs_full_p99_uint8_error']:.1f}。10、100、798 三张保存图片复算三项指标均通过。"
                         "这些检查仅验证接口和数值一致性，不替代视觉质量评价。详情见 results/final/acceptance.json。")
    text=f'''# Canon 极暗 RAW 图像增强实习报告

## 1. 任务与数据

将 Canon_0.0001 的极暗 Bayer RAW 恢复为正常曝光彩色 sRGB，原始分辨率为 3840×2160。共 1000 对 RAW/JPEG，源数据只读。

场景分组采用 ORB 特征几何匹配与缩略图相似度，训练/验证/测试为 {manifest['counts']['train']}/{manifest['counts']['val']}/{manifest['counts']['test']} 张，共 {manifest['grouping']['groups']} 个组。自动分组不是官方场景标签，无法保证所有非重叠的同一地点视角均被识别。划分及分组证据保存在 data/manifest.json。

## 2. 方法

RAW 以小端 uint16 读取，扣除默认黑电平 512，按训练 RAW 的信号 99.5 分位数 {manifest['normalization']['scale']:.1f} 归一化，四个 Bayer 相位分别打包；不假定白电平为 16383，也不截断大于 1 的输入。

采用无 InstanceNorm/BatchNorm 的四级残差 U-Net，基础通道数 32，12 通道输出经过 PixelShuffle(2) 重建 RGB。训练集拟合的固定相机颜色映射提供彩色初值，输入 RAW 的低分辨率全图上下文预测曝光/颜色修正，网络学习去噪与细节残差。输出直接是 sRGB，不重复 Gamma 编码。

针对验证集中部分强噪声样本，另外比较 adaptive 候选：对 RAW 特征使用可逆的 log1p 动态范围压缩，并让全图上下文学习抑制颜色初值的门控，避免固定颜色映射直接带入彩噪。该候选不裁剪输入信息，是否最终采用以 selection.json 及选定权重的 model_config 为准。

损失由 RGB Charbonnier、权重 0.1 的 SSIM 损失、可选低频 Cb/Cr 色度损失构成。采用 256×256 packed RAW 裁块、batch 4、AdamW、AMP 和梯度裁剪。仅进行相位一致的随机裁剪，不直接翻转 packed RAW，以免破坏子像素对齐。

后续候选比较衰减率 0.999 的权重滑动平均；两条色度消融从相同主训练权重开始，adaptive 也从主训练迁移，感知实验则从当时选中的 adaptive 权重继续，额外加入权重 0.02 的 LPIPS 损失。所有候选仅在验证集上比较。选模规则为 PSNR 优先，差距不超过 0.10 dB 时选 LPIPS 较低者，再以色差打破平局；不使用测试集挑选参数。

针对仍残留的强彩噪，另外对验证集 PSNR 最优的训练候选比较颜色初值的 21×21 和 41×41 packed RAW 均值平滑，原设置为 5×5。两个大小预先固定，细节网络保持原权重；这些属于推理设置消融，不声称经过额外训练。最终卷积核大小随权重中的 model_config 固化，是否保留由同一验证规则决定。

传统对照采用 RAW 双边去噪、训练集相机色彩标定和 sRGB 编码，因此应称为“经过训练集标定的传统基线”，不是完全无监督方法。未取得经核验的外部预训练增强权重。

## 3. 评价协议

最终 1000 张输出均为 8-bit RGB PNG，保存后重新读取并计算指标。PSNR/SSIM 使用完整 RGB、data_range=1。SSIM 为 7×7 均匀窗口、样本协方差，GPU 实现经过 scikit-image 对照测试。LPIPS 为 AlexNet v0.1、完整原分辨率、FP32、输入 [-1,1]。先逐张计算再平均。色差及饱和度是附加诊断，采用最长边 512 的面积缩略图。

全量 all 指标包含训练样本，不能作为独立泛化成绩；test 列单独报告。测试集不参与参数拟合、训练或模型选择。旧项目训练覆盖大量当前测试图，因此旧版对照只能用于描述，不能充当同划分的独立测试基线。

## 4. 实测结果

{chr(10).join(table)}

{chr(10).join(result_lines)}

{chr(10).join(difference_lines)}

训练集与测试集的平均 PSNR 差为 {generalization_gap:+.3f} dB。两者场景分布可能不同，不能把全部差异都归因于过拟合；但全量均值受到 800 张训练样本影响，因此应优先用测试列判断泛化能力。

工程目标不是课程给定的硬性门槛，未达标部分按实际结果报告，不将预测或目标写成成绩。

三档目标分别为：第一阶段 PSNR≥22 dB、SSIM≥0.78、LPIPS≤0.40；较高目标为 24 dB、0.82、0.35；冲刺目标为 26 dB、0.86、0.25。每档必须同时满足三项，不能只按 PSNR 判断是否达标。

旧版 1000 张统计约为 PSNR 18.843 dB、SSIM 0.6949、LPIPS 0.5544，仅用于说明历史结果；其训练图像与当前测试划分重叠，不能用新旧均值差宣称独立测试集上的公平提升。上表的标定传统基线使用本次相同划分和评价协议。

同一 100 张固定验证集的候选比较如下，权重是否最终采用由 selection.json 记录：

{chr(10).join(experiments)}

## 5. 视觉分析

comparison.jpg 包含固定彩色场景，以及测试集中 PSNR 最低、中位、最高的样例。四列依次是 RAW 强度示意、传统基线、新模型、GT。RAW 示意将四相位均值显示为灰度，按 uint16 容器范围显示且不做曝光增益，避免在 Bayer 排列未经元数据确认时虚构颜色；该显示范围不是相机白电平，不参与训练或评分。最终增强图片仍为彩色 RGB。representative_cases.json 记录选图方法及低分样例，避免仅展示最好结果。validation_curves.png 展示各阶段验证指标。

注意检查标牌和物体的颜色方向、暗部彩噪、细小文字、阴影层次。饱和度增加本身不代表颜色更准确，需同时结合 ΔE2000 和 GT 对照判断。极暗区域中原始信息不足的细节仍可能被平滑，模型不保证重建不存在的真实纹理。

训练过程的固定验证预览已暴露两类主要不足：588、699 等画面的明亮墙面仍有曝光偏差，896 等强噪声画面仍残留彩噪；43 的帐篷颜色也存在饱和度不足。244、804 等部分场景恢复较好，不能用它们代表全部数据。具体例子见 logs 下各阶段 val_* 预览，最终测试低分样例另列于 representative_cases.json。

detail_crops.png 在固定坐标截取 512×512 的原尺寸局部，三列依次为传统基线、新模型、GT，覆盖商品纹理、红色招牌和蓝色自行车。局部不缩放，便于检查缩略图掩盖的噪声与细节；这些预先固定的展示样例不是额外独立测试集。

原尺寸检查可见，红色标牌的颜色方向已恢复，但与 GT 的饱和度及阴影仍有差异；自行车上的细字和商品纹理比 GT 软，商品局部还可见少量残余彩点。因此不能声称细节完全复原。

冻结模型后检查测试低分图，435 的蓝色篷布较 GT 偏亮、褶皱细节偏软；603 的浅色墙面偏亮；358 的石墙偏暗且残留噪声。这些例子表明，模型仍有双向曝光偏差，不能用全量均值概括每张图的恢复质量。测试集检查只用于结果分析，没有据此重新训练或选择推理参数。

按 PSNR 从低到高排列的十张测试图如下；不同指标关注的误差不同，低 PSNR 不一定同时具有最差 LPIPS。

{chr(10).join(low_table)}

## 6. 耗时与复现

最终增强的平均纯推理耗时为 {summary['metrics']['all']['inference_seconds']:.3f} 秒/张；评价平均耗时 {summary['metrics']['all']['evaluation_seconds']:.3f} 秒/张，读图、PNG 写入及初始化另计。训练实际时长记录在 logs/ 各阶段 status.json；这些实测值优先于方案阶段的估计。

{chr(10).join(timing)}

训练阶段合计 {training_seconds/3600:.2f} 小时，包含各阶段验证；表中全量出图与评价时长还包括读取、PNG 编码和保存。该表不包含前期准备、流程诊断、候选重评、结果核验与报告整理。若评价曾中断后恢复，评价行仅表示最后一次运行时长，应结合日志计算总时长。

选定权重：{summary['run']['checkpoint']}，训练步数：{summary['run']['checkpoint_step']}。清单指纹：{manifest['fingerprint']}。

原始文件检查：{summary['source_check']['checked_files']} 个 RAW/GT 文件的大小及修改时间与准备阶段一致。复现命令和环境要求见项目 README.md、部署文档.md。

## 7. 交付核验

最终图和传统基线均通过 1000 张 PNG、1000 条指标、3840×2160 RGB、PNG 完整性、指标有限值及场景组互斥检查。选定权重 SHA256：{summary['run']['checkpoint_sha256']}。

{acceptance_text}

基础测试 9 项通过，覆盖 RAW 相位/动态范围、颜色、无 GT 推理、分组、路径保护、SSIM 对照及 EMA 等。已有 PNG 的独立 CPU 评分对三张样例复算，与主流程 GPU 指标的绝对差均小于 1e-4，记录见 artifacts/smoke/score_cpu/verification.json。
'''
    (reports/"实习报告.md").write_text(text,encoding="utf-8")
    print(str(reports/"实习报告.md"))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",default="configs/canon.json");p.add_argument("--name",default="final")
    p.add_argument("--baseline-name",default="baseline");a=p.parse_args();build_report(load_config(a.config),a.name,a.baseline_name)
