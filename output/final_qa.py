from pathlib import Path
import json, os, hashlib, re, sys
from PIL import Image, ImageDraw, ImageFont
from pypdf import PdfReader
import pdfplumber

sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path(r'D:\数字图像处理实习作业')
OUT = ROOT / 'output'
PDF = OUT / 'pdf/数字图像处理综合实习报告.pdf'
before = json.loads((OUT/'source_integrity_before.json').read_text(encoding='utf-8'))
after = {}
for base in ('图像增强2', '几何校正workbuddy2', '图像压缩'):
    for folder, dirs, files in os.walk(ROOT/base):
        dirs[:] = [d for d in dirs if d not in ('venv','venv_cai','.venv','.git','__pycache__','.pytest_cache')]
        for name in files:
            p = Path(folder)/name
            st = p.stat()
            after[str(p.relative_to(ROOT))] = [st.st_size, st.st_mtime_ns]
changes = [p for p in before if before[p] != after.get(p)]
added = sorted(set(after)-set(before))
hashes = json.loads((OUT/'source_hashes.json').read_text(encoding='utf-8'))
hash_changes = []
for path, expected in hashes.items():
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):
            h.update(b)
    if h.hexdigest()!=expected:
        hash_changes.append(path)

reader = PdfReader(str(PDF))
texts = [p.extract_text() or '' for p in reader.pages]
(OUT/'tmp/extracted_report.txt').write_text('\n\n'.join(f'--- 实际第{i+1}页 ---\n{t}' for i,t in enumerate(texts)),encoding='utf-8')
images = sum(len(p.images) for p in reader.pages)
starts = []
with pdfplumber.open(PDF) as doc:
    for i,page in enumerate(doc.pages,1):
        for line in page.extract_text_lines():
            if re.match(r'^[，。；：、！？）】》”]',line['text']):
                starts.append({'page':i,'text':line['text'],'top':round(line['top'],1)})
replacement_count = sum(t.count('\ufffd') for t in texts)
mapping = json.loads((OUT/'figure_sources.json').read_text(encoding='utf-8'))
missing_captions = [m['figure'] for m in mapping if m['figure'] not in ''.join(texts)]

md = ['# 实习报告选图来源对照清单','',
      '本清单独立于报告正文，列明实际采用的 12 组图像编号、原始来源及报告图号。全部对比拼图均另存于 output/report_assets，原作业成果文件不作修改。','',
      '## 图像增强','',
      '每组从左到右为 RAW 强度示意、标定传统基线、最终增强结果、正常曝光参考 GT。10、100、798 另含同位置局部裁剪；518、435 为测试集样例。RAW 示意仅按 Bayer 四相均值及 65535 归一化显示，不代表相机白电平标定。','']
last = '图像增强'
for m in mapping:
    if m['assignment'] != last:
        last=m['assignment']
        md += ['## '+last,'']
        if last=='几何校正':
            md += ['各组从左到右为原始畸变文档、四角透视基线及 UVDoc 校正结果。','']
        else:
            md += ['使用已保存的原图与压缩重建对比图，报告副本将编码器标识统一为 DCT 与 AVIF；图示 PSNR 为整图结果。','']
    md += [f"### {m['figure']}：{m['id']}",'']
    if m.get('split'): md += [f"数据划分：{m['split']}。",'']
    if m['assignment']=='图像增强': labels=['RAW 原始数据','传统基线输出','最终增强输出','正常曝光 GT']
    elif m['assignment']=='几何校正': labels=['原始图像','四角透视基线输出','UVDoc 输出']
    else: labels=['原始图像','已保存的压缩对比结果']
    md += [f'- {label}：`{path}`' for label,path in zip(labels,m['sources'])]
    md += [f"- 报告采用的拼图副本：`{m['asset']}`",'']
md += ['## 统计图','',
       '- 图3-1：增强测试集 PSNR 频数分布，依据图像增强2/artifacts/results/baseline/metrics.csv 与 final/metrics.csv 中的 test 记录（各 100 条）。',
       '- 图4-1：OCR 字符数与平均置信度柱状图，依据几何校正workbuddy2/eval/metric_ocr.csv（200 条）。',
       '- 图5-1：平均率失真曲线，依据图像压缩/results/metrics.csv（24 幅图像、10 组配置，共 240 条）。','']
(OUT/'选图来源对照清单.md').write_text('\n'.join(md),encoding='utf-8')

font=ImageFont.truetype(r'C:\Windows\Fonts\simhei.ttf',24)
for first in range(1,17,4):
    sheet=Image.new('RGB',(1240,1810),'#d9d9d9')
    for j,pagen in enumerate(range(first,first+4)):
        im=Image.open(OUT/f'tmp/report_page-{pagen:02}.png').convert('RGB')
        im.thumbnail((600,850),Image.Resampling.LANCZOS)
        x=(j%2)*620+(620-im.width)//2
        y=(j//2)*905+38
        sheet.paste(im,(x,y))
        ImageDraw.Draw(sheet).text(((j%2)*620+15,(j//2)*905+5),f'第 {pagen} 页',font=font,fill='black')
    sheet.save(OUT/f'tmp/contact_{first:02}_{first+3:02}.png')

qa={'pdf_pages':len(reader.pages),'pdf_bytes':PDF.stat().st_size,'comparison_images':images,
    'replacement_characters':replacement_count,'missing_figure_captions':missing_captions,
    'source_inventory_count':len(before),'source_inventory_changed':changes,'source_inventory_added':added,
    'sha256_checked_count':len(hashes),'sha256_changed':hash_changes,
    'punctuation_line_starts':starts}
(OUT/'report_qa.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(qa,ensure_ascii=False,indent=2))
assert len(reader.pages)==16 and images==12
assert not changes and not added and not hash_changes
assert not replacement_count and not missing_captions
