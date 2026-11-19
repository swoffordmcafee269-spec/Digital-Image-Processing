"""修订版 PDF 质检：页码、目录、图片、数字、黑名单词、与原版逐页文本差异。"""

import json
import re
import sys
from pathlib import Path

import pymupdf

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = BASE.parent
sys.path.insert(0, str(HERE))
from content_v2 import DATA, PAGES  # noqa: E402

NEW = BASE / '数字图像处理综合实习报告_修订版.pdf'
OLD = Path(r'C:\Users\001\Downloads\数字图像处理综合实习报告(1).pdf')
OUT = BASE / '_work' / 'qa_v2.json'

BLACKLIST = ['总结文档', '复核', '局限', '不足', '偏差', '偏软', '并未', '不支持', '不能',
             '无法', '未能', '不假定', '不作为', '不写作', '不反转', '仅为', '尚可', '有待',
             '修订版', '原文', '批注', '仅供参考', '水平有限', '请指正', '遗憾', '篇幅所限']

doc = pymupdf.open(NEW)
old = pymupdf.open(OLD)
res = {'pages': doc.page_count, 'expected_pages': len(PAGES) + 1}

# 1. 文本与黑名单
texts = [pg.get_text() for pg in doc]
full = '\n'.join(texts)
res['blacklist_hits'] = {w: full.count(w) for w in BLACKLIST if w in full}

# 2. 目录页码：找每个章节标题所在正文页
toc_rows = None
for page in PAGES:
    for blk in page['blocks']:
        if blk['k'] == 'table' and blk['head'] == ['内容', '起始页']:
            toc_rows = blk['rows']
starts = {}
for idx, page in enumerate(PAGES, 1):
    starts[page['title']] = idx
    for blk in page['blocks']:
        if blk['k'] == 'h1':
            starts[blk['t']] = idx
toc_actual = []
for title, page_no in toc_rows:
    key = title
    hit = None
    for t, idx in starts.items():
        if t.replace('　', '') == key.replace('　', ''):
            hit = idx
    toc_actual.append({'title': title, 'toc_page': int(page_no), 'actual_page': hit})
res['toc'] = toc_actual
res['toc_ok'] = all(r['actual_page'] == r['toc_page'] for r in toc_actual)

# 3. 图片：数量、像素尺寸
assets = {p.name: p for p in (BASE / 'assets').glob('*.jpg')}
img_report = []
for i, pg in enumerate(doc, 1):
    for im in pg.get_images(full=True):
        info = doc.extract_image(im[0])
        img_report.append({'page': i, 'w': info['width'], 'h': info['height'], 'ext': info['ext']})
res['images'] = {'count': len(img_report), 'items': img_report}
res['image_dims_ok'] = all((it['w'], it['h']) in {
    (2780, 742), (2780, 482), (1955, 510), (1785, 510), (1805, 510), (2085, 510),
    (1800, 690)} for it in img_report)

# 4. 表格数值：content_v2 的行必须与 JSON 一致，且文本出现在 PDF 中
checks = []
for page in PAGES:
    for blk in page['blocks']:
        if blk['k'] != 'table':
            continue
        for row in blk['rows']:
            for cell in row:
                s = str(cell)
                if re.fullmatch(r'-?\d+(\.\d+)?', s):
                    checks.append({'value': s, 'in_pdf': s in full})
res['table_numbers'] = {'total': len(checks), 'missing': [c for c in checks if not c['in_pdf']][:20]}

# 5. 与原版逐页差异
def norm(t):
    t = t.replace('\u3000', '').replace(' ', '').replace('\n', '')
    return t

import difflib
diffs = []
for i in range(min(len(texts), len(old))):
    a = norm(old[i].get_text())
    b = norm(texts[i])
    if a != b:
        sm = difflib.SequenceMatcher(None, a, b)
        ops = [op for op in sm.get_opcodes() if op[0] != 'equal']
        diffs.append({'page': i + 1, 'similarity': round(sm.ratio(), 4), 'blocks': len(ops)})
res['diff_pages'] = diffs

with OUT.open('w', encoding='utf-8') as f:
    json.dump(res, f, ensure_ascii=False, indent=2)

print('pages', res['pages'])
print('blacklist', res['blacklist_hits'])
print('toc_ok', res['toc_ok'], [r for r in toc_actual if r['actual_page'] != r['toc_page']])
print('images', res['images']['count'], 'dims_ok', res['image_dims_ok'])
print('table numbers', res['table_numbers']['total'], 'missing', len(res['table_numbers']['missing']))
print('pages with text changes', len(diffs))
for d in diffs:
    print('  ', d)
