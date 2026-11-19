"""DOCX 质检：表格数值、图片像素、黑名单词、页眉页脚、与 PDF 逐页一致性。"""

import difflib
import json
import re
import sys
import zipfile
from pathlib import Path

import docx
import pymupdf
from PIL import Image
from io import BytesIO

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(HERE))
from content_v2 import PAGES  # noqa: E402

DOCX = BASE / '数字图像处理综合实习报告_修订版.docx'
PDF = BASE / '数字图像处理综合实习报告_修订版.pdf'
WORD_PDF = BASE / '_work' / 'word_export.pdf'

BLACKLIST = ['总结文档', '复核', '局限', '不足', '偏差', '偏软', '并未', '不支持', '不能',
             '无法', '未能', '不假定', '不作为', '不写作', '不反转', '仅为', '尚可', '有待']
res = {}

d = docx.Document(str(DOCX))
res['paragraphs'] = len(d.paragraphs)
res['tables'] = len(d.tables)

# 表格逐格比对
expected = []
for page in PAGES:
    for blk in page['blocks']:
        if blk['k'] == 'table':
            expected.append([[str(c) for c in blk['head']]] + [[str(c) for c in r] for r in blk['rows']])
actual = []
for t in d.tables:
    rows = []
    for row in t.rows:
        rows.append([c.text.replace('\n', '\n') for c in row.cells])
    actual.append(rows)
res['table_shape_ok'] = [len(a) for a in actual] == [len(e) for e in expected]
mismatch = []
for ti, (e, a) in enumerate(zip(expected, actual)):
    for ri, (er, ar) in enumerate(zip(e, a)):
        for ci, (ec, ac) in enumerate(zip(er, ar)):
            if ec != ac:
                mismatch.append((ti, ri, ci, ec, ac))
res['table_cell_mismatch'] = mismatch[:10]

# 图片像素（word/media）
with zipfile.ZipFile(DOCX) as z:
    media = [n for n in z.namelist() if n.startswith('word/media/')]
    dims = []
    for n in media:
        with Image.open(BytesIO(z.read(n))) as im:
            dims.append((n.split('/')[-1], im.size))
res['media_count'] = len(media)
res['media_dims'] = dims

# 页眉页脚
sec = d.sections[1]
res['header'] = sec.header.paragraphs[0].text
res['footer'] = sec.footer.paragraphs[0].text
res['sections'] = len(d.sections)

# 全文黑名单
full = '\n'.join(p.text for p in d.paragraphs)
for t in d.tables:
    for row in t.rows:
        for c in row.cells:
            full += '\n' + c.text
res['blacklist_hits'] = {w: full.count(w) for w in BLACKLIST if w in full}

# 与 PDF 逐页文本一致
def norm(t):
    t = t.replace('\u3000', '').replace(' ', '').replace('\n', '')
    t = re.sub(r'第\d+页/共\d+页', '', t)
    t = t.replace('数字图像处理综合实习报告', '', 1)
    t = re.sub(r'图像增强·几何校正·图像压缩', '', t, count=1)
    return t

A = pymupdf.open(PDF)
B = pymupdf.open(WORD_PDF)
res['pdf_pages'] = A.page_count
res['word_pages'] = B.page_count
ratios = []
for i in range(min(A.page_count, B.page_count)):
    ratios.append(round(difflib.SequenceMatcher(None, norm(A[i].get_text()), norm(B[i].get_text())).ratio(), 4))
res['page_text_ratio'] = ratios

(BASE / '_work' / 'qa_docx.json').write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
print('paragraphs', res['paragraphs'], 'tables', res['tables'], 'shape_ok', res['table_shape_ok'])
print('cell mismatch', len(res['table_cell_mismatch']), res['table_cell_mismatch'][:3])
print('media', res['media_count'], res['media_dims'][:3])
print('sections', res['sections'], '| header:', res['header'], '| footer:', res['footer'])
print('blacklist', res['blacklist_hits'])
print('pages', res['pdf_pages'], res['word_pages'])
print('page ratios', ratios)
