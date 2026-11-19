"""句子级对照：列出修订版中与原版差异较大的句子，用于确认改动范围。

PDF 按行断字，直接比行会全是噪声；这里按句号/分号切句，再找最佳匹配。
"""

import difflib
import re
from pathlib import Path

import pymupdf

BASE = Path(__file__).resolve().parent.parent
NEW = BASE / '数字图像处理综合实习报告_修订版.pdf'
OLD = Path(r'C:\Users\001\Downloads\数字图像处理综合实习报告(1).pdf')


def sentences(path):
    doc = pymupdf.open(path)
    chunks = []
    for pg in doc:
        t = pg.get_text()
        t = re.sub(r'第 \d+ 页 / 共 \d+ 页', '', t)
        t = t.replace('数字图像处理综合实习报告', '').replace('图像增强 · 几何校正 · 图像压缩', '')
        chunks.append(t)
    text = re.sub(r'\s+', '', ''.join(chunks))
    parts = re.split(r'(?<=[。；])', text)
    return [p for p in parts if len(p) >= 10]


old = sentences(OLD)
new = sentences(NEW)
out = []
for s in new:
    best = max(old, key=lambda o: difflib.SequenceMatcher(None, o, s).ratio())
    r = difflib.SequenceMatcher(None, best, s).ratio()
    if r < 0.90:
        out.append((round(r, 3), best, s))

lines = []
for r, o, n in out:
    lines.append(f'相似度 {r}')
    lines.append('  旧: ' + o)
    lines.append('  新: ' + n)
    lines.append('')
(BASE / '_work' / 'sentence_diff.txt').write_text('\n'.join(lines), encoding='utf-8')
print('低相似度句子:', len(out), '/', len(new))
for r, o, n in out:
    print(f'{r} | {n[:56]}')
