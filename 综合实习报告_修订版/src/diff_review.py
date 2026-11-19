"""逐段对照：列出修订版与原版中相似度低于阈值的段落，便于人工确认改动范围。"""

import difflib
import re
from pathlib import Path

import pymupdf

BASE = Path(__file__).resolve().parent.parent
NEW = BASE / '数字图像处理综合实习报告_修订版.pdf'
OLD = Path(r'C:\Users\001\Downloads\数字图像处理综合实习报告(1).pdf')


def paras(path):
    doc = pymupdf.open(path)
    out = []
    for i, pg in enumerate(doc, 1):
        for blk in pg.get_text('blocks'):
            t = blk[4].strip()
            if not t:
                continue
            if blk[1] < 55 or blk[3] > 790:      # 页眉页脚
                continue
            t = re.sub(r'\s+', '', t)
            if len(t) < 8:
                continue
            out.append((i, blk[1], t))
    return out


old = paras(OLD)
new = paras(NEW)
report = []
for pno, y, t in new:
    best = max(old, key=lambda o: difflib.SequenceMatcher(None, o[2], t).ratio())
    r = difflib.SequenceMatcher(None, best[2], t).ratio()
    if r < 0.985:
        report.append((pno, round(y), round(r, 3), best[2], t))

lines = []
for pno, y, r, o, n in report:
    lines.append(f'--- 新第 {pno} 页 y={y} 相似度={r}')
    lines.append('旧: ' + o)
    lines.append('新: ' + n)
    lines.append('')
(BASE / '_work' / 'paragraph_diff.txt').write_text('\n'.join(lines), encoding='utf-8')
print('changed paragraphs:', len(report))
for pno, y, r, o, n in report:
    print(f'p{pno} y={y} r={r} | 新: {n[:60]}')
