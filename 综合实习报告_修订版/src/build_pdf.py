"""渲染修订版 PDF：复用原报告的版式常量与图表绘制，只更换内容源。"""

import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib import textsplit
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.graphics import renderPDF
from reportlab.graphics.shapes import Circle, Drawing, Line, PolyLine, Rect, String
from reportlab.platypus import Flowable, Image, Paragraph, Spacer, Table, TableStyle

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(HERE))
from content_v2 import CW, DATA, PAGES  # noqa: E402

ASSETS = BASE / 'assets'
WORK = BASE / '_work'
TARGET = BASE / '数字图像处理综合实习报告_修订版.pdf'

pdfmetrics.registerFont(TTFont('Song', r'C:\Windows\Fonts\simsun.ttc', subfontIndex=0))
pdfmetrics.registerFont(TTFont('Hei', r'C:\Windows\Fonts\simhei.ttf'))
pdfmetrics.registerFontFamily('Song', normal='Song', bold='Hei', italic='Song', boldItalic='Hei')

W, H = 595.276, 841.89
MARGIN = 57
TOP, BOTTOM = 778, 58
INK = colors.HexColor('#202020')
BLUE = colors.HexColor('#315c7e')
ORANGE = colors.HexColor('#b86b42')
GRAY = colors.HexColor('#8a929b')
LIGHT = colors.HexColor('#d9dde1')

styles = {
    'body': ParagraphStyle('body', fontName='Song', fontSize=11.5, leading=20.2, firstLineIndent=23, alignment=TA_JUSTIFY, wordWrap='CJK', spaceAfter=7, textColor=INK),
    'plain': ParagraphStyle('plain', fontName='Song', fontSize=11.5, leading=20.2, wordWrap='CJK', spaceAfter=7, textColor=INK),
    'h1': ParagraphStyle('h1', fontName='Hei', fontSize=15, leading=22, spaceBefore=0, spaceAfter=15, textColor=INK),
    'h2': ParagraphStyle('h2', fontName='Hei', fontSize=11.8, leading=18, spaceBefore=7, spaceAfter=7, textColor=INK),
    'cap': ParagraphStyle('cap', fontName='Song', fontSize=9.1, leading=13.6, alignment=TA_CENTER, wordWrap='CJK', spaceBefore=4, spaceAfter=9, textColor=INK),
    'note': ParagraphStyle('note', fontName='Song', fontSize=9, leading=14, wordWrap='CJK', spaceBefore=4, spaceAfter=7, textColor=colors.HexColor('#414141')),
    'cell': ParagraphStyle('cell', fontName='Song', fontSize=9.1, leading=13.7, wordWrap='CJK', textColor=INK),
    'headcell': ParagraphStyle('headcell', fontName='Hei', fontSize=9.2, leading=14, alignment=TA_CENTER, wordWrap='CJK', textColor=INK),
    'eq': ParagraphStyle('eq', fontName='Song', fontSize=10.7, leading=20, alignment=TA_CENTER, spaceBefore=4, spaceAfter=10, wordWrap='CJK'),
    'ref': ParagraphStyle('ref', fontName='Song', fontSize=9.6, leading=16.2, wordWrap='CJK', spaceAfter=12, leftIndent=20, firstLineIndent=-20),
}

try:
    textsplit.ALL_CANNOT_START += '，；：！？）》】”’'
except Exception:
    pass


def p(t, kind='body'):
    return Paragraph(t, styles[kind])


def table(headers, rows, widths=None, small=False):
    if widths is None:
        widths = [CW / len(headers)] * len(headers)
    data = [[p(str(x), 'headcell') for x in headers]]
    for row in rows:
        data.append([p(str(x), 'cell') for x in row])
    t = Table(data, colWidths=widths, hAlign='CENTER')
    t.setStyle(TableStyle([
        ('LINEABOVE', (0, 0), (-1, 0), 0.85, INK),
        ('LINEBELOW', (0, 0), (-1, 0), 0.5, INK),
        ('LINEBELOW', (0, -1), (-1, -1), 0.85, INK),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 5), ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 5 if not small else 3.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5 if not small else 3.5),
    ]))
    return t


def pic(name, want_w, want_h):
    """与原报告一致：先按版心宽度铺满，超过给定高度上限时按上限等比缩小。"""
    path = ASSETS / name
    with PILImage.open(path) as im:
        iw, ih = im.size
    width, height = CW, CW * ih / iw
    if height > want_h + 0.01:
        width, height = want_h * iw / ih, want_h
    width, height = round(width, 2), round(height, 2)
    if abs(width - want_w) > 0.05 or abs(height - want_h) > 0.05:
        raise SystemExit(f'{name}: 记录尺寸 {want_w}x{want_h} 与实际 {width}x{height} 不符')
    return Image(str(path), width=width, height=height, hAlign='CENTER')


class Graphic(Flowable):
    def __init__(self, drawing):
        Flowable.__init__(self)
        self.drawing = drawing
        self.width = drawing.width
        self.height = drawing.height
        self.hAlign = 'CENTER'

    def draw(self):
        renderPDF.draw(self.drawing, self.canv, 0, 0)


def txt(d, x, y, t, size=8.4, font='Song', color=INK, anchor='start'):
    d.add(String(x, y, t, fontName=font, fontSize=size, fillColor=color, textAnchor=anchor))


def histogram():
    d = Drawing(CW, 174)
    left, bottom, pw, ph = 37, 34, CW - 49, 116
    bins = np.arange(10, 46, 5)
    a = np.histogram(DATA['enh']['baseline']['test']['psnr_values'], bins=bins)[0]
    b = np.histogram(DATA['enh']['final']['test']['psnr_values'], bins=bins)[0]
    ymax = math.ceil(max(max(a), max(b)) / 10) * 10
    for y in range(0, ymax + 1, 10):
        yy = bottom + ph * y / ymax
        d.add(Line(left, yy, left + pw, yy, strokeColor=LIGHT, strokeWidth=.45))
        txt(d, left - 5, yy - 3, str(y), anchor='end')
    group = pw / len(a)
    for i, (va, vb) in enumerate(zip(a, b)):
        x = left + i * group
        for j, (v, col) in enumerate([(va, GRAY), (vb, BLUE)]):
            d.add(Rect(x + group * (.13 + .36 * j), bottom, group * .32, ph * v / ymax, fillColor=col, strokeWidth=0))
            if v:
                txt(d, x + group * (.29 + .36 * j), bottom + ph * v / ymax + 3, str(int(v)), size=7.4, anchor='middle')
        txt(d, x + group / 2, bottom - 14, f'{bins[i]}-{bins[i + 1]}', size=8, anchor='middle')
    txt(d, 4, 154, '张数', size=9)
    txt(d, left + pw, 4, 'PSNR 区间 / dB', anchor='end', size=9)
    for x, col, label in [(270, GRAY, '标定传统基线'), (374, BLUE, '最终增强')]:
        d.add(Rect(x, 158, 9, 6, fillColor=col, strokeWidth=0))
        txt(d, x + 13, 157, label, 8.3)
    return Graphic(d)


def ocrbars():
    d = Drawing(CW, 180)
    specs = [('chars', '平均识别字符数', 1000, 200, False), ('lines', '平均识别框数', 60, 20, False),
             ('conf', '平均识别置信度 / %', 100, 20, True)]
    method = [('original', GRAY, '原图'), ('baseline', BLUE, '透视'), ('uvdoc', ORANGE, 'UVDoc')]
    for i, (key, title, hi, step, percent) in enumerate(specs):
        x0 = i * (CW / 3) + 30
        y0, pw, ph = 32, CW / 3 - 39, 108
        txt(d, x0 + pw / 2, 163, title, 8.8, 'Hei', anchor='middle')
        for v in range(0, hi + 1, step):
            yy = y0 + v / hi * ph
            d.add(Line(x0, yy, x0 + pw, yy, strokeColor=LIGHT, strokeWidth=.4))
            txt(d, x0 - 4, yy - 3, str(v), 7.3, anchor='end')
        for j, (tag, color, label) in enumerate(method):
            v = DATA['geo'][tag][key] * (100 if percent else 1)
            x = x0 + pw * (j + .17) / 3
            bw = pw * .20
            d.add(Rect(x, y0, bw, v / hi * ph, fillColor=color, strokeWidth=0))
            txt(d, x + bw / 2, y0 + v / hi * ph + 5, f'{v:.2f}', 7.7, anchor='middle')
            txt(d, x + bw / 2, 16, label, 8, anchor='middle')
    return Graphic(d)


def rdplots():
    d = Drawing(CW, 204)
    for i, (metric, ylow, yhigh, yticks, title) in enumerate([
            ('psnr', 24, 44, [25, 30, 35, 40], 'PSNR / dB'),
            ('ssim', .65, 1.0, [.7, .8, .9, 1.0], 'SSIM')]):
        left = 34 + i * (CW / 2 + 3)
        bottom, pw, ph = 34, CW / 2 - 46, 135
        for tick in yticks:
            yy = bottom + (tick - ylow) / (yhigh - ylow) * ph
            d.add(Line(left, yy, left + pw, yy, strokeColor=LIGHT, strokeWidth=.4))
            txt(d, left - 5, yy - 3, f'{tick:g}', 7.8, anchor='end')
        for xx in [0, 1, 2, 3]:
            x = left + xx / 3 * pw
            d.add(Line(x, bottom, x, bottom + ph, strokeColor=LIGHT, strokeWidth=.4))
            txt(d, x, bottom - 13, str(xx), 8, anchor='middle')
        for fam, color in [('jpeg-dct', BLUE), ('avif', ORANGE)]:
            rows = [r for r in DATA['compression'] if r['codec'].startswith(fam)]
            points = []
            for r in rows:
                x = left + r['bpp'] / 3 * pw
                y = bottom + (r[metric] - ylow) / (yhigh - ylow) * ph
                points.extend([x, y])
                d.add(Circle(x, y, 2.3, fillColor=color, strokeColor=color))
            d.add(PolyLine(points, strokeColor=color, strokeWidth=1.15))
        txt(d, left, 181, title, 9.2, 'Hei')
        txt(d, left + pw, 5, '码率 / (bit/pixel)', 8.3, anchor='end')
    for x, color, label in [(150, BLUE, '自研 DCT'), (252, ORANGE, 'AVIF')]:
        d.add(Line(x, 198, x + 16, 198, strokeColor=color, strokeWidth=1.4))
        txt(d, x + 21, 195, label, 8.4)
    return Graphic(d)


CHARTS = {'histogram': histogram, 'ocrbars': ocrbars, 'rdplots': rdplots}


def to_flowables(title, blocks):
    out = [p(title, 'h1')]
    for blk in blocks:
        k = blk['k']
        if k == 'p':
            out.append(p(blk['t'], blk.get('style', 'body')))
        elif k == 'h1':
            out.append(p(blk['t'], 'h1'))
        elif k == 'h2':
            out.append(p(blk['t'], 'h2'))
        elif k == 'eq':
            out.append(p(blk['t'], 'eq'))
        elif k == 'cap':
            out.append(p(blk['t'], 'cap'))
        elif k == 'note':
            out.append(p(blk['t'], 'note'))
        elif k == 'table':
            out.append(table(blk['head'], blk['rows'], blk['widths'], blk.get('small', False)))
        elif k == 'pic':
            out.append(pic(blk['name'], blk['w'], blk['h']))
        elif k == 'chart':
            out.append(CHARTS[blk['name']]())
        elif k == 'spacer':
            out.append(Spacer(1, blk['h']))
        else:
            raise SystemExit('未知 block: ' + k)
    return out


def cover(c):
    c.setFillColor(INK)
    c.setFont('Hei', 19)
    c.drawCentredString(W / 2, 716, '数字图像处理课程')
    c.setFont('Hei', 29)
    c.drawCentredString(W / 2, 633, '综合实习报告')
    c.setStrokeColor(colors.HexColor('#595959'))
    c.setLineWidth(.8)
    c.line(121, 603, W - 121, 603)
    c.setFont('Song', 14)
    c.drawCentredString(W / 2, 565, '图像增强 · 文档几何校正 · 图像压缩')
    c.setFont('Song', 12)
    fields = ['姓　　名', '学　　号', '专业班级', '指导教师']
    values = ['陈善樟', '2025302132090', '2505', '易尧华']
    for i, (t, v) in enumerate(zip(fields, values)):
        y = 428 - i * 47
        c.drawString(156, y, t + '：')
        c.line(239, y - 5, 434, y - 5)
        c.setFont('Song', 12)
        c.drawCentredString((239 + 434) / 2, y, v)
    c.setFont('Song', 11)
    c.drawCentredString(W / 2, 174, '报告整理日期：2026 年 10 月 24 日')
    c.showPage()


def build():
    c = canvas.Canvas(str(TARGET), pagesize=(W, H), pageCompression=1)
    c.setTitle('数字图像处理综合实习报告')
    c.setSubject('极暗RAW增强、文档几何校正与图像压缩的实现及评价')
    c.setAuthor('')
    cover(c)
    qa = []
    for number, page in enumerate(PAGES, 1):
        c.bookmarkPage(page['bookmark'])
        c.addOutlineEntry(page['title'], page['bookmark'], level=0)
        c.setFillColor(colors.HexColor('#666666'))
        c.setFont('Song', 8.4)
        c.drawString(MARGIN, 809, '数字图像处理综合实习报告')
        c.drawRightString(W - MARGIN, 809, '图像增强 · 几何校正 · 图像压缩')
        c.setStrokeColor(colors.HexColor('#aaaaaa'))
        c.setLineWidth(.4)
        c.line(MARGIN, 801, W - MARGIN, 801)
        items = to_flowables(page['title'], page['blocks'])
        total = 0
        dims = []
        for f in items:
            fw, fh = f.wrap(CW, 10000)
            before = getattr(f, 'spaceBefore', 0)
            after = getattr(f, 'spaceAfter', 0)
            total += before + fh + after
            dims.append((f, fw, fh, before, after))
        if total > TOP - BOTTOM:
            raise SystemExit(f'第 {number} 页溢出：{total:.1f} > {TOP - BOTTOM:.1f}；{page["title"]}')
        y = TOP
        for f, fw, fh, before, after in dims:
            y -= before + fh
            x = MARGIN + (CW - fw) / 2 if isinstance(f, (Image, Graphic, Table)) else MARGIN
            f.drawOn(c, x, y)
            y -= after
        qa.append({'body_page': number, 'title': page['title'], 'used_height': round(total, 2),
                   'remaining_height': round(y - BOTTOM, 2)})
        c.setFillColor(colors.HexColor('#555555'))
        c.setFont('Song', 9)
        c.drawCentredString(W / 2, 32, f'第 {number} 页 / 共 {len(PAGES)} 页')
        c.showPage()
    c.save()
    (WORK / 'pdf_layout.json').write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Created', TARGET, 'pages', len(PAGES) + 1)
    for r in qa:
        print(f"  P{r['body_page']:02d} used={r['used_height']:7.2f} remain={r['remaining_height']:7.2f}")


if __name__ == '__main__':
    build()
