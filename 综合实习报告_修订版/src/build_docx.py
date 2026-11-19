"""渲染修订版 DOCX：与 PDF 同源，按页分节排版，三线表、页眉页脚域、图表光栅化。

页面参数与 PDF 完全对齐：A4、左右 57pt、上 63.89pt、下 58pt（版心 481.276pt），
正文宋体 11.5pt 精确行距 20.2pt，首行缩进 23pt，段后 7pt。
"""

import re
import sys
from pathlib import Path

import pymupdf
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt
from reportlab.graphics import renderPDF

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(HERE))
from content_v2 import CW, DATA, PAGES  # noqa: E402
from build_pdf import CHARTS  # noqa: E402

ASSETS = BASE / 'assets'
WORK = BASE / '_work'
TARGET = BASE / '数字图像处理综合实习报告_修订版.docx'

SONG, HEI = '宋体', '黑体'


# ------------------------------------------------------------------ 基础工具
def set_run_font(run, name, size, bold=False, color=None):
    run.font.name = name
    run.font.size = Pt(size)
    run.font.bold = bold
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn('w:rFonts'))
    if rfonts is None:
        rfonts = OxmlElement('w:rFonts')
        rpr.append(rfonts)
    for attr in ('w:ascii', 'w:hAnsi', 'w:eastAsia', 'w:cs'):
        rfonts.set(qn(attr), name)
    if color:
        run.font.color.rgb = color


TAG = re.compile(r'(</?b>|</?sub>|</?super>)')


def add_runs(p, text, font=SONG, size=11.5, color=None):
    """极简标记解析：<b>…</b>、<sub>…</sub>、<super>…</super>。"""
    bold = sub = sup = False
    for piece in TAG.split(text):
        if piece == '<b>':
            bold = True
        elif piece == '</b>':
            bold = False
        elif piece == '<sub>':
            sub = True
        elif piece == '</sub>':
            sub = False
        elif piece == '<super>':
            sup = True
        elif piece == '</super>':
            sup = False
        elif piece:
            run = p.add_run(piece)
            set_run_font(run, HEI if bold else font, size, bold=False, color=color)
            run.font.subscript = sub or None
            run.font.superscript = sup or None


def cjk_flags(p):
    """关闭中英文自动间距与网格对齐，使换行位置与 ReportLab 版一致。"""
    ppr = p._p.get_or_add_pPr()
    anchor = None
    for tag in ('w:spacing', 'w:ind', 'w:jc', 'w:rPr'):
        found = ppr.find(qn(tag))
        if found is not None:
            anchor = found
            break
    for tag in ('w:autoSpaceDE', 'w:autoSpaceDN', 'w:snapToGrid'):
        old = ppr.find(qn(tag))
        if old is not None:
            ppr.remove(old)
        el = OxmlElement(tag)
        el.set(qn('w:val'), '0')
        if anchor is not None:
            anchor.addprevious(el)
        else:
            ppr.append(el)


def fmt(p, size, leading, before=0, after=0, indent=0, align=WD_ALIGN_PARAGRAPH.JUSTIFY,
        first_indent=None, keep_next=False, page_break=False):
    pf = p.paragraph_format
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    pf.line_spacing = Pt(leading)
    pf.left_indent = Pt(indent)
    if first_indent is not None:
        pf.first_line_indent = Pt(first_indent)
    pf.alignment = align
    pf.widow_control = False
    pf.keep_with_next = keep_next
    pf.page_break_before = page_break
    cjk_flags(p)
    return p


# ReportLab 版实际只按行高堆叠、不使用段前段后（Flowable 无该属性），
# 这里保持同样的零间距，才能得到相同的分页。
STYLES = {
    'body': dict(size=11.5, leading=20.2, first_indent=23),
    'plain': dict(size=11.5, leading=20.2, first_indent=0),
    'h1': dict(size=15, leading=22, first_indent=0, align=WD_ALIGN_PARAGRAPH.LEFT),
    'h2': dict(size=11.8, leading=18, first_indent=0, align=WD_ALIGN_PARAGRAPH.LEFT),
    'cap': dict(size=9.1, leading=13.6, first_indent=0, align=WD_ALIGN_PARAGRAPH.CENTER),
    'note': dict(size=9, leading=14, first_indent=0),
    'eq': dict(size=10.7, leading=20, first_indent=0, align=WD_ALIGN_PARAGRAPH.CENTER),
    'ref': dict(size=9.6, leading=16.2, first_indent=-20, indent=20, align=WD_ALIGN_PARAGRAPH.LEFT),
}


def add_block_para(doc, kind, text, page_break=False):
    st = dict(STYLES[kind])
    p = doc.add_paragraph()
    fmt(p, st['size'], st['leading'], st.get('before', 0), st.get('after', 0),
        st.get('indent', 0), st.get('align', WD_ALIGN_PARAGRAPH.JUSTIFY), st.get('first_indent', 0),
        keep_next=(kind == 'cap'), page_break=page_break)
    if kind == 'eq':
        # 与原版一致：整行居中，编号前的全角空格保留，使编号落在公式右侧
        add_runs(p, text, SONG, st['size'])
    else:
        font = HEI if kind in ('h1', 'h2') else SONG
        add_runs(p, text, font, st['size'])
    return p


# ------------------------------------------------------------------ 三线表
def cell_border(cell, edge, sz_eighth, color='202020'):
    tcpr = cell._tc.get_or_add_tcPr()
    borders = tcpr.find(qn('w:tcBorders'))
    if borders is None:
        borders = OxmlElement('w:tcBorders')
        tcpr.append(borders)
    e = OxmlElement('w:' + edge)
    e.set(qn('w:val'), 'single')
    e.set(qn('w:sz'), str(sz_eighth))
    e.set(qn('w:space'), '0')
    e.set(qn('w:color'), color)
    borders.append(e)


def add_table(doc, head, rows, widths, small=False):
    t = doc.add_table(rows=1 + len(rows), cols=len(head))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    tblpr = t._tbl.tblPr

    borders = OxmlElement('w:tblBorders')
    for edge in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
        e = OxmlElement('w:' + edge)
        e.set(qn('w:val'), 'none')
        e.set(qn('w:sz'), '0')
        borders.append(e)
    tblpr.append(borders)

    layout = OxmlElement('w:tblLayout')
    layout.set(qn('w:type'), 'fixed')
    tblpr.append(layout)

    tw = OxmlElement('w:tblW')
    tw.set(qn('w:w'), str(round(sum(widths) * 20)))
    tw.set(qn('w:type'), 'dxa')
    tblpr.append(tw)

    ind = OxmlElement('w:tblInd')
    ind.set(qn('w:w'), '0')
    ind.set(qn('w:type'), 'dxa')
    tblpr.append(ind)

    mar = OxmlElement('w:tblCellMar')
    pad = 70 if small else 100
    for edge, val in (('top', pad), ('bottom', pad), ('left', 100), ('right', 100)):
        m = OxmlElement('w:' + edge)
        m.set(qn('w:w'), str(val))
        m.set(qn('w:type'), 'dxa')
        mar.append(m)
    tblpr.append(mar)

    for i, w in enumerate(widths):
        t.columns[i].width = Pt(w)
        t._tbl.tblGrid.gridCol_lst[i].set(qn('w:w'), str(round(w * 20)))

    for r_idx, row in enumerate([head] + rows):
        row_obj = t.rows[r_idx]
        row_obj.cantSplit = True
        for c_idx, value in enumerate(row):
            cell = row_obj.cells[c_idx]
            cell.width = Pt(widths[c_idx])
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            p = cell.paragraphs[0]
            fmt(p, 9.2 if r_idx == 0 else 9.1, 13.7 if r_idx else 14,
                0, 0, 0,
                WD_ALIGN_PARAGRAPH.CENTER if r_idx == 0 else WD_ALIGN_PARAGRAPH.LEFT,
                0)
            lines = str(value).split('\n')
            add_runs(p, lines[0], HEI if r_idx == 0 else SONG, 9.2 if r_idx == 0 else 9.1)
            for extra in lines[1:]:
                r = p.add_run()
                r.add_break()
                add_runs(p, extra, HEI if r_idx == 0 else SONG, 9.2 if r_idx == 0 else 9.1)

    for cell in t.rows[0].cells:
        cell_border(cell, 'top', 7)
        cell_border(cell, 'bottom', 4)
    for cell in t.rows[-1].cells:
        cell_border(cell, 'bottom', 7)
    return t


# ------------------------------------------------------------------ 图片与图表
def chart_png(name):
    drawing = CHARTS[name]().drawing
    pdf_path = WORK / f'chart_{name}.pdf'
    png_path = WORK / f'chart_{name}.png'
    renderPDF.drawToFile(drawing, str(pdf_path))
    pix = pymupdf.open(str(pdf_path))[0].get_pixmap(matrix=pymupdf.Matrix(4, 4), alpha=False)
    pix.save(str(png_path))
    return png_path, drawing.width, drawing.height


def add_picture(doc, path, width, height):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(0)
    pf.space_after = Pt(0)
    pf.line_spacing = 1.0
    pf.first_line_indent = Pt(0)
    pf.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf.keep_with_next = True
    pf.widow_control = False
    p.add_run().add_picture(str(path), width=Pt(width), height=Pt(height))
    return p


# ------------------------------------------------------------------ 页眉页脚
def add_field(p, instr, cached, size=9, color=None):
    def fld(kind):
        r = p.add_run()
        el = OxmlElement('w:fldChar')
        el.set(qn('w:fldCharType'), kind)
        r._r.append(el)
        set_run_font(r, SONG, size, color=color)

    fld('begin')
    r = p.add_run()
    it = OxmlElement('w:instrText')
    it.set(qn('xml:space'), 'preserve')
    it.text = f' {instr} '
    r._r.append(it)
    set_run_font(r, SONG, size, color=color)
    fld('separate')
    r = p.add_run(cached)
    set_run_font(r, SONG, size, color=color)
    fld('end')


def build_header(section, doc):
    section.header.is_linked_to_previous = False
    for name in ('Header', 'header'):
        try:
            doc.styles[name].paragraph_format.tab_stops.clear_all()
        except KeyError:
            pass
    hp = section.header.paragraphs[0]
    fmt(hp, 8.4, 12, 0, 0, 0, WD_ALIGN_PARAGRAPH.LEFT, 0)
    hp.paragraph_format.tab_stops.clear_all()
    hp.paragraph_format.tab_stops.add_tab_stop(Pt(CW), WD_TAB_ALIGNMENT.RIGHT)
    add_runs(hp, '数字图像处理综合实习报告', SONG, 8.4, color=None)
    add_runs(hp, '\t图像增强 · 几何校正 · 图像压缩', SONG, 8.4)
    pbdr = OxmlElement('w:pBdr')
    bottom = OxmlElement('w:bottom')
    for k, v in (('w:val', 'single'), ('w:sz', '3'), ('w:space', '1'), ('w:color', 'AAAAAA')):
        bottom.set(qn(k), v)
    pbdr.append(bottom)
    hp._p.get_or_add_pPr().append(pbdr)


def build_footer(section):
    section.footer.is_linked_to_previous = False
    fp = section.footer.paragraphs[0]
    fmt(fp, 9, 12, 0, 0, 0, WD_ALIGN_PARAGRAPH.CENTER, 0)
    add_runs(fp, '第 ', SONG, 9)
    add_field(fp, 'PAGE', '1', 9)
    add_runs(fp, ' 页 / 共 ', SONG, 9)
    add_field(fp, 'SECTIONPAGES', '15', 9)
    add_runs(fp, ' 页', SONG, 9)


# ------------------------------------------------------------------ 封面
def build_cover(doc):
    def blank(height):
        p = doc.add_paragraph()
        fmt(p, 11, height, 0, 0, 0, WD_ALIGN_PARAGRAPH.CENTER, 0)
        return p

    blank(46)
    p = doc.add_paragraph()
    fmt(p, 19, 24, 0, 0, 0, WD_ALIGN_PARAGRAPH.CENTER, 0)
    add_runs(p, '数字图像处理课程', HEI, 19)

    p = doc.add_paragraph()
    fmt(p, 29, 36, 46, 0, 0, WD_ALIGN_PARAGRAPH.CENTER, 0)
    add_runs(p, '综合实习报告', HEI, 29)

    p = doc.add_paragraph()
    fmt(p, 10, 14, 10, 0, 0, WD_ALIGN_PARAGRAPH.CENTER, 0)
    pbdr = OxmlElement('w:pBdr')
    bottom = OxmlElement('w:bottom')
    for k, v in (('w:val', 'single'), ('w:sz', '6'), ('w:space', '1'), ('w:color', '595959')):
        bottom.set(qn(k), v)
    pbdr.append(bottom)
    p._p.get_or_add_pPr().append(pbdr)
    p.paragraph_format.left_indent = Pt(121)
    p.paragraph_format.right_indent = Pt(121)

    p = doc.add_paragraph()
    fmt(p, 14, 20, 19, 0, 0, WD_ALIGN_PARAGRAPH.CENTER, 0)
    add_runs(p, '图像增强 · 文档几何校正 · 图像压缩', SONG, 14)

    blank(111)
    for label, value in [('姓　　名', '陈善樟'), ('学　　号', '2025302132090'),
                         ('专业班级', '2505'), ('指导教师', '易尧华')]:
        p = doc.add_paragraph()
        fmt(p, 12, 28, 0, 19, 0, WD_ALIGN_PARAGRAPH.LEFT, 0)
        add_runs(p, label + '：', SONG, 12)
        r = p.add_run(value)
        set_run_font(r, SONG, 12)
        r.font.underline = True

    blank(75)
    p = doc.add_paragraph()
    fmt(p, 11, 16, 0, 0, 0, WD_ALIGN_PARAGRAPH.CENTER, 0)
    add_runs(p, '报告整理日期：2026 年 10 月 24 日', SONG, 11)


# ------------------------------------------------------------------ 主流程
def build():
    doc = Document()
    for p in list(doc.paragraphs):
        p._element.getparent().remove(p._element)

    sec = doc.sections[0]
    sec.page_width = Pt(595.276)
    sec.page_height = Pt(841.89)
    sec.left_margin = sec.right_margin = Pt(57)
    sec.top_margin = Pt(63.89)
    sec.bottom_margin = Pt(58)
    sec.header_distance = Pt(28)
    sec.footer_distance = Pt(30)

    style = doc.styles['Normal']
    style.font.name = SONG
    style.font.size = Pt(11.5)
    rpr = style.element.get_or_add_rPr()
    rfonts = rpr.find(qn('w:rFonts'))
    if rfonts is None:
        rfonts = OxmlElement('w:rFonts')
        rpr.append(rfonts)
    for attr in ('w:ascii', 'w:hAnsi', 'w:eastAsia', 'w:cs'):
        rfonts.set(qn(attr), SONG)

    build_cover(doc)

    body = doc.add_section(WD_SECTION.NEW_PAGE)
    body.page_width = Pt(595.276)
    body.page_height = Pt(841.89)
    body.left_margin = body.right_margin = Pt(57)
    body.top_margin = Pt(63.89)
    body.bottom_margin = Pt(58)
    body.header_distance = Pt(28)
    body.footer_distance = Pt(30)
    pgnum = OxmlElement('w:pgNumType')
    pgnum.set(qn('w:start'), '1')
    body._sectPr.append(pgnum)
    build_header(body, doc)
    build_footer(body)

    for idx, page in enumerate(PAGES):
        add_block_para(doc, 'h1', page['title'], page_break=idx > 0)
        for blk in page['blocks']:
            k = blk['k']
            if k == 'p':
                add_block_para(doc, blk.get('style', 'body'), blk['t'])
            elif k == 'h1':
                add_block_para(doc, 'h1', blk['t'])
            elif k == 'h2':
                add_block_para(doc, 'h2', blk['t'])
            elif k == 'eq':
                add_block_para(doc, 'eq', blk['t'])
            elif k == 'cap':
                add_block_para(doc, 'cap', blk['t'])
            elif k == 'note':
                add_block_para(doc, 'note', blk['t'])
            elif k == 'table':
                add_table(doc, blk['head'], blk['rows'], blk['widths'], blk.get('small', False))
            elif k == 'pic':
                add_picture(doc, ASSETS / blk['name'], blk['w'], blk['h'])
            elif k == 'chart':
                png, w, h = chart_png(blk['name'])
                add_picture(doc, png, w, h)
            elif k == 'spacer':
                p = doc.add_paragraph()
                fmt(p, 10, blk['h'], 0, 0, 0, WD_ALIGN_PARAGRAPH.LEFT, 0)

    settings = doc.settings.element
    upd = OxmlElement('w:updateFields')
    upd.set(qn('w:val'), 'true')
    settings.append(upd)

    doc.core_properties.title = '数字图像处理综合实习报告'
    doc.core_properties.subject = '极暗RAW增强、文档几何校正与图像压缩的实现及评价'
    doc.core_properties.author = ''
    doc.save(str(TARGET))
    print('Created', TARGET)


if __name__ == '__main__':
    build()
