from pathlib import Path
import json, sys
from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT = Path(r'D:\数字图像处理实习作业'); OUT = ROOT/'output'; ASSETS=OUT/'report_assets'
DATA=json.loads((OUT/'verified_metrics.json').read_text(encoding='utf-8'))
TARGET=OUT/'docx/数字图像处理综合实习报告_修订版.docx'; TARGET.parent.mkdir(parents=True,exist_ok=True)
doc=Document(); sec=doc.sections[0]
sec.top_margin=Cm(2.4); sec.bottom_margin=Cm(2.2); sec.left_margin=Cm(2.6); sec.right_margin=Cm(2.6); sec.header_distance=Cm(1.2); sec.footer_distance=Cm(1.2)
styles=doc.styles
styles['Normal'].font.name='宋体'; styles['Normal']._element.rPr.rFonts.set(qn('w:eastAsia'),'宋体'); styles['Normal'].font.size=Pt(10.5); styles['Normal'].paragraph_format.line_spacing=1.45; styles['Normal'].paragraph_format.space_after=Pt(6)
for name,size,bold,before,after in [('Title',26,True,0,16),('Heading 1',16,True,12,8),('Heading 2',12,True,8,5)]:
    st=styles[name]; st.font.name='黑体'; st._element.rPr.rFonts.set(qn('w:eastAsia'),'黑体'); st.font.size=Pt(size); st.font.bold=bold; st.font.color.rgb=RGBColor(0,0,0); st.paragraph_format.space_before=Pt(before); st.paragraph_format.space_after=Pt(after)
styles['Title'].paragraph_format.alignment=WD_ALIGN_PARAGRAPH.CENTER
st=styles['Caption']; st.font.name='宋体'; st._element.rPr.rFonts.set(qn('w:eastAsia'),'宋体'); st.font.size=Pt(9); st.font.color.rgb=RGBColor(60,60,60); st.paragraph_format.alignment=WD_ALIGN_PARAGRAPH.CENTER; st.paragraph_format.space_before=Pt(2); st.paragraph_format.space_after=Pt(7)

def shade(cell,fill):
    tcPr=cell._tc.get_or_add_tcPr(); shd=tcPr.find(qn('w:shd'))
    if shd is None: shd=OxmlElement('w:shd'); tcPr.append(shd)
    shd.set(qn('w:fill'),fill)

def borders(cell):
    tcPr=cell._tc.get_or_add_tcPr(); b=tcPr.first_child_found_in('w:tcBorders')
    if b is None: b=OxmlElement('w:tcBorders'); tcPr.append(b)
    for edge in ('top','left','bottom','right'):
        e=OxmlElement('w:'+edge); e.set(qn('w:val'),'single'); e.set(qn('w:sz'),'4'); e.set(qn('w:color'),'B7C3D0'); b.append(e)

def p(text='',style='Normal',align=None):
    q=doc.add_paragraph(style=style)
    if align is not None:q.alignment=align
    q.add_run(text); return q

def h(text,level=1): return doc.add_heading(text,level=level)
def br(): doc.add_page_break()
def cap(text): q=doc.add_paragraph(style='Caption'); q.add_run(text); return q

def table(headers,rows):
    t=doc.add_table(rows=1,cols=len(headers)); t.alignment=WD_TABLE_ALIGNMENT.CENTER; t.style='Table Grid'
    for i,x in enumerate(headers):
        c=t.rows[0].cells[i]; c.text=str(x); shade(c,'D9E2F3'); borders(c); c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
        for r in c.paragraphs[0].runs: r.bold=True; r.font.name='黑体'; r._element.rPr.rFonts.set(qn('w:eastAsia'),'黑体'); r.font.size=Pt(9)
    for ri,row in enumerate(rows):
        cells=t.add_row().cells
        for i,x in enumerate(row):
            c=cells[i]; c.text=str(x); borders(c); c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            if ri%2: shade(c,'F5F7FA')
            for q in c.paragraphs:
                q.paragraph_format.space_after=Pt(1)
                for r in q.runs: r.font.name='宋体'; r._element.rPr.rFonts.set(qn('w:eastAsia'),'宋体'); r.font.size=Pt(8.5)
    doc.add_paragraph().paragraph_format.space_after=Pt(1); return t

def img(name,width=6.15,maxheight=None):
    q=doc.add_paragraph(); q.alignment=WD_ALIGN_PARAGRAPH.CENTER; pic=q.add_run().add_picture(str(ASSETS/name),width=Inches(width))
    if maxheight and pic.height.inches>maxheight: pic.width=int(pic.width*maxheight/pic.height); pic.height=Inches(maxheight)
    q.paragraph_format.space_after=Pt(2)

def header_footer():
    q=sec.header.paragraphs[0]; q.alignment=WD_ALIGN_PARAGRAPH.RIGHT; r=q.add_run('数字图像处理综合实习报告'); r.font.name='宋体'; r._element.rPr.rFonts.set(qn('w:eastAsia'),'宋体'); r.font.size=Pt(8); r.font.color.rgb=RGBColor(145,145,145)
    q=sec.footer.paragraphs[0]; q.alignment=WD_ALIGN_PARAGRAPH.CENTER; r=q.add_run('第 '); r.font.size=Pt(8); fld=OxmlElement('w:fldSimple'); fld.set(qn('w:instr'),'PAGE'); q._p.append(fld); r=q.add_run(' 页'); r.font.size=Pt(8)
header_footer()

# cover
for _ in range(3): p('')
r=p('数字图像处理课程','Title',WD_ALIGN_PARAGRAPH.CENTER); r.runs[0].font.size=Pt(18)
r=p('综合实习报告','Title',WD_ALIGN_PARAGRAPH.CENTER); r.runs[0].font.size=Pt(25)
p('图像增强　·　文档几何校正　·　图像压缩',align=WD_ALIGN_PARAGRAPH.CENTER)
for _ in range(4): p('')
for x in ['姓　　名：','学　　号：','专业班级：','指导教师：']: p(x+'　　　　　　　　　　　　　　　',align=WD_ALIGN_PARAGRAPH.CENTER)
for _ in range(3): p('')
p('报告整理日期：2026 年 10 月 24 日',align=WD_ALIGN_PARAGRAPH.CENTER); br()

# abstract
h('摘　要',1)
p('本实习围绕极暗 RAW 图像增强、文档几何校正与彩色图像压缩三个任务展开，分别完成了从数据预处理、模型或算法处理到定量评价和视觉对照的完整流程。增强任务使用 Canon 极暗 RAW 与正常曝光参考图像 1000 对，几何校正任务处理 200 幅文档图像，压缩任务在 Kodak 24 幅图像上完成两类编码器、十组质量配置的对比。')
p('图像增强最终结果在测试集上取得 PSNR 25.318 dB、SSIM 0.8065 和 LPIPS 0.3728，相对传统基线分别提升 7.365 dB、0.2437，并降低 LPIPS 0.2054。几何校正中，UVDoc 结果的平均 OCR 输出字符数达到 798.89，平均识别置信度达到 0.7769，页面边界和书脊、折叠区域的视觉规整效果更加明显。压缩实验中，AVIF Q50 的平均码率为 0.6216 bpp、PSNR 为 33.0699 dB；自研 DCT Q50 的平均码率为 1.0280 bpp、PSNR 为 32.6949 dB。')
p('关键词：数字图像处理；低照度增强；文档校正；图像压缩；结果评价')
h('目　录',1)
table(['内容','页码'],[['1 实习任务与评价设计','3'],['2 数据与评价指标','3'],['3 极暗 RAW 图像增强','4'],['4 文档几何校正','8'],['5 图像压缩与率失真分析','11'],['6 综合讨论','14'],['7 实习结论','15'],['参考文献','16']]); br()

# chapter 1/2
h('1　实习任务与评价设计',1)
p('本次实习包含图像增强、文档几何校正和图像压缩三个任务。三个任务分别面向亮度与颜色恢复、空间结构重排和信息编码表示，均按照“处理方法—结果保存—指标计算—可视化检查”的步骤完成。报告中的数值来自三个最终作业目录中保存的逐图结果。')
table(['任务','数据规模','处理与对照'],[['图像增强','1000 对，3840×2160','Canon 极暗 RAW、传统基线、最终增强、正常曝光参考'],['几何校正','200 幅文档','原图、四角透视基线、UVDoc 校正结果'],['图像压缩','24 幅 RGB 图像','自研 DCT 与 AVIF，各 5 档质量配置']])
h('2　数据与评价指标',1)
p('增强结果分别保存为最终模型和传统基线的逐图 CSV，共 1000 条记录；数据划分为训练、验证和测试三部分。几何校正使用 OCR 输出统计和页面规整度统计评价结果。压缩实验保存了 24 幅图像在 10 组配置下的完整记录，共 240 条。')
p('增强和压缩采用 PSNR、SSIM 等有参考指标，结合逐图视觉结果进行分析；几何校正重点观察页面边界、文字行形态和 OCR 输出。所有均值均由逐图结果计算得到。')
h('2.1　评价指标',2); p('PSNR 用于衡量重建图像与参考图像之间的像素误差，SSIM 关注亮度、对比度和结构相似性，LPIPS 用于补充感知层面的差异评价。压缩实验还统计码率、压缩比以及编码和解码耗时。'); br()

# enhancement
h('3　极暗 RAW 图像增强',1); h('3.1　数据准备与处理流程',2)
p('增强任务使用 Canon 极暗 RAW 与对应正常曝光参考图像共 1000 对。RAW 数据经过黑电平处理、Bayer 四相位打包和训练集标定的颜色映射后输入四级残差 U-Net。网络输出通过 PixelShuffle 重建全尺寸 RGB，并结合全图上下文和细节分支完成曝光、颜色与纹理恢复。')
p('训练采用 256×256 packed RAW 裁块、AdamW、混合精度和梯度裁剪。最终选用 adaptive 阶段的 24000 步模型，测试集仅用于模型冻结后的最终评价。')
h('3.2　定量结果',2)
e=DATA['enh']; rows=[]
for split,label in [('all','全量'),('train','训练'),('val','验证'),('test','测试')]:
    for tag,name in [('baseline','传统基线'),('final','最终增强')]:
        r=e[tag][split]; rows.append([f'{label}/{name}',r['count'],f"{r['psnr']:.3f}",f"{r['ssim']:.4f}",f"{r['lpips']:.4f}"])
table(['范围/方法','张数','PSNR/dB','SSIM','LPIPS'],rows)
p('测试集上，最终增强相对传统基线的 PSNR 提升 7.365 dB，SSIM 提升 0.2437，LPIPS 降低 0.2054，三个指标均呈现出一致的质量改善。全量 1000 幅图像的最终平均 PSNR、SSIM、LPIPS 分别为 23.742 dB、0.7940、0.3964。')
h('3.3　典型结果对照',2)
p('图3-1至图3-3展示训练集场景中的商品、招牌和自行车，最终结果在主体颜色、整体曝光和局部结构方面均较传统基线更接近正常曝光参考。')
img('enh_10.jpg'); cap('图3-1　商品货架场景：传统基线、最终增强与参考图像对照')
img('enh_100.jpg'); cap('图3-2　红色招牌场景：色彩方向和文字边缘对照'); br()
img('enh_798.jpg'); cap('图3-3　蓝色自行车场景：物体颜色与细节对照')
img('enh_518.jpg'); cap('图3-4　测试集路面场景：亮度与结构恢复对照')
img('enh_435.jpg'); cap('图3-5　测试集蓝色篷布场景：整体色彩与曝光对照')
h('3.4　小结',2); p('图像增强完成了从极暗 RAW 到彩色 sRGB 的完整处理链路。定量指标显示最终模型在测试集上相对传统基线取得明显改善，视觉结果进一步体现了颜色、曝光和主体结构恢复效果。'); br()

# geometry
h('4　文档几何校正',1); h('4.1　处理方法',2)
p('几何校正使用 200 幅文档图像及配套掩膜。第一条路线从文档轮廓提取四角，建立四边形到矩形的透视变换；第二条路线使用 UVDoc 预训练模型预测稠密采样网格，对页面的局部形变进行校正。两种方法均按图像标识保存对应输出。')
h('4.2　全量评价结果',2)
g=DATA['geo']; rows=[]
for tag,label in [('original','原图'),('baseline','透视基线'),('uvdoc','UVDoc')]:
    r=g[tag]; rows.append([label,f"{r['curv']:.4f}",f"{r['lv']:.5f}",f"{r['chars']:.2f}",f"{r['lines']:.2f}",f"{r['conf']:.4f}"])
table(['方法','curvature','line_var','字符数','识别框数','置信度'],rows)
p('UVDoc 平均 OCR 输出字符数为 798.89，较原图 565.14 增加 41.36%；平均识别框数为 46.98，较原图 37.61 增加 24.91%；平均识别置信度由 0.7533 提升至 0.7769。透视基线同样提高了字符数、识别框数和平均置信度。')
h('4.3　典型结果对照',2)
img('geo_150.jpg',maxheight=1.6); cap('图4-1　书页正文：页面边界与靠近书脊区域对照')
img('geo_174.jpg',maxheight=1.6); cap('图4-2　双页教材：四角透视与稠密网格校正对照')
img('geo_180.jpg',maxheight=1.6); cap('图4-3　带标记教材页面：正文行与页边形态对照'); br()
img('geo_dewarp20260728_curved_029.jpg',maxheight=2.4); cap('图4-4　手写折叠纸张：弯曲和折痕区域校正对照')
h('4.4　小结',2); p('几何校正完成了透视基线和 UVDoc 两条处理路线。视觉结果显示，UVDoc 对书脊、折叠和局部起伏区域具有更强的形变适应能力；OCR 统计也表明校正后文本输出数量和平均置信度均得到提升。'); br()

# compression
h('5　图像压缩与率失真分析',1); h('5.1　实验配置',2)
p('压缩实验使用 Kodak 24 幅 24-bit RGB 图像。自研 DCT 编码器采用质量因子 10、30、50、75、90；AVIF 采用质量因子 10、30、50、70、90。每幅图像完成全部配置，形成 240 条评价记录。')
p('自研 DCT 编码器依次完成 RGB 到 YCbCr、8×8 分块 DCT、量化、Zigzag 扫描、DC 差分、AC 游程和哈夫曼编码，并执行对应的解码重建；AVIF 通过 libavif 完成编码和解码。')
h('5.2　平均结果',2)
rows=[]
for r in DATA['compression']:
    fam,q=r['codec'].rsplit('-q',1); rows.append([('DCT' if fam=='jpeg-dct' else 'AVIF')+' '+q,f"{r['bpp']:.4f}",f"{r['psnr']:.4f}",f"{r['ssim']:.5f}",f"{r['compression_ratio']:.2f}"])
table(['方法/Q','bpp','PSNR/dB','SSIM','CR/倍'],rows)
p('同一编码器内，随着质量因子提高，平均码率、PSNR 和 SSIM 均同步上升，体现出清晰的率失真变化。Q50 配置下，AVIF 的平均码率为 0.6216 bpp、PSNR 为 33.0699 dB；DCT 的平均码率为 1.0280 bpp、PSNR 为 32.6949 dB。')
h('5.3　典型结果对照',2)
img('compression_01.jpg',maxheight=2.4); cap('图5-1　石墙与窗框：细密砖缝及边缘重建对照')
img('compression_05.jpg',maxheight=2.4); cap('图5-2　摩托车场景：复杂纹理与高对比细节重建对照'); br()
img('compression_23.jpg',maxheight=2.4); cap('图5-3　鹦鹉场景：平滑颜色区域与羽毛纹理重建对照')
h('5.4　小结',2); p('压缩实验完整验证了自研 DCT 编码器的变换、量化和熵编码流程。平均率失真结果表明，AVIF 在本次测量配置下以较低码率取得了较高的平均 PSNR 和 SSIM；自研 DCT 则呈现出质量因子与码率、重建质量同步变化的清晰规律。'); br()

# conclusion
h('6　综合讨论',1)
p('三个任务分别从恢复、重采样和编码角度处理图像信息。增强模型通过 RAW 四相位输入和残差 U-Net 恢复颜色、曝光与结构；几何校正通过透视模型和稠密采样网格重排页面像素；压缩算法通过变换、量化和熵编码控制码率。三项作业均形成了较完整的处理、评价与可视化闭环。')
table(['任务','主要结果'],[['图像增强','测试集 PSNR 提升 7.365 dB，SSIM 提升 0.2437，LPIPS 降低 0.2054；颜色、曝光和主体结构恢复明显。'],['几何校正','UVDoc 平均字符数 798.89，平均置信度 0.7769；局部页面形变和书脊、折痕区域得到改善。'],['图像压缩','AVIF Q50 达到 0.6216 bpp、33.0699 dB；DCT 完成完整的 DCT、量化和熵编码流程。']])
h('7　实习结论',1)
p('本次实习完成了三项数字图像处理任务的最终实现和综合评价。图像增强结果在测试集上相对传统基线取得明确指标提升；几何校正结果在页面规整和 OCR 输出方面表现出较好的处理效果；图像压缩实验验证了自研 DCT 编码链路，并给出了与 AVIF 的率失真对照。')
p('从结果看，算法结构、处理参数和评价指标形成了相互配合的实验体系。增强任务中，RAW 四相位输入和 adaptive 模型有助于颜色与曝光恢复；几何校正中，稠密网格能够处理透视模型之外的局部形变；压缩任务中，变换域表示与熵编码共同决定码率和重建质量。三项实习均达到了预期的处理与分析目标。'); br()

h('参考文献',1)
for x in ['[1] 武汉大学. 数字图像处理课程课间实习任务书[Z]. 2026-09-30.','[2] WANG Z, BOVIK A C, SHEIKH H R, et al. Image quality assessment: From error visibility to structural similarity[J]. IEEE Transactions on Image Processing, 2004, 13(4): 600-612.','[3] ZHANG R, ISOLA P, EFROS A A, et al. The unreasonable effectiveness of deep features as a perceptual metric[C]. CVPR, 2018: 586-595.','[4] VERHOEVEN F, MAGNE T, SORKINE-HORNUNG O. UVDoc: Neural grid-based document unwarping[C]. SIGGRAPH Asia, 2023.']: p(x)
h('数据依据',1); p('图像增强数据与逐图指标来自“图像增强2”最终目录；几何校正数据、UVDoc 输出和 OCR 统计来自“几何校正workbuddy2”最终目录；压缩指标和对比结果来自“图像压缩”最终目录。报告中的图像均由已保存结果整理生成。')
doc.core_properties.title='数字图像处理综合实习报告 修订版'; doc.core_properties.author=''; doc.save(TARGET); print(TARGET)
