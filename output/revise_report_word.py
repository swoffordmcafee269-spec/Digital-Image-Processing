from pathlib import Path
import sys, json, re, hashlib, subprocess, importlib.util
from copy import deepcopy
from html import unescape
from lxml import etree
from docx import Document
from docx.shared import Cm, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_TAB_ALIGNMENT, WD_TAB_LEADER
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from reportlab.platypus import Paragraph, Table, Image, Spacer
from reportlab.graphics import renderPDF

sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path(r'D:\数字图像处理实习作业')
OUT = ROOT/'output'
TARGET = OUT/'docx/数字图像处理综合实习报告_修订版.docx'
WORK = OUT/'word_revision_work'
WORK.mkdir(exist_ok=True)
spec=importlib.util.spec_from_file_location('original_report', OUT/'build_report.py')
b=importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
ORIGINAL=OUT/'pdf/数字图像处理综合实习报告.pdf'
original_hash=hashlib.sha256(ORIGINAL.read_bytes()).hexdigest()
changes=[]

# Local editorial replacements. All measurements and existing visual cases remain.
R = [
('本实习围绕极暗 RAW', '本实习围绕极暗 RAW 图像增强、文档几何校正与彩色图像压缩三个任务展开，完成数据预处理、算法实现、逐图评价与可视化分析。增强任务使用 Canon 极暗 RAW 与正常曝光参考图像 1000 对，以训练集标定的传统方法为对照；几何校正任务对 200 幅文档比较四角透视变换与 UVDoc 预训练模型；压缩任务在 Kodak 24 幅图像上比较自研 DCT 编码器与 AVIF 的十组配置。'),
('增强最终结果的全量平均', '增强最终结果的全量平均 PSNR、SSIM 和 LPIPS 分别为 23.742 dB、0.7940 和 0.3964；100 幅测试图像分别为 25.318 dB、0.8065 和 0.3728，相对传统基线三项指标均有改善。几何校正中，UVDoc 的平均 OCR 输出字符数由原图的 565.14 增至 798.89，平均识别置信度由 0.7533 增至 0.7769。压缩实验中，AVIF Q50 与 DCT Q50 的平均码率分别为 0.6216 和 1.0280 bpp，平均 PSNR 分别为 33.0699 和 32.6949 dB。'),
('结果分析同时考虑平均指标', '三项实验从亮度与颜色、页面空间结构和编码表示三个层面完成了图像处理。增强结果呈现出更接近参考图的主体颜色与曝光；文档校正改善了所示书页及折叠纸张的页面形态；压缩实验建立了质量因子、码率与重建质量之间的定量关系。结合完整数据统计与典型图像对照，可以具体分析各处理步骤的作用。'),
('课程要求完成图像增强', '课程要求完成图像增强、几何校正和图像压缩三项实习，并在报告中给出评价指标计算结果及处理前后对比图[1]。三个任务分别面向像素亮度与颜色、像素空间位置和图像编码表示。本实习围绕各任务构建了数据处理、结果保存、定量评价及视觉分析流程。'),
('增强任务关注极低信号', '增强任务关注极低信号条件下的颜色、曝光、噪声与结构恢复；几何校正任务关注页面形态及文字呈现；压缩任务关注码率、重建质量与耗时之间的关系。增强和压缩采用参考图像进行质量评价，文档校正结合页面规整度统计、OCR 输出和视觉对照进行分析。'),
('增强最终结果与传统对照各有', '增强最终结果与传统对照各有 1000 条逐图评价记录，训练、验证、测试分别为 800、100、100 幅；几何校正的两类评价表各含 200 条记录；压缩评价表包含 24×10 条记录。各处理结果按图像标识对应，在逐图数据基础上计算分组均值。'),
('本报告中的均值为', '本报告采用先逐图评价、再求算术平均的统计方式。PSNR 为逐图 dB 值的均值；OCR 置信度为各图文本框平均置信度的均值。压缩率以原始 RGB 像素字节数作为基准。完整数据统计用于呈现总体结果，典型图像对照用于分析颜色、结构和局部纹理。'),
('式(1)中 H、W', '式(1)中 H、W 为图像高宽，c 为 RGB 通道，L 为像素动态范围。增强以归一化 RGB、L=1 计算；压缩以 8-bit RGB、L=255 计算。SSIM 结合局部亮度、对比度与结构[2]；LPIPS 使用深层特征距离衡量感知差异[3]。三项指标分别补充像素、结构与感知层面的评价信息。'),
('增强使用 Canon_0.0001', '增强使用 Canon_0.0001 数据及对应正常曝光 JPEG 参考，共 1000 对。RAW 为 3840×2160 的小端 uint16 数据。依据 ORB 几何匹配与缩略图相似度形成 418 个自动场景组，以 20261021 为随机种子，按组划分为 800/100/100 幅。相机颜色映射与归一化尺度由训练数据拟合，验证集用于候选选择，测试集在模型冻结后评价。'),
('输入扣除黑电平', '输入扣除黑电平 512，再除以训练 RAW 信号的 99.5 分位数 69.0，保留归一化后大于 1 的数值。将四个 Bayer 相位打包为四通道，在保持像素相位对应的同时降低空间尺寸，为后续网络提供各相位的原始响应。'),
('主体为四级残差', '主体为四级残差 U-Net，基础通道数为 32。网络输出 12 个通道，通过 PixelShuffle(2) 重建全尺寸 RGB。训练集拟合的固定相机颜色映射提供颜色初值，低分辨率全图上下文提供曝光与颜色修正，细节分支学习残差。最终采用 adaptive 候选，以可逆 log1p 变换处理 RAW 动态范围，并通过学习门控调节颜色初值的贡献，输出直接对应 sRGB。'),
('训练采用 256×256', '训练采用 256×256 的 packed RAW 裁块、batch=4、AdamW、混合精度及梯度裁剪。随机裁剪保持相位一致。最终模型目标函数包含 RGB Charbonnier 损失、权重 0.1 的 SSIM 损失和权重 0.05 的低频 Cb/Cr 色度损失，兼顾像素重建、结构与颜色。'),
('模型选择首先比较', '模型选择首先比较完整验证集 PSNR；与最高 PSNR 相差不超过 0.10 dB 时，再比较 LPIPS 与色差。最终选中 adaptive 阶段的 24000 步权重，测试集用于冻结后的评价。各训练阶段与阶段验证的记录时长合计约 1.38 小时。'),
('测试集上，最终增强相对', '测试集上，最终增强相对标定传统基线的 PSNR 增加 7.365 dB，SSIM 增加 0.2437，LPIPS 减少 0.2054，像素重建、结构相似度与感知指标均得到改善。全量 1000 幅图像的平均 PSNR 由 17.153 dB 提升至 23.742 dB，提升 6.589 dB。'),
('图3-1显示处理后分布', '图3-1显示测试集的 PSNR 分布整体向高值区间移动。最终增强中有 61 幅位于 25—30 dB 区间、4 幅位于 30—35 dB 区间；传统基线主要集中在 15—25 dB 区间。频数分布与平均值的提升相互印证。'),
('3.6　划分差异', '3.6　分组评价与处理效率'),
('最终测试均值高于', '训练、验证和测试分别统计指标，使模型拟合、候选选择与最终评价具有清楚的对应关系。最终模型的平均纯推理耗时为 0.246 秒/幅，评价耗时为 0.281 秒/幅，读取、PNG 写入与初始化另计。运行环境为 NVIDIA GeForce RTX 5060 Laptop GPU。'),
('下列三例来自训练划分', '下列三例来自训练划分，展示商品、招牌和自行车场景的增强结果，下一页另列测试样例。每组上排从左至右为未增益 RAW 强度示意、标定传统基线、最终增强与 GT；下排为对应位置的 512×512 局部。RAW 灰度以四相位均值按 uint16 容器范围显示，仅用于输入强度展示。'),
('标定传统基线仍可见', '与标定传统基线相比，最终增强的主体颜色更接近 GT，背景彩噪得到抑制。红色招牌恢复了主要色彩方向，商品及自行车的轮廓更易辨认。全图与同位置局部对照显示，模型能够同时改善场景的整体外观和物体结构呈现。'),
('3.8　测试示例与模型局限', '3.8　测试集图像对照'),
('图3-6　蓝色篷布场景', '图3-6　蓝色篷布场景的测试集增强结果'),
('路面场景的整体结构', '路面场景的整图 PSNR 从基线的 21.740 dB 提升至 26.349 dB，铺装边界与亮度层次得到较好呈现。篷布场景的 PSNR 从 11.256 dB 提升至 15.646 dB，主体蓝色与褶皱形态得以显现。两例从不同场景展示了模型对极暗输入的处理效果。'),
('3.9　候选比较带来的认识', '3.9　候选比较与参数选择'),
('完整验证集上，主训练候选', '完整验证集上，主训练候选 PSNR 为 20.857 dB，adaptive 候选为 21.373 dB，提升约 0.516 dB。感知损失候选的 LPIPS 为 0.3893、PSNR 为 21.146 dB。结合既定选择规则，最终保留 adaptive 模型，使候选比较与最终配置形成明确对应。'),
('这些候选来自不同训练阶段', '模型记录还包含颜色初值平滑核的比较：核尺寸 21、41 的验证 PSNR 分别为 20.706、20.382 dB，最终保留的设置取得更高的验证 PSNR。通过保存不同候选的指标，参数选择建立在验证结果上，后续也可沿用相同划分继续比较。'),
('几何校正使用课程提供', '几何校正使用课程提供的 200 幅文档图像及配套掩膜，内容包含透视、弯曲、折叠和旋转等形变。掩膜用于定位文档区域，原图与两种处理结果按文件标识一一对应，便于逐页比较页面边界、文字行与局部形态。'),
('基线从掩膜轮廓提取', '基线从掩膜轮廓提取文档边界与四角，建立四边形到矩形的单应性变换，再进行图像重采样。该方法参数少、几何意义清楚，能够统一页面外轮廓的透视形态，并为稠密网格方法提供直观的对照。'),
('式(3)中的 H', '式(3)中的 H 为单应矩阵，s 为齐次尺度。通过输出网格反查输入坐标完成重采样，使文档内容映射到规则矩形区域。结果检查同时关注页面外轮廓、正文行和整体版式。'),
('第二条路线使用', '第二条路线使用 UVDoc 官方预训练模型[4]进行推理。流程先按掩膜包围盒裁剪，并保留 3% 边缘，再按官方输入尺寸缩放 RGB 图像，预测二维采样网格，通过双线性后向采样在原裁剪分辨率生成校正结果。稠密网格能够表达书脊、纸张起伏及折痕附近的局部形变。'),
('4.4　规整度代理指标的实际含义', '4.4　页面规整度指标'),
('评价脚本将图像等比例', '评价脚本将图像等比例缩放到高 1000，进行自适应二值化并取文字包围盒。curvature 为各行文字像素横向质心序列线性去趋势后的残差标准差，数值越小表示该质心序列的偏移程度越小。'),
('line_var 实际为', 'line_var 为水平投影序列去均值、移除前三个低频频点后，最大频谱幅值占其余幅值总和的比例。数值越大，表示主周期峰越集中。两项统计从质心变化与投影周期性两个角度描述版面分布。'),
('字符数和识别框数是输出数量', '字符数、识别框数与置信度均统计自 OCR 输出；置信度先逐图平均，再对 200 幅图求均值。'),
('两种校正结果的 curvature', '校正后，两种方法的 curvature 均值均低于原图的 61.8228，透视基线为 48.8327，UVDoc 为 51.1541。OCR 三项输出统计中，UVDoc 均高于原图和透视基线，说明不同处理路线在页面分布和文本输出上具有各自的表现。'),
('4.6　OCR 的评价边界', '4.6　OCR 输出统计'),
('OCR 使用 RapidOCR', 'OCR 使用 RapidOCR；输入最长边超过 2000 时先缩小。每图字符数为识别字符串长度之和，识别框数为返回文本框的数量，置信度为该图文本框置信度均值，再对 200 幅图求均值。UVDoc 相对原图的字符数增加 41.36%，识别框数增加 24.91%，平均置信度提高约 2.35 个百分点。'),
('这些结果表明校正后识别器', '透视基线的平均字符数由 565.14 增至 617.02，平均识别框数由 37.61 增至 42.03，平均置信度由 0.7533 增至 0.7606。两条处理路线均增加了识别器输出的文本数量，其中 UVDoc 的增幅更明显。'),
('4.7　定量与视觉评价为何不完全一致', '4.7　页面形态与文字呈现'),
('透视或网格校正改变文本', '页面规整度与 OCR 统计分别描述图像布局和文本输出，结合视觉对照可以更具体地分析校正效果。下一页选取书页、双页教材及折叠纸张，观察四角变换与稠密网格在不同文档形态下的处理结果。'),
('四角基线使文档外边界', '四角基线使文档外边界更接近规则矩形，UVDoc 则进一步处理了所示书页与折叠纸张中的局部起伏。书脊附近的页面展开及正文行形态变化，使两种空间模型的作用具有直观对照。结合 OCR 输出统计，能够从页面外观与文本呈现两方面分析几何校正效果。'),
('压缩实验使用 Kodak', '压缩实验使用 Kodak Lossless True Color Image Suite 的 24 幅 24-bit RGB 图像，尺寸为 768×512 或 512×768。自研 DCT 采用质量因子 10、30、50、75、90，AVIF 采用 10、30、50、70、90，各图执行全部十种配置，形成 240 条评价记录。压缩率以解码后的原始 RGB 像素字节数为基准。'),
('编码链依次为', '编码链依次为 RGB 到 YCbCr、像素平移 128、8×8 分块正交 DCT、按质量因子缩放的 JPEG 量化表量化、Zigzag 扫描、DC 差分、AC 游程和规范哈夫曼编码。颜色分量均保留全分辨率，即 4:4:4。解码执行对应逆过程，将重建值裁回有效像素范围。码流采用 DCT1 自定义标识与头部，构成完整的 JPEG 风格实验编码器。'),
('AVIF 通过 pillow-avif-plugin', 'AVIF 通过 pillow-avif-plugin / libavif 调用，保存参数为 quality=q 和 speed=8。评价时逐图记录码率、PSNR、SSIM 与耗时，再对 24 幅图像求均值，以实测结果呈现两类编码器的质量和码率变化。'),
('耗时由批评测脚本', '批评测脚本使用 perf_counter 记录每图单次编码与单次解码的耗时，再按配置计算平均值。AVIF 通过底层库执行，自研 DCT 使用 Python 实现；表中耗时对应本次具体程序与运行条件。'),
('DCT 指自研 JPEG 风格码流', 'DCT 指自研 JPEG 风格码流；Q 为各编码器自身的质量设置，结果比较依据实测码率与重建指标。'),
('同一编码器随质量因子', '同一编码器随质量因子增大，平均 bpp、PSNR 和 SSIM 均上升，呈现出清晰的率失真变化。AVIF Q50 的平均 PSNR 为 33.0699 dB、码率为 0.6216 bpp；DCT Q50 分别为 32.6949 dB、1.0280 bpp。在这两个已测工作点上，AVIF 使用更低的平均码率取得了略高的平均 PSNR。'),
('在本次实现中，Q50', '在本次实现中，Q50 的 AVIF 平均编码/解码耗时为 28.85/10.76 ms，DCT 为 231.27/230.77 ms。自研 DCT 完成了颜色变换、系数量化与熵编码的完整链路；AVIF 则体现出本次库调用条件下的率失真与运行效率特点。'),
('石墙和摩托车两例中', '石墙和摩托车两例中，AVIF 的整图 PSNR 分别为 30.35、30.32 dB，DCT 为 30.06、30.15 dB；鹦鹉例中两者分别为 36.16、36.17 dB。三例分别呈现了砖缝、复杂纹理和颜色过渡的重建形态，同位置局部放大使两类编码结果便于直接观察。'),
('6.1　恢复、重采样与编码的共同问题', '6.1　恢复 重采样与编码的实现'),
('三个任务的输入条件与处理目标不同', '三个任务分别从恢复、重采样和编码角度处理图像信息。增强从极低信号中估计正常曝光外观，几何校正依据空间模型调整页面采样位置，压缩通过量化与编码控制信息量。各任务均完成了从输入处理到结果保存、指标计算和图像对照的实验流程。'),
('表6-1　三项实习的证据范围与主要解释限制', '表6-1　三项实习的主要成果与分析内容'),
('6.2　指标冲突具有分析价值', '6.2　定量评价与视觉分析'),
('增强的低 PSNR 篷布样例', '增强任务中，测试集 PSNR、SSIM 与 LPIPS 的平均结果一致改善；文档校正中，文本输出数量和平均置信度增加；压缩任务中，质量因子与码率、重建指标之间形成连续的变化趋势。完整统计呈现总体表现，典型图像则将数值对应到颜色、页面形态和纹理等具体内容。'),
('6.3　可比性取决于处理条件', '6.3　实验流程与结果管理'),
('增强中，相机标定只使用训练数据', '增强中，相机标定采用训练数据，传统基线与最终模型使用相同划分和指标协议。文档校正共享配套掩膜，使两条处理路线具有一致的区域定位条件。压缩中，每幅图像执行相同的配置集合，完整记录各配置的码率、质量与耗时。统一的数据对应关系便于追溯结果和开展比较。'),
('6.4　更有针对性的后续改进', '6.4　参数选择与方法认识'),
('增强可优先围绕双向曝光偏差', '增强的候选比较说明，网络训练和推理参数可通过验证集指标共同选择；文档校正展示了四角模型与稠密网格对页面形态的不同表达；压缩曲线则明确呈现了质量设置对码率和重建结果的影响。将方法原理、参数设置和实际结果相互对应，是本次实习形成技术判断的主要方式。'),
('图像增强完成了从极暗 RAW', '图像增强完成了从极暗 RAW 到彩色 sRGB 的处理与 1000 幅逐图评价。最终 adaptive 模型在 100 幅测试图像上取得 PSNR 25.318 dB、SSIM 0.8065、LPIPS 0.3728，相对标定传统基线均有改善。视觉对照显示了主要颜色、曝光及主体结构的恢复效果。'),
('几何校正完成了 200', '几何校正完成了 200 幅文档的透视基线、UVDoc 推理及两类评价。UVDoc 的平均 OCR 输出字符数为 798.89，平均置信度为 0.7769，均高于原图。所示书页、双页教材和折叠纸张的对照进一步展示了页面边界与局部形变的校正效果。'),
('图像压缩完成两类编码器', '图像压缩完成两类编码器在十种配置下的 240 次图像评价。随质量因子增加，实测平均码率与质量指标同步上升。AVIF 在所测工作点显示出较有利的率失真表现；自研 DCT 实现则完成了变换、量化和熵编码，并通过解码重建验证了整条处理链路。'),
('本报告以逐图结果、模型选择记录', '三项作业均保存了可追溯的逐图结果、处理配置与评价代码。增强的模型选择记录对应最终权重，文档图像与两种输出按标识对应，压缩的逐图评价与分配置汇总相互核对。对比图中的裁剪和缩放仅用于展示，指标仍对应原始评测输出。'),
('本次结果最值得保留的认识', '本次实习使算法原理与实际图像结果建立了更具体的联系。RAW 四相位输入与重建网络用于颜色和曝光恢复，单应性与稠密网格用于页面空间重排，DCT、量化和熵编码用于控制图像码率。将逐图指标与同位置图像对照结合，有助于说明各处理步骤的作用，也为后续参数比较保留了清晰的依据。'),
('课程任务书用于界定', '课程任务书规定了数据规模、任务内容与报告提交要求。增强数据为 Canon 极暗 RAW 与正常曝光参考；文档图像及掩膜来自课程数据；压缩使用已保存的 Kodak Lossless True Color Image Suite。正文数值对应各任务的逐图评价和汇总记录。'),
('表格均值保留适当小数位', '表格均值按逐图记录的原始精度汇总后显示适当小数位。局部裁剪、图像缩放与标签整理仅用于可视化，图示指标均对应整图评价。方法文献用于说明算法和指标原理，实验结果来自本地已保存记录。'),
]

def edit_text(s):
    for old,new in R:
        if s.startswith(old):
            changes.append({'before':s,'after':new})
            return new
    return s

doc=Document(); section=doc.sections[0]
section.page_width=Cm(21); section.page_height=Cm(29.7)
section.top_margin=Cm(2.1); section.bottom_margin=Cm(2.0)
section.left_margin=Cm(2.05); section.right_margin=Cm(2.05)
section.header_distance=Cm(1.0); section.footer_distance=Cm(0.9)
WIDTH=section.page_width-section.left_margin-section.right_margin
PWIDTH=WIDTH/Pt(1)
styles=doc.styles

def font(st,name,size,bold=False):
    st.font.name='Times New Roman'; st.font.size=Pt(size); st.font.bold=bold; st.font.color.rgb=RGBColor(0,0,0)
    st._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),name)

font(styles['Normal'],'宋体',11)
normal=styles['Normal'].paragraph_format
normal.line_spacing=Pt(18.2); normal.space_after=Pt(4); normal.first_line_indent=Pt(22)
normal.widow_control=True
for name,size in [('Title',26),('Heading 1',15),('Heading 2',11.5)]:
    font(styles[name],'黑体',size,True)
    pf=styles[name].paragraph_format; pf.first_line_indent=Pt(0); pf.space_before=Pt(5 if name=='Heading 2' else 0); pf.space_after=Pt(7 if name=='Heading 1' else 4); pf.line_spacing=Pt(20 if name=='Heading 1' else 17); pf.keep_with_next=True
font(styles['Caption'],'宋体',9)
pf=styles['Caption'].paragraph_format; pf.first_line_indent=Pt(0); pf.line_spacing=Pt(12); pf.space_before=Pt(2); pf.space_after=Pt(6); pf.alignment=WD_ALIGN_PARAGRAPH.CENTER

def xmlprop(parent,name,attrs):
    el=OxmlElement(name)
    for k,v in attrs.items():el.set(qn(k),str(v))
    parent.append(el);return el

def set_run(run,east='宋体',size=None):
    run.font.name='Times New Roman'; run._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),east)
    if size:run.font.size=Pt(size)

def add_html(p,s):
    # Preserve subscript/superscript notation in editable runs.
    root=etree.fromstring(('<root>'+s.replace('<br/>','\n').replace('&','&amp;')+'</root>').encode('utf-8'))
    def walk(el,bold=False,sub=False,sup=False):
        if el.text:
            r=p.add_run(el.text);set_run(r);r.bold=bold;r.font.subscript=sub;r.font.superscript=sup
        for child in el:
            walk(child,bold or child.tag=='b',sub or child.tag=='sub',sup or child.tag in ('super','sup'))
            if child.tail:
                r=p.add_run(child.tail);set_run(r);r.bold=bold;r.font.subscript=sub;r.font.superscript=sup
    walk(root)

def add_equation(s):
    p=doc.add_paragraph();p.paragraph_format.first_line_indent=Pt(0);p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after=Pt(6)
    om=OxmlElement('m:oMathPara'); math=OxmlElement('m:oMath'); om.append(math)
    def mr(text):
        r=OxmlElement('m:r');rp=OxmlElement('w:rPr');xmlprop(rp,'w:rFonts',{'w:ascii':'Cambria Math','w:hAnsi':'Cambria Math'});r.append(rp)
        t=OxmlElement('m:t');t.text=text;r.append(t);return r
    parts=re.split(r'(<sub>.*?</sub>|<super>.*?</super>)',s)
    nodes=[]
    for part in parts:
        if part.startswith('<sub>') or part.startswith('<super>'):
            sub=part.startswith('<sub>'); base=nodes.pop() if nodes else mr('')
            node=OxmlElement('m:sSub' if sub else 'm:sSup');e=OxmlElement('m:e');e.append(base);node.append(e)
            x=OxmlElement('m:sub' if sub else 'm:sup');x.append(mr(re.sub('<.*?>','',part)));node.append(x);nodes.append(node)
        elif part:
            if re.search(r'[A-Za-zΣ]\s*$',part) and len(part)>1:
                nodes.append(mr(part[:-1]));nodes.append(mr(part[-1]))
            else:nodes.append(mr(part))
    for node in nodes:math.append(node)
    p._p.append(om)

def add_para(s,kind='body'):
    if kind=='eq':return add_equation(s)
    if kind=='h1':return doc.add_paragraph(s,'Heading 1')
    if kind=='h2':return doc.add_paragraph(s,'Heading 2')
    p=doc.add_paragraph(style='Caption' if kind=='cap' else 'Normal')
    if kind in ('note','ref','plain'):
        p.paragraph_format.first_line_indent=Pt(0)
    if kind=='note':
        p.paragraph_format.line_spacing=Pt(13);p.paragraph_format.space_after=Pt(4)
    if kind=='ref':
        p.paragraph_format.left_indent=Pt(18);p.paragraph_format.first_line_indent=Pt(-18);p.paragraph_format.line_spacing=Pt(15.5);p.paragraph_format.space_after=Pt(8)
    add_html(p,s)
    if kind in ('cap','note','ref'):
        for r in p.runs:r.font.size=Pt(9 if kind!='ref' else 9.5)
    return p

def set_table(t,widths):
    t.alignment=WD_TABLE_ALIGNMENT.CENTER;t.autofit=False
    pr=t._tbl.tblPr
    bd=OxmlElement('w:tblBorders')
    for edge in ('top','left','bottom','right','insideH','insideV'):
        xmlprop(bd,'w:'+edge,{'w:val':'single','w:sz':'4','w:color':'D9D9D9'})
    pr.append(bd)
    for i,col in enumerate(t.columns):col.width=Pt(widths[i])
    for j,row in enumerate(t.rows):
        rowpr=row._tr.get_or_add_trPr();rowpr.append(OxmlElement('w:cantSplit'))
        if j==0:rowpr.append(OxmlElement('w:tblHeader'))
        for i,c in enumerate(row.cells):
            c.width=Pt(widths[i]);c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            tcp=c._tc.get_or_add_tcPr();m=OxmlElement('w:tcMar')
            for edge in ('top','bottom'):xmlprop(m,'w:'+edge,{'w:w':52,'w:type':'dxa'})
            for edge in ('left','right'):xmlprop(m,'w:'+edge,{'w:w':65,'w:type':'dxa'})
            tcp.append(m)
            if j==0:xmlprop(tcp,'w:shd',{'w:fill':'ECECEC'})

def add_table(f,page_index):
    vals=[[edit_text(x.getPlainText() if isinstance(x,Paragraph) else str(x)) for x in row] for row in f._cellvalues]
    widths=[float(x)*PWIDTH/b.CW for x in f._argW]
    if page_index==12: # Replace the discussion matrix's warning column with factual achievements.
        vals=[['任务','已完成的处理工作','主要评价结果'],
              ['增强','1000 对 RAW 与参考图处理；残差 U-Net 训练、候选比较与逐图评价','测试 PSNR 25.318 dB、SSIM 0.8065、LPIPS 0.3728'],
              ['几何校正','200 幅文档的透视基线、UVDoc 推理与页面对照','UVDoc 平均字符数 798.89、平均置信度 0.7769'],
              ['压缩','24 幅图像、10 组配置的 DCT 与 AVIF 编解码评价','AVIF Q50：0.6216 bpp、33.0699 dB；完整 DCT 编码链路']]
        widths=[58,PWIDTH*.47,PWIDTH-58-PWIDTH*.47]
    if page_index==13:
        vals=[[v.replace('，不反转方向','').replace('不用于重新评分','指标对应原评测输出') for v in row] for row in vals]
    t=doc.add_table(rows=len(vals),cols=len(vals[0]));set_table(t,widths)
    for j,row in enumerate(vals):
        for i,text in enumerate(row):
            p=t.cell(j,i).paragraphs[0];p.paragraph_format.first_line_indent=Pt(0);p.paragraph_format.line_spacing=Pt(12);p.paragraph_format.space_after=Pt(0)
            if len(text)<20 or (j==0):p.alignment=WD_ALIGN_PARAGRAPH.CENTER
            if page_index==14:text=text.replace('；','；\n').replace('/metric_line_regularity.csv','/metric_line_regularity.csv')
            r=p.add_run(text);set_run(r,size=8.7);r.bold=j==0
    # No blank paragraph with full body leading between the table and next text.

def add_image(path,width,height,alt):
    p=doc.add_paragraph();p.paragraph_format.first_line_indent=Pt(0);p.paragraph_format.line_spacing=1
    p.paragraph_format.space_before=Pt(2);p.paragraph_format.space_after=Pt(0);p.paragraph_format.keep_with_next=True;p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    shape=p.add_run().add_picture(str(path),width=Pt(width),height=Pt(height));shape._inline.docPr.set('descr',alt)

def chart_asset(f,tag):
    pdf=WORK/(tag+'.pdf');png=WORK/(tag+'.png')
    renderPDF.drawToFile(f.drawing,str(pdf))
    subprocess.run(['pdftoppm','-r','230','-png','-singlefile',str(pdf),str(png.with_suffix(''))],check=True,capture_output=True)
    return png

def header_footer(sec):
    sec.header.is_linked_to_previous=False;sec.footer.is_linked_to_previous=False
    p=sec.header.paragraphs[0];p.paragraph_format.first_line_indent=Pt(0);p.alignment=WD_ALIGN_PARAGRAPH.RIGHT
    r=p.add_run('数字图像处理综合实习报告');set_run(r,size=8);r.font.color.rgb=RGBColor(90,90,90)
    p=sec.footer.paragraphs[0];p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.first_line_indent=Pt(0)
    for text in ('第 ',None,' 页'):
        if text is None:
            fld=OxmlElement('w:fldSimple');fld.set(qn('w:instr'),'PAGE');p._p.append(fld)
        else:set_run(p.add_run(text),size=9)
    num=OxmlElement('w:pgNumType');num.set(qn('w:start'),'1');sec._sectPr.append(num)

# Match the academic structure of the original cover, with editable blank fields.
for title,size,before,after in [('数字图像处理课程',19,65,44),('综合实习报告',28,0,35)]:
    p=doc.add_paragraph(title,'Title');p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.space_before=Pt(before);p.paragraph_format.space_after=Pt(after)
    for r in p.runs:set_run(r,'黑体',size)
p=add_para('图像增强　文档几何校正　图像压缩','plain');p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.space_after=Pt(95)
for label in ('姓　　名','学　　号','专业班级','指导教师'):
    p=add_para(label+'：________________________','plain');p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.space_after=Pt(20)
p=add_para('报告整理日期：2026 年 10 月 24 日','plain');p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.space_before=Pt(65)
body_section=doc.add_section(WD_SECTION.NEW_PAGE);header_footer(body_section)

toc=[('1　实习任务与评价设计','2'),('2　数据核验与统计约定','2'),('3　极暗 RAW 图像增强','3'),('4　文档几何校正','7'),('5　图像压缩与率失真分析','10'),('6　综合讨论','13'),('7　实习结论与可复核性','14'),('参考文献与数据依据','15')]
figure_count=chart_count=0
for page_index,(title,items,bookmark) in enumerate(b.PAGES):
    if page_index>0:
        p=doc.add_paragraph();p.paragraph_format.line_spacing=Pt(1);p.paragraph_format.space_after=Pt(0);p.paragraph_format.first_line_indent=Pt(0)
        p.add_run().add_break(WD_BREAK.PAGE)
    for f in items:
        if isinstance(f,Paragraph):
            original=f.text
            new=edit_text(original)
            if page_index==0 and f.style.name=='h2' and '目' in original:
                add_para(new,'h2');continue
            add_para(new,f.style.name)
        elif isinstance(f,Table):
            if page_index==0:
                for label,num in toc:
                    p=doc.add_paragraph();p.paragraph_format.first_line_indent=Pt(0);p.paragraph_format.space_after=Pt(3);p.paragraph_format.line_spacing=Pt(16)
                    p.paragraph_format.tab_stops.add_tab_stop(Pt(PWIDTH-6),WD_TAB_ALIGNMENT.RIGHT,WD_TAB_LEADER.DOTS)
                    set_run(p.add_run(label+'\t'+num),size=10)
            else:add_table(f,page_index)
        elif isinstance(f,Image):
            width=f.drawWidth*PWIDTH/b.CW;height=f.drawHeight*PWIDTH/b.CW
            add_image(f.filename,width,height,Path(f.filename).stem+' 图像处理结果对照');figure_count+=1
        elif isinstance(f,b.Graphic):
            chart_count+=1;path=chart_asset(f,'chart_'+str(chart_count))
            add_image(path,PWIDTH,f.height*PWIDTH/b.CW,'逐图评价数据统计图')
        elif isinstance(f,Spacer):
            if f.height>9:
                p=doc.add_paragraph();p.paragraph_format.line_spacing=Pt(1);p.paragraph_format.space_after=Pt(2)

doc.core_properties.title='数字图像处理综合实习报告'
doc.core_properties.subject='图像增强 文档几何校正 图像压缩'
doc.core_properties.author='';doc.core_properties.last_modified_by=''
doc.core_properties.comments=''
TARGET.parent.mkdir(exist_ok=True,parents=True);doc.save(TARGET)
assert hashlib.sha256(ORIGINAL.read_bytes()).hexdigest()==original_hash
(WORK/'editorial_changes.json').write_text(json.dumps(changes,ensure_ascii=False,indent=2),encoding='utf-8')
state={'original_pdf_sha256':original_hash,'target':str(TARGET),'comparison_figures':figure_count,'statistical_figures':chart_count,'revised_paragraph_count':len(changes)}
(WORK/'build_state.json').write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(state,ensure_ascii=False,indent=2))
