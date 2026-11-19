from pathlib import Path
import hashlib, json, os, re
from docx import Document
from zipfile import ZipFile
from lxml import etree

root=Path(r'D:\数字图像处理实习作业'); out=root/'output'
docx=out/'docx/数字图像处理综合实习报告_修订版.docx'
original=out/'pdf/数字图像处理综合实习报告.pdf'
before=json.loads((out/'source_integrity_before.json').read_text(encoding='utf-8'))
after={}
for base in ('图像增强2','几何校正workbuddy2','图像压缩'):
    for folder,dirs,files in os.walk(root/base):
        dirs[:]=[d for d in dirs if d not in ('venv','venv_cai','.venv','.git','__pycache__','.pytest_cache')]
        for name in files:
            p=Path(folder)/name; st=p.stat()
            after[str(p.relative_to(root))]=[st.st_size,st.st_mtime_ns]
reader=Document(docx)
paras=[p.text for p in reader.paragraphs]
tables=[[[c.text for c in row.cells] for row in t.rows] for t in reader.tables]
text='\n'.join(paras+[v for table in tables for row in table for v in row])
with ZipFile(docx) as z:
    names=z.namelist()
    for name in names:
        if name.endswith(('.xml','.rels')): etree.fromstring(z.read(name))
    image_count=sum(1 for name in names if name.startswith('word/media/'))
    xml=z.read('word/document.xml')
    hard_breaks=len(re.findall(rb'<w:br[^>]*w:type="page"',xml))
    math_equations=len(re.findall(rb'<m:oMathPara',xml))
checks={
 'docx_bytes':docx.stat().st_size,
 'body_paragraphs':len(reader.paragraphs),
 'tables':len(reader.tables),
 'embedded_images':image_count,
 'manual_page_breaks':hard_breaks,
 'native_word_equations':math_equations,
 'comparison_figures':sum(f'图{a}-{b}' in text for a,b in [('3',2),('3',3),('3',4),('3',5),('3',6),('4',2),('4',3),('4',4),('4',5),('5',2),('5',3),('5',4)]),
 'metrics_present':all(x in text for x in ['25.318','0.8065','0.3728','0.6216','33.0699','0.7769']),
 'annotated_sentence_removed':'但几何代理指标并未一致改善' not in text,
 'assignment_changes':sum(before[k]!=after.get(k) for k in before),
 'assignment_files_added':len(set(after)-set(before)),
 'original_pdf_sha256':hashlib.sha256(original.read_bytes()).hexdigest(),
 'all_ooxml_xml_valid':True,
}
(out/'word_revision_work/verification.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(checks,ensure_ascii=False,indent=2))
assert checks['embedded_images']==15 and checks['manual_page_breaks']==14
assert checks['metrics_present'] and checks['annotated_sentence_removed']
assert checks['assignment_changes']==0 and checks['assignment_files_added']==0
assert checks['original_pdf_sha256']=='ec1dfd2c75c884095200197aa33f2b87c31e5fd025f330b9e1682e1d78df87b3'
