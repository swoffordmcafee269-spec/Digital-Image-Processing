"""用 Word COM 将 DOCX 导出为 PDF，并返回页数。"""

import sys
from pathlib import Path

import pythoncom
import win32com.client

BASE = Path(__file__).resolve().parent.parent
DOCX = BASE / '数字图像处理综合实习报告_修订版.docx'
PDF = BASE / '_work' / 'word_export.pdf'

pythoncom.CoInitialize()
app = None
doc = None
try:
    app = win32com.client.DispatchEx('Word.Application')
    app.Visible = False
    app.DisplayAlerts = 0
    app.ScreenUpdating = False
    doc = app.Documents.Open(str(DOCX), False, True, False)
    doc.Repaginate()
    pages = doc.ComputeStatistics(2)
    doc.ExportAsFixedFormat(str(PDF), 17)
    print('pages', pages)
    print('pdf', PDF, PDF.exists(), PDF.stat().st_size if PDF.exists() else 0)
finally:
    if doc is not None:
        try:
            doc.Close(0)
        except Exception as exc:
            print('close doc:', exc)
    if app is not None:
        try:
            app.Quit()
        except Exception as exc:
            print('quit:', exc)
    pythoncom.CoUninitialize()
