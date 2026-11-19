from pathlib import Path
import csv, json, hashlib, math, os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path(r'D:\数字图像处理实习作业')
OUT = ROOT / 'output'
ASSETS = OUT / 'report_assets'
ASSETS.mkdir(exist_ok=True, parents=True)
FONT = r'C:\Windows\Fonts\simsun.ttc'
HEI = r'C:\Windows\Fonts\simhei.ttf'
ENH = ROOT / '图像增强2'
GEO = ROOT / '几何校正workbuddy2'
CMP = ROOT / '图像压缩'

def readcsv(p):
    with p.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def inventory():
    found = {}
    for base in (ENH, GEO, CMP):
        for folder, dirs, files in os.walk(base):
            dirs[:] = [d for d in dirs if d not in ('venv', 'venv_cai', '.venv', '.git', '__pycache__', '.pytest_cache')]
            for name in files:
                p = Path(folder) / name
                st = p.stat()
                found[str(p.relative_to(ROOT))] = [st.st_size, st.st_mtime_ns]
    return found

snap = OUT / 'source_integrity_before.json'
if not snap.exists():
    snap.write_text(json.dumps(inventory(), ensure_ascii=False, indent=2), encoding='utf-8')

used = set()
def source(p):
    p = Path(p)
    assert p.exists(), p
    used.add(p)
    return p

def label(draw, xy, text, size=29, color='#202020', anchor='mm', bold=False):
    draw.text(xy, text, font=ImageFont.truetype(HEI if bold else FONT, size), fill=color, anchor=anchor)

def fit(im, size):
    im = im.copy().convert('RGB')
    im.thumbnail(size, Image.Resampling.LANCZOS)
    return im

final = readcsv(source(ENH / 'artifacts/results/final/metrics.csv'))
base = readcsv(source(ENH / 'artifacts/results/baseline/metrics.csv'))
assert len(final) == len(base) == 1000
assert {x['id'] for x in final} == {x['id'] for x in base}
ef = {x['id']: x for x in final}
eb = {x['id']: x for x in base}
enh_stats = {}
for tag, rows in [('final', final), ('baseline', base)]:
    enh_stats[tag] = {}
    for split in ['all', 'train', 'val', 'test']:
        rr = [r for r in rows if split == 'all' or r['split'] == split]
        v = {'count': len(rr)}
        for k in ['psnr', 'ssim', 'lpips', 'delta_e_2000_512', 'inference_seconds', 'evaluation_seconds']:
            a = [float(r[k]) for r in rr]
            assert all(math.isfinite(x) for x in a)
            v[k] = float(np.mean(a))
        v['psnr_values'] = [float(r['psnr']) for r in rr]
        enh_stats[tag][split] = v

manifest = json.loads(source(ENH / 'artifacts/data/manifest.json').read_text(encoding='utf-8-sig'))
records = {r['id']: r for r in manifest['items']}
data_root = Path(r'D:\BaiduNetdiskDownload\BIED')
mapping = []
coords = {'10': (2800, 1050), '100': (1900, 250), '798': (550, 700)}
for idx, key in enumerate(['10', '100', '798', '518', '435'], 2):
    rec = records[key]
    rawp, gtp = source(data_root / rec['raw']), source(data_root / rec['gt'])
    bp = source(ENH / f'artifacts/results/baseline/images/{key}.png')
    fp = source(ENH / f'artifacts/results/final/images/{key}.png')
    a = np.fromfile(rawp, dtype='<u2').reshape(2160, 3840).astype(np.float32)
    gray = (a[::2, ::2] + a[1::2, ::2] + a[::2, 1::2] + a[1::2, 1::2]) / 4
    raw = Image.fromarray(np.uint8(np.clip(gray * 255 / 65535, 0, 255))).convert('RGB')
    ims = [raw, Image.open(bp).convert('RGB'), Image.open(fp).convert('RGB'), Image.open(gtp).convert('RGB')]
    cw, pad, head = 680, 12, 47
    fh = round(cw * 2160 / 3840)
    detail = key in coords
    dh = 260 if detail else 0
    im = Image.new('RGB', (4 * cw + 5 * pad, head + fh + 53 + dh), 'white')
    d = ImageDraw.Draw(im)
    names = ['RAW 强度示意（未增益）', '标定传统基线', '最终增强结果', '正常曝光参考 GT']
    for i, (pic, title) in enumerate(zip(ims, names)):
        x = pad + i * (cw + pad)
        d.rectangle((x, head, x + cw, head + fh), fill='#eeeeee')
        im.paste(pic.resize((cw, fh), Image.Resampling.LANCZOS), (x, head))
        label(d, (x + cw / 2, 24), title, 29)
    label(d, (pad + (cw + pad) * 1.5 + cw/2-pad/2, head + fh + 24),
          f"整图：PSNR {float(ef[key]['psnr']):.3f} dB   SSIM {float(ef[key]['ssim']):.4f}   LPIPS {float(ef[key]['lpips']):.4f}", 28)
    if detail:
        x0, y0 = coords[key]
        for i in range(1, 4):
            x = pad + i * (cw + pad)
            crop = ims[i].crop((x0, y0, x0 + 512, y0 + 512)).resize((234, 234), Image.Resampling.LANCZOS)
            im.paste(crop, (int(x + cw/2 - 117), head + fh + 47))
        label(d, (pad + cw/2, head + fh + 125), '下排：对应位置', 29)
        label(d, (pad + cw/2, head + fh + 165), '512×512 局部', 29)
    dest = ASSETS / f'enh_{key}.jpg'
    im.save(dest, quality=94, subsampling=0)
    mapping.append({'assignment': '图像增强', 'id': key, 'figure': f'图3-{idx}', 'sources': [str(rawp), str(bp), str(fp), str(gtp)], 'asset': str(dest), 'split': rec['split']})

ocr = readcsv(source(GEO / 'eval/metric_ocr.csv'))
line = readcsv(source(GEO / 'eval/metric_line_regularity.csv'))
assert len(ocr) == len(line) == 200
assert {r['name'] for r in ocr} == {r['name'] for r in line}
geo_stats = {}
for method in ['original', 'baseline', 'uvdoc']:
    geo_stats[method] = {}
    for tag, rows, key in [('chars',ocr,'chars'), ('lines',ocr,'lines'), ('conf',ocr,'conf'), ('curv',line,'curv'), ('lv',line,'lv')]:
        vals = [float(r[f'{method}_{key}']) for r in rows]
        assert all(math.isfinite(x) for x in vals)
        geo_stats[method][tag] = float(np.mean(vals))
for i, key in enumerate(['150', '174', '180', 'dewarp20260728_curved_029'], 2):
    ps = [source(GEO / f'GeoCorrection/_extract/GeoCorrection/images/{key}.jpg'), source(GEO / f'results/baseline/{key}.png'), source(GEO / f'results/uvdoc/{key}.png')]
    ims = [Image.open(p).convert('RGB') for p in ps]
    # Match the height for display, without stretching the page geometry.
    height, pad, header = 460, 14, 42
    widths = [round(im.width / im.height * height) for im in ims]
    canvas = Image.new('RGB', (sum(widths) + 4 * pad, height + header + 8), 'white')
    d = ImageDraw.Draw(canvas)
    x = pad
    for pic, w, title in zip(ims, widths, ['原始畸变文档', '四角透视基线', 'UVDoc 校正结果']):
        canvas.paste(pic.resize((w, height), Image.Resampling.LANCZOS), (x, header))
        label(d, (x+w/2,20), title, 25)
        x += w+pad
    dest = ASSETS / f'geo_{key}.jpg'
    canvas.save(dest, quality=94, subsampling=0)
    mapping.append({'assignment':'几何校正', 'id':key, 'figure':f'图4-{i}', 'sources':[str(p) for p in ps], 'asset':str(dest)})

cr = readcsv(source(CMP / 'results/metrics.csv'))
summary = readcsv(source(CMP / 'results/summary.csv'))
assert len(cr) == 240 and len(summary) == 10
assert len({r['image'] for r in cr}) == 24
comp_stats = []
for s in summary:
    rr = [r for r in cr if r['codec'] == s['codec']]
    assert len(rr) == 24
    record = {'codec':s['codec']}
    for k in ['bpp','psnr','ssim','compression_ratio','t_enc_ms','t_dec_ms']:
        record[k] = float(np.mean([float(r[k]) for r in rr]))
    comp_stats.append(record)
for i, key in enumerate(['01','05','23'], 2):
    p = source(CMP / f'results/compare_{key}.png')
    im = Image.open(p).convert('RGB')
    d = ImageDraw.Draw(im)
    d.rectangle((0,0,1800,71), fill='white')
    d.rectangle((0,454,1800,507), fill='white')
    for j, codec in enumerate([None, 'jpeg-dct-q50', 'avif-q50']):
        cx = 308 + j*592
        row = next((r for r in cr if r['image']==key and r['codec']==codec),None)
        title = ['原始 RGB','DCT Q50','AVIF Q50'][j]
        label(d,(cx,34),title,29)
        subtitle = '中心 160×160 局部' if row is None else f"局部放大；整图 PSNR {float(row['psnr']):.2f} dB"
        label(d,(cx,483),subtitle,24)
    dest = ASSETS / f'compression_{key}.jpg'
    im.save(dest,quality=96,subsampling=0)
    raw = source(CMP / f'Kodak-Lossless-True-Color-Image-Suite-master/Kodak-Lossless-True-Color-Image-Suite-master/PhotoCD_PCD0992/{key}.png')
    mapping.append({'assignment':'图像压缩','id':key,'figure':f'图5-{i}','sources':[str(raw),str(p)],'asset':str(dest)})

for p in [ENH/'artifacts/selection.json', ENH/'artifacts/reports/实习报告.md', GEO/'总结文档.md', GEO/'eval/04_line_regularity.py', GEO/'eval/05_ocr_eval.py', CMP/'run_benchmark.py', CMP/'plot.py', CMP/'compressors/jpeg_dct.py', CMP/'compressors/avif.py']:
    source(p)
source(Path(r'D:\下载1\数字图像处理课程-课间实习-2026-09-30.pdf'))
hashes = {str(p):digest(p) for p in sorted(used)}
(OUT/'source_hashes.json').write_text(json.dumps(hashes,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'figure_sources.json').write_text(json.dumps(mapping,ensure_ascii=False,indent=2),encoding='utf-8')
data = {'enh':enh_stats,'enh_cases':{k:{'final':ef[k],'baseline':eb[k]} for k in ['10','100','798','518','435']}, 'geo':geo_stats,'geo_cases':{k:{'ocr':next(r for r in ocr if r['name']==k),'line':next(r for r in line if r['name']==k)} for k in ['150','174','180','dewarp20260728_curved_029']},'compression':comp_stats,'compression_cases':[r for r in cr if r['image'] in ['01','05','23'] and r['codec'] in ['jpeg-dct-q50','avif-q50']]}
(OUT/'verified_metrics.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
print('Prepared',len(mapping),'comparison cases; all 1000/200/240 metric records verified.')
print('All generated files are under',OUT)

if __name__ == '__main__':
    pass
