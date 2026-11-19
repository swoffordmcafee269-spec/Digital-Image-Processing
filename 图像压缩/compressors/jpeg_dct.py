"""自研 JPEG 风格 DCT 有损图像压缩编解码器（从零实现）。

流水线（编码）：
    RGB -> YCbCr（4:4:4，无色度下采样）-> 电平移位(-128)
        -> 8x8 分块 DCT -> 量化（JPEG 标准量化表按质量因子 Q 缩放）
        -> Zigzag 扫描 -> DC 差分编码 + AC 游程编码(RLE)
        -> 规范哈夫曼(Canonical Huffman)熵编码 -> 字节流

解码为严格逆过程。质量因子 Q ∈ [1, 100]，越大质量越高、码率越大。

统一接口：
    encode(img_rgb: np.uint8[H,W,3], quality=75) -> bytes
    decode(data: bytes) -> np.uint8[H,W,3]

依赖：numpy, scipy（仅用于 DCT/IDCT）。
"""

import heapq

import numpy as np
from scipy.fft import dctn, idctn

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
MAGIC = b"DCT1"

# JPEG 标准亮度量化表（8x8）
QY = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99],
], dtype=np.float64)

# JPEG 标准色度量化表（8x8）
QC = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
], dtype=np.float64)

# 8x8 块的 Zigzag 扫描顺序（索引 -> 块内线性位置）
ZIGZAG = np.array([
    0, 1, 8, 16, 9, 2, 3, 10,
    17, 24, 32, 25, 18, 11, 4, 5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13, 6, 7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63,
], dtype=np.int32)

ZRL_SYM = 240  # AC 中表示 16 个连续零的符号 (run=15, size=0)


# ---------------------------------------------------------------------------
# 位级读写
# ---------------------------------------------------------------------------
class BitWriter:
    def __init__(self):
        self.out = bytearray()
        self.cur = 0
        self.nbits = 0

    def write_bits(self, value, n):
        for i in range(n - 1, -1, -1):
            self.cur = (self.cur << 1) | ((value >> i) & 1)
            self.nbits += 1
            if self.nbits == 8:
                self.out.append(self.cur)
                self.cur = 0
                self.nbits = 0

    def flush(self):
        if self.nbits > 0:
            self.cur <<= (8 - self.nbits)
            self.out.append(self.cur)
            self.cur = 0
            self.nbits = 0
        return bytes(self.out)


class BitReader:
    def __init__(self, data):
        self.data = data
        self.bytepos = 0
        self.bitpos = 0

    def read_bit(self):
        bit = (self.data[self.bytepos] >> (7 - self.bitpos)) & 1
        self.bitpos += 1
        if self.bitpos == 8:
            self.bitpos = 0
            self.bytepos += 1
        return bit

    def read_bits(self, n):
        v = 0
        for _ in range(n):
            v = (v << 1) | self.read_bit()
        return v


# ---------------------------------------------------------------------------
# 规范哈夫曼编码
# ---------------------------------------------------------------------------
def build_huffman_lengths(freq):
    """由符号频率构建哈夫曼码长 {symbol: length}。"""
    heap = [[w, [sym]] for sym, w in freq.items()]
    heapq.heapify(heap)
    if len(heap) == 0:
        return {}
    if len(heap) == 1:
        return {heap[0][1][0]: 1}
    lengths = {}
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for sym in lo[1]:
            lengths[sym] = lengths.get(sym, 0) + 1
        for sym in hi[1]:
            lengths[sym] = lengths.get(sym, 0) + 1
        heapq.heappush(heap, [lo[0] + hi[0], lo[1] + hi[1]])
    return lengths


def canonical_codes(lengths):
    """由码长生成规范哈夫曼码 {symbol: (code_int, length)}。"""
    items = sorted(lengths.items(), key=lambda kv: (kv[1], kv[0]))
    codes = {}
    code = 0
    prev_len = 0
    for i, (sym, l) in enumerate(items):
        if i == 0:
            codes[sym] = (0, l)
        else:
            code = (code + 1) << (l - prev_len)
            codes[sym] = (code, l)
        prev_len = l
    return codes


def _decode_map(codes):
    """构建 {(code_int, length): symbol} 用于解码。"""
    return {(c, l): s for s, (c, l) in codes.items()}


# ---------------------------------------------------------------------------
# 量化表缩放
# ---------------------------------------------------------------------------
def _scale_qtable(base, quality):
    """按 libjpeg 规则由质量因子缩放标准量化表。"""
    q = int(np.clip(quality, 1, 100))
    if q < 50:
        s = 5000 // q
    else:
        s = 200 - 2 * q
    table = np.floor((base * s + 50) / 100).astype(np.int32)
    return np.clip(table, 1, 255)


# ---------------------------------------------------------------------------
# 颜色空间变换
# ---------------------------------------------------------------------------
def rgb_to_ycbcr(rgb):
    rgb = rgb.astype(np.float64)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b
    return y, cb, cr


def ycbcr_to_rgb(y, cb, cr):
    r = y + 1.402 * (cr - 128.0)
    g = y - 0.344136 * (cb - 128.0) - 0.714136 * (cr - 128.0)
    b = y + 1.772 * (cb - 128.0)
    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(np.round(rgb), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 分块 DCT + 量化
# ---------------------------------------------------------------------------
def _blocks_of(channel, block=8):
    h, w = channel.shape
    nh, nw = h // block, w // block
    ch = channel[: nh * block, : nw * block]
    return ch.reshape(nh, block, nw, block).transpose(0, 2, 1, 3).reshape(-1, block, block), nh, nw


def _forward(channel, qt):
    blocks, nh, nw = _blocks_of(channel)
    coeff = dctn(blocks - 128.0, norm="ortho", axes=(1, 2))
    quant = np.round(coeff / qt).astype(np.int32)
    return quant, nh, nw


def _inverse(quant, qt, nh, nw):
    deq = quant.astype(np.float64) * qt
    sp = idctn(deq, norm="ortho", axes=(1, 2)) + 128.0
    sp = np.clip(np.round(sp), 0, 255).astype(np.float64)
    blocks = sp.reshape(nh, nw, 8, 8).transpose(0, 2, 1, 3)
    return blocks.reshape(nh * 8, nw * 8)


# ---------------------------------------------------------------------------
# 符号化（DC 差分 + AC 游程）
# ---------------------------------------------------------------------------
def _collect_tokens(quant):
    """返回 (blocks, dc_freq, ac_freq)。

    blocks = [(dc_token, [ac_tokens]), ...] 每块一组，便于按块交错写入。
    token = (symbol, value, size)，size=0 表示无需附加比特。
    DC symbol = |diff| 的比特数（0..11）；AC symbol = run*16+size，0=EOB，240=ZRL。
    """
    blocks = []
    dc_freq, ac_freq = {}, {}
    prev_dc = 0
    for blk in quant:
        zig = blk.reshape(-1)[ZIGZAG]
        dc = int(zig[0])
        diff = dc - prev_dc
        prev_dc = dc
        size = abs(diff).bit_length()
        dc_token = (size, diff, size)
        dc_freq[size] = dc_freq.get(size, 0) + 1

        ac_tokens = []
        run = 0
        for c in zig[1:]:
            if c == 0:
                run += 1
            else:
                while run >= 16:
                    ac_tokens.append((ZRL_SYM, 0, 0))
                    ac_freq[ZRL_SYM] = ac_freq.get(ZRL_SYM, 0) + 1
                    run -= 16
                sz = abs(int(c)).bit_length()
                sym = run * 16 + sz
                ac_tokens.append((sym, int(c), sz))
                ac_freq[sym] = ac_freq.get(sym, 0) + 1
                run = 0
        # EOB 仅在存在"尾部零"时写入（JPEG 约定：块被非零系数填满时省略 EOB）
        if run > 0:
            ac_tokens.append((0, 0, 0))  # EOB
            ac_freq[0] = ac_freq.get(0, 0) + 1
        blocks.append((dc_token, ac_tokens))
    return blocks, dc_freq, ac_freq


def _write_signed(bw, value, size):
    if size == 0:
        return
    if value >= 0:
        bw.write_bits(value, size)
    else:
        bw.write_bits(value - 1 & ((1 << size) - 1), size)


def _read_signed(br, size):
    if size == 0:
        return 0
    v = br.read_bits(size)
    if v >= (1 << (size - 1)):
        return v
    return v - (1 << size) + 1


# ---------------------------------------------------------------------------
# 主接口
# ---------------------------------------------------------------------------
def encode(img_rgb, quality=75):
    img = np.asarray(img_rgb, dtype=np.uint8)
    h, w = img.shape[:2]
    qy = _scale_qtable(QY, quality)
    qc = _scale_qtable(QC, quality)

    y, cb, cr = rgb_to_ycbcr(img)
    channels = [y, cb, cr]
    qts = [qy, qc, qc]

    all_blocks = []
    dc_freq, ac_freq = {}, {}
    for ch, qt in zip(channels, qts):
        quant, _, _ = _forward(ch, qt)
        blocks, dc_f, ac_f = _collect_tokens(quant)
        all_blocks.append(blocks)
        for k, v in dc_f.items():
            dc_freq[k] = dc_freq.get(k, 0) + v
        for k, v in ac_f.items():
            ac_freq[k] = ac_freq.get(k, 0) + v

    dc_codes = canonical_codes(build_huffman_lengths(dc_freq))
    ac_codes = canonical_codes(build_huffman_lengths(ac_freq))

    # 序列化头
    head = bytearray()
    head += MAGIC
    head += h.to_bytes(2, "big") + w.to_bytes(2, "big")
    head += (int(quality) & 0xFF).to_bytes(1, "big")
    head += len(dc_codes).to_bytes(2, "big")
    for sym, (_, l) in sorted(dc_codes.items()):
        head += sym.to_bytes(2, "big") + l.to_bytes(1, "big")
    head += len(ac_codes).to_bytes(2, "big")
    for sym, (_, l) in sorted(ac_codes.items()):
        head += sym.to_bytes(2, "big") + l.to_bytes(1, "big")

    # 写码流（按块交错：每块 DC -> AC，与解码端一致）
    bw = BitWriter()
    for blocks in all_blocks:
        for dc_token, ac_tokens in blocks:
            sym, val, size = dc_token
            c, l = dc_codes[sym]
            bw.write_bits(c, l)
            _write_signed(bw, val, size)
            for sym, val, size in ac_tokens:
                c, l = ac_codes[sym]
                bw.write_bits(c, l)
                _write_signed(bw, val, size)

    return bytes(head) + bw.flush()


def decode(data):
    data = bytes(data)
    if data[:4] != MAGIC:
        raise ValueError("不是本编解码器产生的码流")
    pos = 4
    h = int.from_bytes(data[pos:pos + 2], "big"); pos += 2
    w = int.from_bytes(data[pos:pos + 2], "big"); pos += 2
    quality = data[pos]; pos += 1

    n_dc = int.from_bytes(data[pos:pos + 2], "big"); pos += 2
    dc_lengths = {}
    for _ in range(n_dc):
        sym = int.from_bytes(data[pos:pos + 2], "big"); pos += 2
        l = data[pos]; pos += 1
        dc_lengths[sym] = l

    n_ac = int.from_bytes(data[pos:pos + 2], "big"); pos += 2
    ac_lengths = {}
    for _ in range(n_ac):
        sym = int.from_bytes(data[pos:pos + 2], "big"); pos += 2
        l = data[pos]; pos += 1
        ac_lengths[sym] = l

    dc_codes = canonical_codes(dc_lengths)
    ac_codes = canonical_codes(ac_lengths)
    dc_map = _decode_map(dc_codes)
    ac_map = _decode_map(ac_codes)

    qy = _scale_qtable(QY, quality)
    qc = _scale_qtable(QC, quality)
    qts = [qy, qc, qc]

    br = BitReader(data[pos:])
    nh, nw = h // 8, w // 8
    nblocks = nh * nw

    channels_out = []
    for qt in qts:
        quant = np.zeros((nblocks, 8, 8), dtype=np.int32)
        prev_dc = 0
        for bi in range(nblocks):
            code = 0
            length = 0
            while True:
                code = (code << 1) | br.read_bit()
                length += 1
                s = dc_map.get((code, length))
                if s is not None:
                    break
            diff = _read_signed(br, s)
            prev_dc = prev_dc + diff
            quant[bi, 0, 0] = prev_dc

            k = 1
            while k < 64:
                code = 0
                length = 0
                while True:
                    code = (code << 1) | br.read_bit()
                    length += 1
                    sym = ac_map.get((code, length))
                    if sym is not None:
                        break
                if sym == 0:  # EOB
                    break
                run = sym >> 4
                size = sym & 0xF
                if sym == ZRL_SYM:
                    k += 16
                    continue
                val = _read_signed(br, size)
                k += run
                if k < 64:
                    zz = np.zeros(64, dtype=np.int32)
                    zz[k] = val
                    # 逆 zigzag：把 1D 第 k 个系数放回 2D 位置
                    pos2d = np.unravel_index(int(ZIGZAG[k]), (8, 8))
                    quant[bi, pos2d[0], pos2d[1]] = val
                k += 1
        channels_out.append(_inverse(quant, qt, nh, nw))

    y, cb, cr = channels_out
    return ycbcr_to_rgb(y, cb, cr)


def encode_file(path_in, path_out, quality=75):
    from PIL import Image
    img = np.array(Image.open(path_in).convert("RGB"))
    data = encode(img, quality)
    with open(path_out, "wb") as f:
        f.write(data)
    return data
