"""High-quality Visio-style diagrams for thesis — refined edition."""
import xml.etree.ElementTree as ET

NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", NS)

# ── SVG helpers ─────────────────────────────────────────────
def svg(w, h):
    return ET.Element("svg", {"xmlns": NS, "viewBox": f"0 0 {w} {h}", "width": str(w), "height": str(h)})

def defs(svg_el):
    d = ET.SubElement(svg_el, "defs")
    # Arrowhead
    for aid, clr in [("ah", "#455A64"), ("ah_blue", "#1565C0"), ("ah_orange", "#E65100")]:
        m = ET.SubElement(d, "marker", {"id": aid, "markerWidth": "10", "markerHeight": "7",
            "refX": "9", "refY": "3.5", "orient": "auto"})
        ET.SubElement(m, "polygon", {"points": "0 0, 10 3.5, 0 7", "fill": clr})
    # Drop shadow filter
    f = ET.SubElement(d, "filter", {"id": "shadow", "x": "-5%", "y": "-5%", "width": "115%", "height": "115%"})
    ET.SubElement(f, "feDropShadow", {"dx": "1.5", "dy": "2", "stdDeviation": "2", "floodOpacity": "0.15"})
    return d

def rect(g, x, y, w, h, rx=4, fill="#FFF", stroke="#455A64", sw=1.2, shadow=True):
    a = {"x": str(x), "y": str(y), "width": str(w), "height": str(h),
         "rx": str(rx), "ry": str(rx), "fill": fill, "stroke": stroke, "stroke-width": str(sw)}
    if shadow: a["filter"] = "url(#shadow)"
    return ET.SubElement(g, "rect", a)

def text(g, x, y, s, fs=11, bold=False, color="#222", anchor="middle", font="Microsoft YaHei, SimHei, sans-serif"):
    t = ET.SubElement(g, "text", {"x": str(x), "y": str(y), "text-anchor": anchor,
        "font-family": font, "font-size": str(fs), "fill": color})
    if bold: t.set("font-weight", "bold")
    t.text = s
    return t

def multiline(g, cx, y, lines, fs=10, bold_first=False, color="#222", line_h=16):
    h = len(lines) * line_h
    for i, line in enumerate(lines):
        text(g, cx, y - h / 2 + line_h / 2 + i * line_h, line,
             fs=fs, bold=(bold_first and i == 0), color=color)

def box(svg_el, x, y, w, h, lines, fill="#FFF", stroke="#455A64", fs=10, bold_first=False, color="#222", rx=5, shadow=True):
    g = ET.SubElement(svg_el, "g")
    rect(g, x, y, w, h, rx=rx, fill=fill, stroke=stroke, shadow=shadow)
    multiline(g, x + w / 2, y + h / 2, lines, fs=fs, bold_first=bold_first, color=color)
    return g

def arrow(g, x1, y1, x2, y2, color="#455A64", marker="ah", sw=1.8):
    ET.SubElement(g, "line", {"x1": str(x1), "y1": str(y1), "x2": str(x2), "y2": str(y2),
        "stroke": color, "stroke-width": str(sw), "marker-end": f"url(#{marker})"})

def edge_label(g, x, y, s, color="#666", fs=8):
    text(g, x, y, s, fs=fs, color=color)

# ── Palette ─────────────────────────────────────────────────
P = {
    "data_in":  "#E8EAF6", "data_out": "#C5CAE9",
    "feat_in":  "#E3F2FD", "feat_out": "#BBDEFB",
    "conv":     "#B3E5FC", "conv_hi":  "#4FC3F7",
    "proj":     "#81D4FA",
    "pos":      "#80DEEA",
    "enc_bg":   "#FFF3E0", "enc_border": "#FF9800",
    "attn":     "#FFECB3", "ffn":      "#FFE082",
    "ln":       "#FFCC80",
    "out_proj": "#C8E6C9", "act":      "#A5D6A7",
    "final":    "#66BB6A",
    "loss":     "#F8BBD0", "lr":       "#F48FB1",
    "eval":     "#D1C4E9",
}
CW, CG, CB = 160, 160, 160  # column widths

# ═══════════════════════════════════════════════════════════════
#  FIGURE 1 — System Flow
# ═══════════════════════════════════════════════════════════════
def draw_system_flow():
    s = svg(1200, 620)
    defs(s)

    # Title background
    g_title = ET.SubElement(s, "g")
    rect(g_title, 0, 0, 1200, 42, rx=0, fill="#37474F", stroke="none", shadow=False)
    text(g_title, 600, 26, "图 3-1  系统整体流水线架构 — 六阶段模块化设计", fs=15, bold=True, color="#FFF")

    # ── Phase boxes ──────────────────────────────────────────
    phases = [
        ("阶段一\n数据构建", "干净语音 TIMIT\n噪声 NOISEX-92\n↓\nRMS能量归一化混合\n4 SNR: -5/0/5/10 dB\n↓\n6876 含噪-干净对\n2000 训练 / 100 测试", P["data_in"], P["data_out"]),
        ("阶段二\n特征提取", "音频 → STFT\nn_fft=512 hop=128\n↓\n3 种特征可选:\n  · STFT对数幅度 (257维)\n  · Log-Mel频谱 (40维)\n  · RNNoise风格 (44维)\n↓\n幅度谱 + 相位 (留存)", P["feat_in"], P["feat_out"]),
        ("阶段三\n模型训练", "TransformerDenoiser\nConv1D 频率前端\n2层4头 Encoder (d=128)\n↓\n频域加权 L1 Loss\n余弦退火 LR 1e-3→1e-6\n↓\n30 Epochs, Batch=4\n保存 best checkpoint", "#FFF8E1", "#FFE082"),
        ("阶段四\n模型推理", "加载 .pt checkpoint\n输入含噪音频 → STFT\n↓\n模型前向传播\n预测幅度掩膜 [0,1]\n↓\n掩膜 × 含噪幅度谱\n复用含噪相位 → ISTFT\n输出增强 .wav", "#E8F5E9", "#A5D6A7"),
        ("阶段五\n基线对照", "同一测试集 (100条)\n分别运行:\n  · RNNoise (pyrnnoise)\n  · GTCRN (ONNX 344KB)\n  · noisereduce (谱减法)\n  · 含噪不处理 (下界)\n↓\n收集 5 组增强音频", "#F3E5F5", "#CE93D8"),
        ("阶段六\n评价分析", "增强音频 vs 干净参考\n↓\n计算客观指标:\n  · STOI (可懂度 0-1)\n  · SDR (失真比 dB)\n↓\n绘制对比图表:\n  损失曲线 + 指标柱状图", "#E3F2FD", "#90CAF9"),
    ]

    bx, by = 20, 56
    bw, bh = 180, 160
    gap = 12

    for i, (title, desc, fill_in, fill_out) in enumerate(phases):
        x = bx + i * (bw + gap)
        # Title badge
        box(s, x, by, bw, 36, [title], fill=fill_out, stroke="#546E7A", fs=11, bold_first=True, rx=3)
        # Detail box
        lines = desc.split("\n")
        box(s, x, by + 44, bw, 190, lines, fill="#FAFAFA", stroke="#BDBDBD", fs=8, rx=3, shadow=False)
        # Arrows between phases
        if i < 5:
            ax = x + bw + 2
            ay = by + bh / 2
            g_arr = ET.SubElement(s, "g")
            arrow(g_arr, ax, ay, ax + gap - 4, ay, color="#78909C", sw=2.5)
            # Phase label on arrow
            labels = ["wav", "npz", ".pt", ".wav", ".wav"]
            edge_label(g_arr, ax + gap / 2, ay - 10, labels[i], fs=9, color="#546E7A")

    # ── Bottom: innovation callouts ──────────────────────────
    callouts = [
        (240, 320, "Conv1D 频率前端\n捕捉频谱谐波结构", "#1565C0"),
        (600, 320, "频域加权 L1 Loss\n+ 余弦退火调度", "#E65100"),
        (960, 320, "轻量设计 ~150K 参数\nCPU 毫秒级推理", "#2E7D32"),
    ]
    for cx, cy, ctext, clr in callouts:
        g_c = ET.SubElement(s, "g")
        box(g_c, cx - 110, cy, 220, 48, ctext.split("\n"), fill="#FFF", stroke=clr, fs=9, bold_first=True, color=clr, rx=6)
        arrow(g_c, cx, by + 234, cx, cy, color=clr, marker="ah", sw=1.2)

    # Save
    ET.ElementTree(s).write("outputs/plots/system_flow.svg", encoding="utf-8", xml_declaration=True)
    print("Saved: outputs/plots/system_flow.svg")


# ═══════════════════════════════════════════════════════════════
#  FIGURE 2 — TransformerDenoiser Architecture
# ═══════════════════════════════════════════════════════════════
def draw_model_arch():
    s = svg(960, 1060)
    defs(s)

    # Title
    g_t = ET.SubElement(s, "g")
    rect(g_t, 0, 0, 960, 42, rx=0, fill="#37474F", stroke="none", shadow=False)
    text(g_t, 480, 26, "图 3-2  TransformerDenoiser 模型架构详图", fs=15, bold=True, color="#FFF")

    cx = 350  # center of main column
    rx = 630  # right annotation start
    y = 60
    bw = 440  # box width

    col = {
        "input": ("#E3F2FD", "#1565C0"),
        "conv":  ("#E1F5FE", "#0277BD"),
        "proj":  ("#B3E5FC", "#01579B"),
        "pos":   ("#80DEEA", "#00695C"),
        "enc":   ("#FFF8E1", "#E65100"),
        "attn":  ("#FFECB3", "#F57F17"),
        "outp":  ("#E8F5E9", "#2E7D32"),
        "act":   ("#C8E6C9", "#1B5E20"),
        "final": ("#A5D6A7", "#1B5E20"),
    }

    def block(cy, h, lines, palette_key, bold_first=True, fs=10, w_override=None):
        w = w_override or bw
        fill, stroke = col[palette_key]
        box(s, cx - w / 2, cy, w, h, lines, fill=fill, stroke=stroke, fs=fs, bold_first=bold_first, rx=4)
        return cy + h

    def down_arrow(from_y, to_y, lab="", clr="#455A64", marker="ah"):
        g_a = ET.SubElement(s, "g")
        arrow(g_a, cx, from_y, cx, to_y, color=clr, marker=marker, sw=1.6)
        if lab:
            edge_label(g_a, cx + 16, (from_y + to_y) / 2, lab, color=clr, fs=8)

    def dim_label(cy, s_text):
        text(ET.SubElement(s, "g"), cx + bw / 2 + 10, cy + 8, s_text, fs=8, color="#78909C", anchor="start")

    def innovation_badge(cy, s_text, clr="#E65100"):
        g_b = ET.SubElement(s, "g")
        box(g_b, cx + bw / 2 - 145, cy - 2, 150, 18, [s_text], fill="#FFF", stroke=clr, fs=7, bold_first=True, color=clr, rx=9, shadow=False)

    # ── 1. Input ─────────────────────────────────────────────
    y0 = y
    y = block(y, 44, ["输入特征  (Batch, Time, Freq)", "Freq ∈ {257 (STFT) | 40 (Mel) | 44 (RNNoise)}"], "input", fs=9)
    dim_label(y0, "[B, T, F]")

    down_arrow(y, y + 10)
    y += 10

    # ── 2. ConvFrontend ──────────────────────────────────────
    y0 = y
    y = block(y, 66, [
        "Conv1D 频率前端  ★ 创新点 1",
        "kernel 7 (256→64) → BN+ReLU → kernel 5 (64→64) → BN+ReLU → kernel 3 (64→1)",
        "沿频率轴逐帧独立卷积 · 残差连接保留原始频谱"
    ], "conv", fs=9)
    dim_label(y0, "[B×T, 1, F]  →  [B, T, F]")
    innovation_badge(y0 + 22, "★ 创新点 1: 谐波结构感知")

    down_arrow(y, y + 10, "残差: x + Conv1D(x)", "#E65100", "ah_orange")
    y += 10

    # ── 3. Input Projection ──────────────────────────────────
    y0 = y
    y = block(y, 36, ["输入投影  Linear(F → d_model=128)", "若 Freq≠F (如 Mel 40维) → 先经 freq_expand 升维"], "proj", fs=9)
    dim_label(y0, "[B, T, 128]")

    down_arrow(y, y + 10)
    y += 10

    # ── 4. Positional Encoding ───────────────────────────────
    y0 = y
    y = block(y, 42, [
        "正弦位置编码  (Positional Encoding)",
        "PE(pos,2i)=sin(pos/10000²ⁱ/¹²⁸)    PE(pos,2i+1)=cos(pos/10000²ⁱ/¹²⁸)"
    ], "pos", fs=9)
    dim_label(y0, "[B, T, 128]")

    down_arrow(y, y + 10)
    y += 10

    # ── 5. Transformer Encoder ───────────────────────────────
    enc_top = y

    # Encoder header
    y = block(y, 36, ["Transformer Encoder  (×2 layers,  d_model=128,  nhead=4,  d_ff=512)"], "enc", fs=10, w_override=bw + 40)
    dim_label(enc_top, "[B, T, 128]")

    # Layer 1
    y = block(y, 62, [
        "第 1 层 EncoderLayer",
        "Multi-Head Self-Attention (h=4)  →  Add & LayerNorm  →  Feed-Forward (128→512→128)  →  Add & LayerNorm"
    ], "attn", fs=9, w_override=bw + 10)

    # Between-layer arrow
    g_la = ET.SubElement(s, "g")
    arrow(g_la, cx, y, cx, y + 8, color="#FF9800", marker="ah", sw=1.2)

    # Layer 2
    y += 8
    y = block(y, 62, [
        "第 2 层 EncoderLayer",
        "Multi-Head Self-Attention (h=4)  →  Add & LayerNorm  →  Feed-Forward (128→512→128)  →  Add & LayerNorm"
    ], "attn", fs=9, w_override=bw + 10)

    # Encoder bounding box
    g_enc = ET.SubElement(s, "g")
    rect(g_enc, cx - (bw + 44) / 2, enc_top - 2, bw + 44, y - enc_top + 4,
         rx=8, fill="none", stroke=P["enc_border"], sw=2.5, shadow=False)

    # Attention detail callout
    g_att = ET.SubElement(s, "g")
    box(g_att, cx + bw / 2 + 30, enc_top + 10, 200, 56, [
        "Self-Attention 计算过程:",
        "Q,K,V = Linear(X)",
        "Score = softmax(QKᵀ/√dₖ)",
        "Output = Score · V",
    ], fill="#FFFDE7", stroke="#F9A825", fs=7, rx=3, shadow=False)
    ET.SubElement(g_att, "line", {
        "x1": str(cx + bw / 2 + 5), "y1": str(enc_top + 30),
        "x2": str(cx + bw / 2 + 28), "y2": str(enc_top + 30),
        "stroke": "#F9A825", "stroke-width": "1", "stroke-dasharray": "3,2",
    })

    down_arrow(y, y + 10)
    y += 10

    # ── 6. Output Projection ─────────────────────────────────
    y0 = y
    y = block(y, 36, [
        "输出投影  Linear(128 → Freq)",
        "幅度掩膜:  Freq=257    复数掩膜(探索):  Freq=514"
    ], "outp", fs=9)
    dim_label(y0, "[B, T, 257]")

    down_arrow(y, y + 10)
    y += 10

    # ── 7. Activation ────────────────────────────────────────
    y0 = y
    y = block(y, 36, [
        "激活函数    幅度掩膜: Sigmoid → 值域 [0, 1]    复数掩膜: Identity (无界)"
    ], "act", fs=9)

    down_arrow(y, y + 10)
    y += 10

    # ── 8. Output ────────────────────────────────────────────
    y0 = y
    y = block(y, 48, [
        "输出掩膜  [B, T, F]  (0=噪声抑制,  1=语音保留)",
        "增强信号 = ISTFT ( 掩膜 × 含噪幅度 · e^{j·含噪相位} )"
    ], "final", fs=10, bold_first=True)
    dim_label(y0, "→ 增强语音 .wav")

    # ── Innovation 2 callout (bottom) ─────────────────────────
    g_i2 = ET.SubElement(s, "g")
    box(g_i2, cx - 95, y + 40, 190, 56, [
        "★ 创新点 2: 训练策略优化",
        "频域加权 L1 Loss",
        "低频权重 3×  →  高频权重 1×",
    ], fill="#FFF", stroke="#E65100", fs=8, bold_first=True, color="#E65100", rx=6)
    arrow(g_i2, cx, y + 8, cx, y + 38, color="#E65100", marker="ah_orange", sw=1.2)

    # ── Right-side annotations ───────────────────────────────
    right_notes = [
        (72, "支持变长序列批处理\npadding_mask 机制\n排除填充帧的梯度贡献"),
        (128, "三层 Conv1D 逐级精炼:\n 7核 → 宽频带谐波包络\n 5核 → 中频谐波间距\n 3核 → 紧邻谐波峰峰值\n残差连接防梯度退化"),
        (203, "若输入特征维度 ≠ 257\n(如 Mel 40维)\n先经 freq_expand 升维"),
        (255, "sin/cos 编码注入\n时序位置信息\n使注意力感知帧间顺序"),
        (333, "每层包含两个子层:\n① 多头自注意力 (h=4)\n② 前馈网络 (d_ff=512)\n各子层后: Add & LayerNorm"),
        (510, "幅度掩膜 (本文方案)\nSigmoid → [0,1] 约束\n与含噪幅度逐点相乘\n复用含噪相位 iSTFT"),
    ]
    for ny, ntext in right_notes:
        g_n = ET.SubElement(s, "g")
        ET.SubElement(g_n, "line", {
            "x1": str(cx + bw / 2 + 8), "y1": str(ny),
            "x2": str(rx - 10), "y2": str(ny),
            "stroke": "#CFD8DC", "stroke-width": "0.8", "stroke-dasharray": "3,3",
        })
        for j, line in enumerate(ntext.split("\n")):
            text(g_n, rx, ny + j * 14, line, fs=8, color="#546E7A", anchor="start")

    # ── Legend ────────────────────────────────────────────────
    ly = y + 105
    txt_leg = ET.SubElement(s, "g")
    text(txt_leg, 20, ly + 12, "图例:", fs=9, bold=True, anchor="start")
    items = [
        ("输入/数据", "#E3F2FD"), ("卷积处理", "#E1F5FE"), ("位置编码", "#80DEEA"),
        ("Transformer", "#FFF8E1"), ("注意力子层", "#FFECB3"), ("输出/掩膜", "#E8F5E9"),
    ]
    for i, (nm, clr) in enumerate(items):
        lx = 75 + i * 125
        g_lg = ET.SubElement(s, "g")
        rect(g_lg, lx, ly, 14, 14, rx=2, fill=clr, stroke="#999", sw=0.8, shadow=False)
        text(g_lg, lx + 20, ly + 10, nm, fs=8, color="#555", anchor="start")

    # Save
    ET.ElementTree(s).write("outputs/plots/model_architecture.svg", encoding="utf-8", xml_declaration=True)
    print("Saved: outputs/plots/model_architecture.svg")


if __name__ == "__main__":
    draw_system_flow()
    draw_model_arch()
    print("Done — refined SVG diagrams generated.")
