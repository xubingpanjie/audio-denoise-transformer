"""Generate Visio-compatible SVG diagrams: system flow + model architecture."""
import xml.etree.ElementTree as ET

def svg_elem(tag, attrib=None, text="", **extra):
    el = ET.Element(tag, attrib or {})
    for k, v in extra.items():
        el.set(k.replace("_", "-"), str(v))
    if text:
        el.text = text
    return el

def add_rect(svg, x, y, w, h, fill, stroke="#333", rx=6, text="", fs=13, bold=False, color="#222"):
    """Draw a Visio-style rounded rect with label."""
    g = ET.SubElement(svg, "g")
    ET.SubElement(g, "rect", {
        "x": str(x), "y": str(y), "width": str(w), "height": str(h),
        "rx": str(rx), "ry": str(rx),
        "fill": fill, "stroke": stroke, "stroke-width": "1.5",
    })
    lines = text.split("\n")
    cy = y + h / 2 - (len(lines) - 1) * 8
    for i, line in enumerate(lines):
        fw = "bold" if (bold and i == 0) else "normal"
        ET.SubElement(g, "text", {
            "x": str(x + w / 2), "y": str(cy + i * 16),
            "text-anchor": "middle", "dominant-baseline": "middle",
            "font-family": "Microsoft YaHei, SimHei, sans-serif",
            "font-size": str(fs - (1 if i > 0 else 0)),
            "font-weight": fw, "fill": color,
        }).text = line
    return g

def add_arrow(svg, x1, y1, x2, y2, color="#555", label=""):
    """Draw a downward arrow."""
    g = ET.SubElement(svg, "g")
    mid_y = (y1 + y2) / 2
    ET.SubElement(g, "line", {
        "x1": str(x1), "y1": str(y1), "x2": str(x2), "y2": str(y2),
        "stroke": color, "stroke-width": "2",
        "marker-end": "url(#arrowhead)",
    })
    if label:
        ET.SubElement(g, "text", {
            "x": str(x1 + 18), "y": str(mid_y),
            "font-family": "Microsoft YaHei, SimHei, sans-serif",
            "font-size": "10", "fill": "#666",
        }).text = label
    return g

def add_dashed_connector(svg, x1, y1, x2, y2, color="#E65100", label=""):
    """Draw a dashed connector for residual/feedback."""
    g = ET.SubElement(svg, "g")
    mid_y = (y1 + y2) / 2
    ET.SubElement(g, "line", {
        "x1": str(x1), "y1": str(y1), "x2": str(x2), "y2": str(y2),
        "stroke": color, "stroke-width": "1.5", "stroke-dasharray": "6,4",
    })
    if label:
        ET.SubElement(g, "text", {
            "x": str(x2 + 10), "y": str(mid_y),
            "font-family": "Microsoft YaHei, SimHei, sans-serif",
            "font-size": "10", "fill": color,
        }).text = label
    return g

# ============================================================
# FIGURE 1: System Flow Diagram (6-stage pipeline)
# ============================================================

def generate_system_flow():
    svg = ET.Element("svg", {
        "xmlns": "http://www.w3.org/2000/svg",
        "viewBox": "0 0 1000 520",
        "width": "1000", "height": "520",
        "font-family": "Microsoft YaHei, SimHei, sans-serif",
    })

    # Defs: arrowhead marker
    defs = ET.SubElement(svg, "defs")
    marker = ET.SubElement(defs, "marker", {
        "id": "arrowhead", "markerWidth": "10", "markerHeight": "7",
        "refX": "9", "refY": "3.5", "orient": "auto",
    })
    ET.SubElement(marker, "polygon", {
        "points": "0 0, 10 3.5, 0 7", "fill": "#555",
    })

    # Title
    ET.SubElement(svg, "text", {
        "x": "500", "y": "30", "text-anchor": "middle",
        "font-size": "18", "font-weight": "bold", "fill": "#1a1a1a",
    }).text = "图3-1 系统整体流水线架构"

    # Colors
    colors = ["#E3F2FD", "#BBDEFB", "#C8E6C9", "#FFF9C4", "#FFE0B2", "#F3E5F5"]

    # Stage boxes (2 rows x 3 cols)
    stages = [
        ("阶段一\n数据构建", "干净语音 + 噪声\nRMS归一化混合\n4 SNR: -5/0/5/10 dB\n输出: metadata.csv"),
        ("阶段二\n特征提取", "STFT (n_fft=512, hop=128)\n对数幅度谱 257维\nMel频谱 40维\nRNNoise风格 44维"),
        ("阶段三\n模型训练", "TransformerDenoiser\n频域加权L1 Loss\n余弦退火LR调度\n30 Epochs, Adam"),
        ("阶段四\n模型推理", "加载checkpoint\n预测幅度掩膜\n掩膜×含噪幅度\nISTFT重建波形"),
        ("阶段五\n基线对比", "RNNoise\nGTCRN (ONNX)\nnoisereduce\n含噪不处理基线"),
        ("阶段六\n评价分析", "STOI 可懂度\nSDR 信号失真比\n损失曲线图\n指标柱状图"),
    ]

    for i, (title, desc) in enumerate(stages):
        row = i // 3
        col = i % 3
        x = 40 + col * 320
        y = 60 + row * 220

        # Module box
        add_rect(svg, x, y, 120, 60, colors[i], text=title, fs=13, bold=True)
        # Detail box
        add_rect(svg, x, y + 75, 280, 120, "#FAFAFA", stroke="#CCC", rx=4,
                 text=desc, fs=10, color="#555")

        # Arrow between module and detail
        ET.SubElement(svg, "line", {
            "x1": str(x + 60), "y1": str(y + 60), "x2": str(x + 60), "y2": str(y + 75),
            "stroke": "#999", "stroke-width": "1.2",
            "marker-end": "url(#arrowhead)",
        })

    # Arrows between stage columns
    for col in range(2):
        x1 = 180 + col * 320
        x2 = 360 + col * 320
        y = 90
        ET.SubElement(svg, "line", {
            "x1": str(x1), "y1": str(y), "x2": str(x2), "y2": str(y),
            "stroke": "#78909C", "stroke-width": "2.5",
            "marker-end": "url(#arrowhead)",
        })
        # Arrow label
        if col == 0:
            label = "混合音频"
        else:
            label = "特征向量"
        ET.SubElement(svg, "text", {
            "x": str((x1 + x2) / 2), "y": str(y - 10),
            "text-anchor": "middle", "font-size": "10", "fill": "#78909C",
        }).text = label

    # Arrows between rows (wrap around)
    for col in range(3):
        x = 730  # right edge of last stage
        y1 = 170 + col * 0
    for col in range(3):
        x_start = 320 + col * 320
        x_end = 40 + ((col + 1) % 3) * 320
        y_connect = 400

    # Row 1→2 flow arrows
    arrows = [
        (680, 200, 360, 280, "含噪+干净配对"),
        (360, 200, 360, 280, "幅度谱+相位"),
        (680, 200, 680, 280, ""),
    ]
    for x1, y1, x2, y2, lbl in arrows:
        add_arrow(svg, x1, y1, x2, y2, label=lbl)

    # Simple flow indicators between row stages
    for i in range(3):
        x = 180 + i * 320
        ET.SubElement(svg, "line", {
            "x1": str(x), "y1": str(280), "x2": str(x), "y2": str(340),
            "stroke": "#BDBDBD", "stroke-width": "2",
            "marker-end": "url(#arrowhead)",
        })

    # Data flow labels
    labels_row2 = ["checkpoint .pt", "增强 .wav", "对照 .wav"]
    for i, lbl in enumerate(labels_row2):
        x = 180 + i * 320
        ET.SubElement(svg, "text", {
            "x": str(x + 15), "y": "305", "font-size": "9", "fill": "#888",
        }).text = lbl

    # Save
    tree = ET.ElementTree(svg)
    tree.write("outputs/plots/system_flow.svg", encoding="utf-8", xml_declaration=True)
    print("Saved: outputs/plots/system_flow.svg")


# ============================================================
# FIGURE 2: TransformerDenoiser Architecture Detail
# ============================================================

def generate_model_arch():
    svg = ET.Element("svg", {
        "xmlns": "http://www.w3.org/2000/svg",
        "viewBox": "0 0 680 880",
        "width": "680", "height": "880",
        "font-family": "Microsoft YaHei, SimHei, sans-serif",
    })

    # Defs
    defs = ET.SubElement(svg, "defs")
    marker = ET.SubElement(defs, "marker", {
        "id": "arrow", "markerWidth": "10", "markerHeight": "7",
        "refX": "9", "refY": "3.5", "orient": "auto",
    })
    ET.SubElement(marker, "polygon", {
        "points": "0 0, 10 3.5, 0 7", "fill": "#455A64",
    })

    # Title
    ET.SubElement(svg, "text", {
        "x": "340", "y": "28", "text-anchor": "middle",
        "font-size": "16", "font-weight": "bold", "fill": "#1a1a1a",
    }).text = "图3-2 TransformerDenoiser 模型架构详图"

    cx = 340  # center x for the main column
    rx = 520  # annotation column

    # Color palette
    C = {
        "input": "#E3F2FD",
        "conv": "#BBDEFB",
        "proj": "#90CAF9",
        "pos": "#81D4FA",
        "enc": "#FFE0B2",
        "attn": "#FFCC80",
        "ffn": "#FFB74D",
        "out": "#C8E6C9",
        "act": "#A5D6A7",
        "final": "#81C784",
    }

    y_positions = []

    def box(y, h, w, text, color, bold=False, fs=11):
        y_positions.append(y)
        add_rect(svg, cx - w / 2, y, w, h, color, text=text, fs=fs, bold=bold)
        return y + h

    def arrow_to(y, label=""):
        prev_y = y_positions[-1]
        prev_h = 50 if not y_positions else 50
        # Find previous box
        prev_bottom = y_positions[-1] + (50 if len(y_positions) > 1 else 50)
        add_arrow(svg, cx, y_positions[-1] + 50, cx, y, label=label)

    # Build boxes top to bottom
    y = 55

    add_rect(svg, cx - 170, y, 340, 48, C["input"],
             text="输入特征  [Batch, Time, Freq]\nFreq ∈ {257 (STFT), 40 (Mel), 44 (RNNoise)}",
             fs=10, bold=True)
    y += 48
    add_arrow(svg, cx, y, cx, y + 10)
    y += 10

    add_rect(svg, cx - 170, y, 340, 52, C["conv"],
             text="Conv1D 频率前端  (创新点 1)\nkernel 7→5→3  |  channels=64  |  ReLU + BatchNorm",
             fs=10, bold=True)
    y += 52
    add_arrow(svg, cx, y, cx, y + 8)
    y += 8

    # Residual connection indicator
    add_rect(svg, cx + 185, y - 30, 95, 22, "#FFF3E0", stroke="#E65100", rx=3,
             text="+ 残差连接", fs=9, color="#E65100")
    add_dashed_connector(svg, cx + 175, y - 45, cx + 185, y - 18)

    add_rect(svg, cx - 170, y, 340, 44, C["proj"],
             text="输入投影  Linear(Freq → d_model=128)",
             fs=11, bold=False)
    y += 44
    add_arrow(svg, cx, y, cx, y + 8)
    y += 8

    add_rect(svg, cx - 170, y, 340, 44, C["pos"],
             text="正弦位置编码  PositionalEncoding(d=128, max_len=8000)",
             fs=11)
    y += 44
    add_arrow(svg, cx, y, cx, y + 8)
    y += 8

    # === Transformer Encoder Container ===
    enc_y = y
    add_rect(svg, cx - 190, y, 380, 44, C["enc"],
             text="Transformer Encoder (×2 layers, nhead=4)",
             fs=11, bold=True)
    y += 44

    # Layer 1
    add_rect(svg, cx - 170, y, 340, 40, C["attn"],
             text="第 1 层:  4-Head Self-Attention  →  LayerNorm  →  FFN(512)  →  LayerNorm",
             fs=9)
    y += 40
    # Inter-layer arrow
    ET.SubElement(svg, "line", {
        "x1": str(cx), "y1": str(y), "x2": str(cx), "y2": str(y + 6),
        "stroke": "#90A4AE", "stroke-width": "1.3",
    })

    # Layer 2
    add_rect(svg, cx - 170, y + 6, 340, 40, C["attn"],
             text="第 2 层:  4-Head Self-Attention  →  LayerNorm  →  FFN(512)  →  LayerNorm",
             fs=9)
    y += 46
    # Encoder bottom line
    add_rect(svg, cx - 190, enc_y, 380, y - enc_y, "none", stroke="#FF9800", rx=8)

    add_arrow(svg, cx, y, cx, y + 8)
    y += 8

    # Output projection
    add_rect(svg, cx - 170, y, 340, 44, C["out"],
             text="输出投影  Linear(d_model → Freq)\n幅度掩膜: 257  |  复数掩膜 (探索): 514",
             fs=10)
    y += 44
    add_arrow(svg, cx, y, cx, y + 8)
    y += 8

    # Activation
    add_rect(svg, cx - 170, y, 340, 44, C["act"],
             text="激活函数\n幅度掩膜: Sigmoid → [0, 1]  |  复数掩膜: Identity → ℝ",
             fs=10)
    y += 44
    add_arrow(svg, cx, y, cx, y + 8)
    y += 8

    # Final output
    add_rect(svg, cx - 170, y, 340, 48, C["final"],
             text="输出掩膜  [B, T, F]\n增强信号 = ISTFT(掩膜 × 含噪复数谱)",
             fs=10, bold=True)

    # === Right-side annotations ===
    notes = [
        (68, "T: 时间帧数 (不定长, ~80-500帧)\nF: 频率bin数\n支持变长序列批处理 (padding mask)"),
        (122, "Conv1D 沿频率轴独立处理每帧\n3层卷积逐级精炼:\n  · 7核 → 宽频带能量包络\n  · 5核 → 中频谐波间距\n  · 3核 → 紧邻谐波峰\n残差连接防止梯度退化"),
        (208, "将任意维度特征映射至\nd_model=128 统一空间"),
        (256, "sin/cos编码注入时序位置信息\n使无顺序感的自注意力\n获得帧间先后关系"),
        (308, "核心计算单元 (×2):\n多头自注意力 → 全局帧依赖\n  · 每帧可关注任意距离帧\n  · 4个子空间并行学习\nFFN → 非线性增强表达\nLayerNorm → 稳定训练"),
        (455, "掩膜值 ∈ [0,1]\n0 = 纯噪声 (抑制)\n1 = 纯语音 (保留)\n→ 与含噪幅度逐点相乘"),
    ]

    for note_y, note_text in notes:
        ET.SubElement(svg, "line", {
            "x1": str(cx + 175), "y1": str(note_y + 20),
            "x2": str(rx - 15), "y2": str(note_y + 20),
            "stroke": "#B0BEC5", "stroke-width": "0.8", "stroke-dasharray": "3,3",
        })
        ET.SubElement(svg, "text", {
            "x": str(rx), "y": str(note_y + 5),
            "font-size": "8.5", "fill": "#546E7A",
        }).text = note_text

    # Legend
    ly = 850
    legend_items = [
        ("输入/提取", C["input"]), ("卷积处理", C["conv"]),
        ("投影/位置编码", C["proj"]), ("Transformer Encoder", C["enc"]),
        ("注意力/FFN", C["attn"]), ("输出/掩膜", C["final"]),
    ]
    ET.SubElement(svg, "text", {
        "x": "30", "y": str(ly + 12), "font-size": "9", "font-weight": "bold",
    }).text = "图例:"
    for i, (name, col) in enumerate(legend_items):
        lx = 70 + i * 100
        ET.SubElement(svg, "rect", {
            "x": str(lx), "y": str(ly), "width": "16", "height": "12",
            "rx": "3", "fill": col, "stroke": "#999", "stroke-width": "0.8",
        })
        ET.SubElement(svg, "text", {
            "x": str(lx + 20), "y": str(ly + 10),
            "font-size": "8", "fill": "#555",
        }).text = name

    # Save
    tree = ET.ElementTree(svg)
    tree.write("outputs/plots/model_architecture.svg", encoding="utf-8", xml_declaration=True)
    print("Saved: outputs/plots/model_architecture.svg")


if __name__ == "__main__":
    generate_system_flow()
    generate_model_arch()
    print("Done. Open SVG files in Visio: File → Open → select .svg")
    print("Visio will convert to editable shapes automatically.")
