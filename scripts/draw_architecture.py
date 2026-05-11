"""Generate TransformerDenoiser architecture diagram for thesis."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Use Chinese font available on Windows
for fname in ["Microsoft YaHei", "SimHei", "SimSun", "Arial Unicode MS"]:
    try:
        fm.findfont(fname, fallback_to_default=False)
        plt.rcParams["font.family"] = fname
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(1, 1, figsize=(10, 14))
ax.set_xlim(0, 10)
ax.set_ylim(0, 14)
ax.axis("off")

# Color scheme
c_input = "#E3F2FD"
c_conv = "#BBDEFB"
c_pos = "#90CAF9"
c_transformer = "#FFE0B2"
c_attention = "#FFCC80"
c_ffn = "#FFB74D"
c_output = "#C8E6C9"
c_mask = "#A5D6A7"
c_final = "#81C784"
c_arrow = "#546E7A"

box_style = dict(boxstyle="round,pad=0.3", edgecolor="#37474F", linewidth=1.5)

def draw_box(ax, x, y, w, h, text, color, fontsize=9, bold=False):
    """Draw a rounded box with text."""
    weight = "bold" if bold else "normal"
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.15", facecolor=color,
                          edgecolor="#37474F", linewidth=1.5, zorder=2)
    ax.add_patch(rect)
    lines = text.split("\n")
    for i, line in enumerate(lines):
        fs = fontsize - 1 if i > 0 else fontsize
        ax.text(x, y + (len(lines)-1)*0.18 - i*0.36, line,
                ha="center", va="center", fontsize=fs, fontweight=weight,
                fontfamily="Microsoft YaHei", zorder=3)

def draw_arrow(ax, x1, y1, x2, y2, label="", color=c_arrow):
    """Draw downward arrow between boxes."""
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.8), zorder=1)
    if label:
        ax.text(x1 + 0.6, (y1 + y2) / 2, label, fontsize=7, color="#37474F",
                va="center", fontfamily="Microsoft YaHei")

# === Left column: modules ===
cx = 4.5
y_start = 13.5

# Input
draw_box(ax, cx, 13.0, 3.6, 0.8,
         "输入特征  [B, T, F]  F∈{257,40,44}",
         c_input, fontsize=8, bold=True)

# ConvFrontend
draw_box(ax, cx, 11.6, 3.6, 0.8,
         "Conv1D 频率前端\nkernel: 7→5→3  |  channels: 64  |  ReLU+BN",
         c_conv, fontsize=7)

# Residual + sign
ax.text(cx + 2.1, 11.2, "+", fontsize=14, fontweight="bold", color="#37474F", zorder=3)

# Input projection
draw_box(ax, cx, 10.2, 3.6, 0.8,
         "输入投影  Linear(F → 128)  +  ReLU(freq_expand)",
         c_conv, fontsize=7)

# Positional Encoding
draw_box(ax, cx, 8.8, 3.6, 0.8,
         "正弦位置编码  PositionalEncoding(d=128, max_len=8000)",
         c_pos, fontsize=7)

# Transformer Encoder box
draw_box(ax, cx, 7.0, 4.4, 2.4,
         "Transformer Encoder (×2 layers)",
         c_transformer, fontsize=8, bold=True)

# Layer 1
draw_box(ax, cx - 0.2, 6.1, 3.4, 0.8,
         "第1层: 4-Head Self-Attention → LayerNorm\n   → FFN(512) → LayerNorm",
         c_attention, fontsize=7)

# Layer 2
draw_box(ax, cx - 0.2, 5.0, 3.4, 0.8,
         "第2层: 4-Head Self-Attention → LayerNorm\n   → FFN(512) → LayerNorm",
         c_attention, fontsize=7)

# Output projection
draw_box(ax, cx, 3.6, 3.6, 0.8,
         "输出投影  Linear(128 → F)\n幅度掩膜: F=257  |  复数掩膜: F=514",
         c_output, fontsize=7)

# Activation
draw_box(ax, cx, 2.2, 3.6, 0.8,
         "激活函数\n幅度掩膜: Sigmoid → [0,1]  |  复数掩膜: Identity → ℝ",
         c_mask, fontsize=7)

# Output
draw_box(ax, cx, 0.8, 3.6, 0.8,
         "输出掩膜  [B, T, F]\n(增强幅度 = 掩膜 × 含噪幅度)",
         c_final, fontsize=8, bold=True)

# === Arrows ===
draw_arrow(ax, cx, 12.6, cx, 12.0)
draw_arrow(ax, cx, 11.2, cx, 10.6)
draw_arrow(ax, cx, 9.4, cx, 9.2)
draw_arrow(ax, cx, 8.4, cx, 8.2)
draw_arrow(ax, cx, 6.2, cx, 5.4)  # Encoder → Layer1
draw_arrow(ax, cx, 4.2, cx, 4.0)
draw_arrow(ax, cx, 2.8, cx, 2.6)
draw_arrow(ax, cx, 1.8, cx, 1.2)

# Between layers
ax.annotate("", xy=(cx - 0.2, 5.4), xytext=(cx - 0.2, 5.8),
            arrowprops=dict(arrowstyle="->", color=c_arrow, lw=1.3), zorder=1)

# Residual connection arrow (dashed)
ax.annotate("", xy=(cx + 2.1, 11.2), xytext=(cx + 2.1, 10.6),
            arrowprops=dict(arrowstyle="->", color="#E65100", lw=1.3,
                          linestyle="dashed"), zorder=1)
ax.text(cx + 2.7, 10.9, "残差连接", fontsize=6, color="#E65100",
        va="center", fontfamily="Microsoft YaHei")

# === Right column: annotations ===
rx = 8.5
annotations = [
    (12.8, "维度: 时间帧数 T × 频率bin数 F\nF=257 (STFT) | 40 (Mel) | 44 (RNNoise)"),
    (10.4, "沿频率轴的3层一维卷积\n捕捉谐波结构 (归纳偏置)"),
    (8.6, "将任意维度特征映射至\nd_model=128的统一表示空间"),
    (6.8, "sin/cos编码注入时序位置信息\n使自注意力获得顺序感知"),
    (4.8, "多头自注意力捕获全局帧间依赖\nFFN引入非线性增强表达能力\n残差连接+LayerNorm防梯度退化"),
    (2.4, "Sigmoid: 掩膜值∈[0,1]\n0=纯噪声抑制, 1=纯语音保留"),
    (0.6, "掩膜应用到含噪幅度\n复用含噪相位 → iSTFT → 增强语音"),
]

for y, text in annotations:
    ax.text(rx, y, text, fontsize=7, color="#455A64", va="center",
            fontfamily="Microsoft YaHei", linespacing=1.4)

# Connectivity lines from main box to annotations
for y in [12.8, 10.4, 8.6, 6.8, 4.8, 2.4, 0.6]:
    ax.plot([7.5, 7.9], [y, y], color="#B0BEC5", linewidth=0.8, zorder=0)

# === Title ===
ax.text(5, 13.9, "TransformerDenoiser 模型架构图",
        ha="center", fontsize=14, fontweight="bold", fontfamily="Microsoft YaHei")

# === Legend ===
legend_y = 0.1
ax.text(0.2, legend_y + 0.2, "图例:", fontsize=7, fontweight="bold")
for name, color, xoff in [("输入/特征", c_input, 0.7), ("卷积处理", c_conv, 1.8),
                            ("位置编码", c_pos, 2.9), ("Transformer", c_transformer, 3.9),
                            ("输出/掩膜", c_output, 5.4), ("激活函数", c_mask, 6.5)]:
    rect = FancyBboxPatch((xoff, legend_y), 0.4, 0.2,
                          boxstyle="round,pad=0.05", facecolor=color,
                          edgecolor="#37474F", linewidth=0.8)
    ax.add_patch(rect)
    ax.text(xoff + 0.5, legend_y + 0.1, name, fontsize=6, va="center")

plt.tight_layout(pad=0.5)
plt.savefig("outputs/plots/model_architecture.png", dpi=200, bbox_inches="tight",
            facecolor="white", edgecolor="none")
print("Saved: outputs/plots/model_architecture.png")
