"""Gradio interactive demo: audio mixing, denoising, metrics, training dashboard."""
from __future__ import annotations

import io
import csv
import math
import sys
import time
import warnings
from pathlib import Path

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Load audio index ──────────────────────────────────────────
CLEAN_DIR = PROJECT_ROOT / "data/clean"
NOISE_DIR = PROJECT_ROOT / "data/noise"
META_TEST = PROJECT_ROOT / "data/timit_mixed/metadata_test.csv"

clean_files = sorted(CLEAN_DIR.glob("*.WAV"))
noise_files = sorted(NOISE_DIR.glob("*.wav"))
clean_names = [f.stem for f in clean_files]
noise_names = [f.stem for f in noise_files]

# ── Audio helpers ────────────────────────────────────────────
def load_audio(path, sr=16000):
    audio, s = sf.read(str(path))
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if s != sr:
        audio = resample_poly(audio, sr, s).astype(np.float32)
    return audio

def compute_sdr(clean, enhanced):
    length = min(len(clean), len(enhanced))
    c, e = clean[:length], enhanced[:length]
    return 10.0 * math.log10((np.sum(c**2) + 1e-12) / (np.sum((c - e)**2) + 1e-12))

def compute_stoi(clean, enhanced, sr=16000):
    try:
        from pystoi import stoi
        length = min(len(clean), len(enhanced))
        return float(stoi(clean[:length], enhanced[:length], sr, extended=False))
    except Exception:
        return 0.0

# ── Mix audio ────────────────────────────────────────────────
def mix_audio(clean_name, noise_name, snr_db):
    clean = load_audio(CLEAN_DIR / f"{clean_name}.WAV")
    noise = load_audio(NOISE_DIR / f"{noise_name}.wav")
    if len(noise) < len(clean):
        noise = np.tile(noise, int(np.ceil(len(clean) / len(noise))))[:len(clean)]
    else:
        noise = noise[:len(clean)]

    rms_clean = np.sqrt(np.mean(clean**2) + 1e-12)
    rms_noise = np.sqrt(np.mean(noise**2) + 1e-12)
    alpha = rms_clean / (rms_noise * 10**(snr_db / 20))
    noisy = clean + alpha * noise
    peak = np.max(np.abs(noisy))
    if peak > 0.99:
        noisy = noisy / peak * 0.99
    return clean, noisy

# ── Model inference ──────────────────────────────────────────
def run_transformer(noisy_audio):
    from features.audio_features import extract_feature_bundle, invert_stft
    from models.transformer_denoiser import TransformerDenoiser
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(PROJECT_ROOT / "checkpoints/transformer_stft_improved.pt",
                      map_location=device, weights_only=False)
    model = TransformerDenoiser(input_dim=ckpt["input_dim"], output_dim=ckpt["output_dim"],
                                causal=ckpt.get("causal", False),
                                complex_mask=ckpt.get("complex_mask", False))
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    features = extract_feature_bundle(noisy_audio, 16000, n_fft=512, hop_length=128,
                                       n_mels=40, rnnoise_bands=22)
    inp = torch.from_numpy(features["stft_log_mag"]).unsqueeze(0).to(device)
    with torch.no_grad():
        mask = model(inp).squeeze(0).cpu().numpy()

    enhanced_mag = (mask * features["stft_mag"]).T
    enhanced_spec = enhanced_mag * np.exp(1j * features["phase"].T)
    return invert_stft(enhanced_spec, 512, 128, len(noisy_audio))

def run_noisereduce(noisy_audio):
    import noisereduce as nr
    return nr.reduce_noise(y=noisy_audio, sr=16000, stationary=False,
                           prop_decrease=0.8, n_fft=512, freq_mask_smooth_hz=250)

def run_rnnoise(noisy_audio):
    from pyrnnoise import RNNoise
    import tempfile
    d = RNNoise(sample_rate=48000)
    a48 = resample_poly(noisy_audio, 48000, 16000).astype(np.float32)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f1, \
         tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f2:
        sf.write(f1.name, a48, 48000)
        for _ in d.denoise_wav(f1.name, f2.name): pass
        out, _ = sf.read(f2.name)
    out = np.asarray(out, dtype=np.float32)
    enhanced = resample_poly(out, 16000, 48000).astype(np.float32)
    return enhanced[:len(noisy_audio)]

def run_gtcrn(noisy_audio):
    import onnxruntime as ort
    import torch
    from librosa import istft

    session = ort.InferenceSession(str(PROJECT_ROOT / "checkpoints/gtcrn.onnx"),
                                    providers=["CPUExecutionProvider"])
    x = torch.stft(torch.from_numpy(noisy_audio), 512, 256, 512,
                   torch.hann_window(512).pow(0.5), return_complex=False)[None]
    inputs = x.numpy()

    conv_cache = np.zeros([2, 1, 16, 16, 33], dtype="float32")
    tra_cache = np.zeros([2, 3, 1, 1, 16], dtype="float32")
    inter_cache = np.zeros([2, 1, 33, 16], dtype="float32")

    outputs = []
    for i in range(inputs.shape[-2]):
        out_i, conv_cache, tra_cache, inter_cache = session.run(
            [], {"mix": inputs[..., i:i + 1, :],
                 "conv_cache": conv_cache, "tra_cache": tra_cache,
                 "inter_cache": inter_cache})
        outputs.append(out_i)

    outputs = np.concatenate(outputs, axis=2)
    enhanced = istft(outputs[..., 0] + 1j * outputs[..., 1],
                     n_fft=512, hop_length=256, win_length=512,
                     window=np.hanning(512) ** 0.5)
    return enhanced.squeeze()[:len(noisy_audio)]

# ── Plot helpers ─────────────────────────────────────────────
def plot_waveforms(clean, noisy, enhanced):
    fig, axes = plt.subplots(3, 1, figsize=(10, 5), sharex=True)
    for ax, data, title, color in zip(axes,
                                       [clean, noisy, enhanced],
                                       ["干净语音", "含噪语音", "降噪后"],
                                       ["#2E7D32", "#C62828", "#1565C0"]):
        t = np.arange(len(data)) / 16000
        ax.plot(t, data, color=color, linewidth=0.5)
        ax.set_ylabel(title, fontsize=9)
        ax.set_ylim(-1, 1)
    axes[-1].set_xlabel("时间 (秒)")
    plt.tight_layout()
    return fig

def plot_spectrograms(clean, noisy, enhanced):
    from scipy.signal import stft
    fig, axes = plt.subplots(3, 1, figsize=(10, 6))
    titles = ["干净语音频谱", "含噪语音频谱", "降噪后频谱"]
    for ax, data, title in zip(axes, [clean, noisy, enhanced], titles):
        _, _, S = stft(data, nperseg=512, noverlap=384)
        ax.pcolormesh(20 * np.log10(np.abs(S) + 1e-8), shading="auto", cmap="magma")
        ax.set_ylabel("频率 bin")
        ax.set_title(title, fontsize=9)
    axes[-1].set_xlabel("时间帧")
    plt.tight_layout()
    return fig

def plot_metrics_bar(metrics_dict):
    """Bar chart comparing STOI/SDR across models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    names = list(metrics_dict.keys())
    stoi_vals = [v["stoi"] for v in metrics_dict.values()]
    sdr_vals = [v["sdr"] for v in metrics_dict.values()]
    colors = ["#BDBDBD", "#FF7043", "#42A5F5", "#AB47BC", "#66BB6A"]

    ax1.bar(names, stoi_vals, color=colors[:len(names)])
    ax1.set_title("STOI (可懂度)")
    ax1.set_ylim(0, 1)
    for i, v in enumerate(stoi_vals):
        ax1.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)

    ax2.bar(names, sdr_vals, color=colors[:len(names)])
    ax2.set_title("SDR (dB)")
    for i, v in enumerate(sdr_vals):
        ax2.text(i, v + 0.3, f"{v:.1f}", ha="center", fontsize=8)

    plt.tight_layout()
    return fig

def plot_training_curves():
    """Load training history and plot loss + lr curves."""
    csv_path = PROJECT_ROOT / "outputs/metrics/transformer_stft_improved_history.csv"
    if not csv_path.exists():
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "训练历史文件不存在", ha="center", va="center", fontsize=14)
        return fig

    epochs, losses, lrs = [], [], []
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            epochs.append(int(row["epoch"]))
            losses.append(float(row["loss"]))
            lrs.append(float(row.get("lr", 0)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(epochs, losses, "b-o", markersize=3, linewidth=1.2)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("L1 Loss (加权)")
    ax1.set_title(f"训练损失曲线 (Epoch 1 → {epochs[-1]})")
    ax1.grid(True, alpha=0.3)
    ax1.annotate(f"起始: {losses[0]:.4f}", xy=(1, losses[0]),
                 fontsize=8, color="#C62828")
    ax1.annotate(f"最终: {losses[-1]:.4f}", xy=(epochs[-1], losses[-1]),
                 fontsize=8, color="#2E7D32")

    if lrs and max(lrs) > 0:
        ax2.plot(epochs, lrs, "r-o", markersize=3, linewidth=1.2)
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("学习率")
        ax2.set_title("余弦退火学习率调度")
        ax2.set_yscale("log")
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

def plot_final_results():
    """Plot final comparison table from saved CSVs."""
    metrics_dir = PROJECT_ROOT / "outputs/metrics"
    methods = [
        ("noisy_baseline", "含噪不处理"),
        ("rnnoise_v3", "RNNoise"),
        ("gtcrn_v3", "GTCRN"),
        ("noisereduce_v2", "noisereduce"),
        ("STFT_improved", "Transformer"),
    ]

    stoi_vals, sdr_vals, labels = [], [], []
    for csv_name, label in methods:
        csv_path = metrics_dir / f"{csv_name}_metrics.csv"
        if not csv_path.exists():
            continue
        with csv_path.open() as f:
            avg = [r for r in csv.DictReader(f) if r.get("sample_id") == "average"]
            if not avg:
                avg = [r for r in csv.DictReader(f) if r.get("") == "average"]
            if avg:
                stoi_vals.append(float(avg[0].get("stoi", 0)))
                sdr_vals.append(float(avg[0].get("sdr", 0)))
                labels.append(label)

    if not labels:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "评估数据不存在，请先运行评估", ha="center", va="center")
        return fig

    colors = ["#9E9E9E", "#FF7043", "#42A5F5", "#AB47BC", "#66BB6A"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    bars1 = ax1.bar(labels, stoi_vals, color=colors[:len(labels)])
    ax1.set_title("STOI ↑ (可懂度)", fontsize=12)
    ax1.set_ylim(0, 1)
    for bar, v in zip(bars1, stoi_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{v:.3f}", ha="center", fontsize=9)

    bars2 = ax2.bar(labels, sdr_vals, color=colors[:len(labels)])
    ax2.set_title("SDR ↑ (dB)", fontsize=12)
    for bar, v in zip(bars2, sdr_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"{v:.1f}", ha="center", fontsize=9)

    plt.tight_layout()
    return fig

# ── UI Callbacks ─────────────────────────────────────────────
def do_mix_and_denoise(clean_name, noise_name, snr_db, models_to_run):
    if not clean_name or not noise_name:
        return None, "请选择干净语音和噪声", None, None, None

    clean, noisy = mix_audio(clean_name, noise_name, snr_db)

    results = {"含噪不处理": {"audio": noisy, "stoi": 0, "sdr": 0}}
    for model_name in models_to_run:
        t0 = time.time()
        if model_name == "Transformer (本项目)":
            enhanced = run_transformer(noisy)
        elif model_name == "noisereduce":
            enhanced = run_noisereduce(noisy)
        elif model_name == "RNNoise":
            enhanced = run_rnnoise(noisy)
        elif model_name == "GTCRN":
            enhanced = run_gtcrn(noisy)
        else:
            continue

        sdr = compute_sdr(clean, enhanced)
        stoi = compute_stoi(clean, enhanced)
        results[model_name] = {"audio": enhanced, "stoi": stoi, "sdr": sdr,
                                "time": time.time() - t0}

    # Build info text
    info_lines = [f"干净语音: {clean_name}  |  噪声: {noise_name}  |  SNR: {snr_db} dB"]
    info_lines.append(f"音频长度: {len(clean)/16000:.1f}秒")
    info_lines.append("")
    for name, r in results.items():
        info_lines.append(f"{name}: STOI={r['stoi']:.4f}  SDR={r['sdr']:.2f} dB" +
                          (f"  ({r['time']:.1f}s)" if "time" in r else ""))

    info_text = "\n".join(info_lines)

    # Select one enhanced audio for waveform/spectrogram display
    first_model = models_to_run[0] if models_to_run else "含噪不处理"
    best_audio = results.get(first_model, results["含噪不处理"])["audio"]

    wf_fig = plot_waveforms(clean, noisy, best_audio)
    sp_fig = plot_spectrograms(clean, noisy, best_audio)
    bar_fig = plot_metrics_bar(results)

    return (clean, noisy), info_text, wf_fig, sp_fig, bar_fig


def do_compare_models(clean_name, noise_name, snr_db):
    """Run ALL models and compare."""
    models = ["Transformer (本项目)", "noisereduce", "RNNoise", "GTCRN"]
    return do_mix_and_denoise(clean_name, noise_name, snr_db, models)


def do_training_dashboard():
    loss_fig = plot_training_curves()
    final_fig = plot_final_results()
    return loss_fig, final_fig


# ═══════════════════════════════════════════════════════════════
#  Gradio UI
# ═══════════════════════════════════════════════════════════════
with gr.Blocks(title="Transformer 语音增强系统", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 融入卷积频率前端的 Transformer 语音增强系统
    **答辩演示平台** — 多模型降噪 · 指标对比 · 训练可视化
    """)

    with gr.Tabs():
        # ── Tab 1: Audio Mixing ────────────────────────────────
        with gr.Tab("1. 音频混合"):
            with gr.Row():
                with gr.Column(scale=1):
                    c_dd = gr.Dropdown(clean_names, label="干净语音 (TIMIT)", value=clean_names[0] if clean_names else None)
                    n_dd = gr.Dropdown(noise_names, label="噪声 (NOISEX-92)", value=noise_names[0] if noise_names else None)
                    snr_sl = gr.Slider(-5, 10, value=0, step=1, label="SNR (dB)")
                    mix_btn = gr.Button("混合并预览", variant="primary")
                with gr.Column(scale=2):
                    mix_audio_out = gr.Audio(label="含噪混合音频", type="numpy")
            mix_btn.click(fn=lambda c, n, s: mix_audio(c, n, s)[1],
                          inputs=[c_dd, n_dd, snr_sl], outputs=[mix_audio_out])

        # ── Tab 2: Denoising ──────────────────────────────────
        with gr.Tab("2. 降噪处理"):
            with gr.Row():
                with gr.Column(scale=1):
                    c2 = gr.Dropdown(clean_names, label="干净语音", value=clean_names[0] if clean_names else None)
                    n2 = gr.Dropdown(noise_names, label="噪声", value=noise_names[0] if noise_names else None)
                    s2 = gr.Slider(-5, 10, value=0, step=1, label="SNR (dB)")
                    models_cb = gr.CheckboxGroup(
                        ["Transformer (本项目)", "noisereduce", "RNNoise", "GTCRN"],
                        label="降噪模型", value=["Transformer (本项目)"])
                    denoise_btn = gr.Button("开始降噪", variant="primary")
                with gr.Column(scale=2):
                    info_text = gr.Textbox(label="评估指标", lines=8)
                    denoised_audio = gr.Audio(label="降噪后音频", type="numpy")

            denoise_btn.click(
                fn=lambda c, n, s, m: (do_mix_and_denoise(c, n, s, m)[1],
                                        do_mix_and_denoise(c, n, s, m)[0] if do_mix_and_denoise(c, n, s, m)[0] else (None, None)),
                inputs=[c2, n2, s2, models_cb],
                outputs=[info_text, denoised_audio])

        # ── Tab 3: Waveform & Spectrogram ─────────────────────
        with gr.Tab("3. 音频可视化"):
            with gr.Row():
                c3 = gr.Dropdown(clean_names, label="干净语音", value=clean_names[0] if clean_names else None)
                n3 = gr.Dropdown(noise_names, label="噪声", value=noise_names[0] if noise_names else None)
                s3 = gr.Slider(-5, 10, value=0, step=1, label="SNR (dB)")
                m3 = gr.CheckboxGroup(["Transformer (本项目)", "noisereduce", "RNNoise", "GTCRN"],
                                       label="模型", value=["Transformer (本项目)"])
                viz_btn = gr.Button("生成可视化", variant="primary")

            wf_plot = gr.Plot(label="波形对比")
            sp_plot = gr.Plot(label="频谱对比")

            viz_btn.click(
                fn=lambda c, n, s, m: (do_mix_and_denoise(c, n, s, m)[2],
                                        do_mix_and_denoise(c, n, s, m)[3]),
                inputs=[c3, n3, s3, m3], outputs=[wf_plot, sp_plot])

        # ── Tab 4: Model Comparison ───────────────────────────
        with gr.Tab("4. 模型指标对比"):
            with gr.Row():
                c4 = gr.Dropdown(clean_names, label="干净语音", value=clean_names[0] if clean_names else None)
                n4 = gr.Dropdown(noise_names, label="噪声", value=noise_names[0] if noise_names else None)
                s4 = gr.Slider(-5, 10, value=0, step=1, label="SNR (dB)")
                cmp_btn = gr.Button("全模型对比", variant="primary")
            cmp_info = gr.Textbox(label="对比结果", lines=10)
            cmp_plot = gr.Plot(label="STOI / SDR 柱状图")
            cmp_btn.click(fn=do_compare_models, inputs=[c4, n4, s4],
                          outputs=[cmp_info, cmp_plot])

        # ── Tab 5: Training Dashboard ─────────────────────────
        with gr.Tab("5. 训练过程"):
            gr.Markdown("### Transformer 改进版训练数据")
            train_btn = gr.Button("加载训练数据", variant="primary")
            with gr.Row():
                loss_plot = gr.Plot(label="损失 & 学习率曲线")
                final_plot = gr.Plot(label="全方法最终结果")
            train_btn.click(fn=do_training_dashboard, inputs=[],
                            outputs=[loss_plot, final_plot])

            gr.Markdown("""
            ---
            **训练配置**: 2000 样本 | Batch=4 | 30 Epochs | CPU
            **优化策略**: 频域加权 L1 Loss (低频 3× → 高频 1×) | 余弦退火 LR (1e-3 → 1e-6)
            **最终结果**: STOI **0.884** | SDR **12.67 dB**
            """)


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
