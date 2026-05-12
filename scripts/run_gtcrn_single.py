"""Single-file GTCRN ONNX denoising."""
import argparse
import numpy as np
import soundfile as sf
import onnxruntime as ort
from librosa import istft

parser = argparse.ArgumentParser()
parser.add_argument("--input_wav", required=True)
parser.add_argument("--output_wav", required=True)
args = parser.parse_args()

mix, sr = sf.read(args.input_wav, dtype="float32")
if mix.ndim > 1:
    mix = np.mean(mix, axis=1)

# STFT: n_fft=512, hop=256, win=512
import torch
x = torch.stft(torch.from_numpy(mix), 512, 256, 512,
               torch.hann_window(512).pow(0.5), return_complex=False)[None]
inputs = x.numpy()  # (1, 2, F, T)

session = ort.InferenceSession("checkpoints/gtcrn.onnx",
                                providers=["CPUExecutionProvider"])

conv_cache = np.zeros([2, 1, 16, 16, 33], dtype="float32")
tra_cache = np.zeros([2, 3, 1, 1, 16], dtype="float32")
inter_cache = np.zeros([2, 1, 33, 16], dtype="float32")

outputs = []
for i in range(inputs.shape[-2]):
    out_i, conv_cache, tra_cache, inter_cache = session.run(
        [],
        {"mix": inputs[..., i:i + 1, :],
         "conv_cache": conv_cache,
         "tra_cache": tra_cache,
         "inter_cache": inter_cache})
    outputs.append(out_i)

outputs = np.concatenate(outputs, axis=2)
enhanced = istft(outputs[..., 0] + 1j * outputs[..., 1],
                 n_fft=512, hop_length=256, win_length=512,
                 window=np.hanning(512) ** 0.5)
enhanced = enhanced.squeeze()[:len(mix)]

sf.write(args.output_wav, enhanced.astype(np.float32), sr)
print(f"saved: {args.output_wav}")
