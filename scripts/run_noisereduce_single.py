"""Single-file noisereduce denoising."""
import argparse
import numpy as np
import soundfile as sf
import noisereduce as nr

parser = argparse.ArgumentParser()
parser.add_argument("--input_wav", required=True)
parser.add_argument("--output_wav", required=True)
args = parser.parse_args()

audio, sr = sf.read(args.input_wav)
audio = np.asarray(audio, dtype=np.float32)
if audio.ndim > 1:
    audio = np.mean(audio, axis=1)

enhanced = nr.reduce_noise(y=audio, sr=sr, stationary=False,
                           prop_decrease=0.8, n_fft=512, freq_mask_smooth_hz=250)
sf.write(args.output_wav, enhanced, sr)
print(f"saved: {args.output_wav}")
