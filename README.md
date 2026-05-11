# Audio Denoise Graduation Project

This workspace contains a compact graduation-design baseline for:

- running RNNoise on demo audio
- generating noisy/clean training pairs
- training a minimal Transformer spectral-mask denoiser in PyTorch
- extracting STFT/log-mel/RNNoise-like band features
- batch-evaluating PESQ/STOI/SNR

## Structure

- `scripts/mix_audio.py`: mix clean speech and noise at target SNR values
- `scripts/split_dataset.py`: split paired wav files into `train/val/test` folders
- `scripts/run_rnnoise_demo.py`: denoise wav files with `pyrnnoise`
- `scripts/extract_features.py`: extract STFT, log-mel, and RNNoise-like band features
- `scripts/evaluate_metrics.py`: batch-evaluate PESQ, STOI, and SNR
- `scripts/plot_results.py`: plot loss curves and metric bar charts
- `models/transformer_denoiser.py`: minimal Transformer mask model
- `scripts/train_transformer.py`: train on paired noisy/clean wav files
- `scripts/infer_transformer.py`: run model inference on a noisy wav file

## Quick Start

1. Install dependencies:

```powershell
python -m pip install numpy soundfile librosa pyrnnoise torch
```

2. Generate mixed audio:

```powershell
python scripts/mix_audio.py --clean_dir data/clean --noise_dir data/noise --output_dir data/mixed --snrs -5 0 5 10
```

3. Run RNNoise on the mixed file:

```powershell
python scripts/run_rnnoise_demo.py --input_glob "data/mixed/*.wav" --output_dir outputs/rnnoise
```

4. Train the minimal Transformer:

```powershell
python scripts/train_transformer.py --data_dir data/mixed --epochs 5 --batch_size 2 --feature_type stft_log_mag
```

Optional: split the generated pairs into `train/val/test` folders:

```powershell
python scripts/split_dataset.py --source_dir data/mixed --output_root data/splits
```

5. Inference:

```powershell
python scripts/infer_transformer.py --checkpoint checkpoints/transformer_best.pt --input_wav data/mixed/demo_clean_snr0.wav --output_wav outputs/transformer/demo_clean_snr0_denoised.wav
```

6. Extract features:

```powershell
python scripts/extract_features.py --input_glob "data/mixed/*.wav" --output_dir outputs/features
```

7. Evaluate RNNoise output:

```powershell
python -m pip install pesq pystoi
python scripts/evaluate_metrics.py --metadata_csv data/mixed/metadata.csv --enhanced_dir outputs/rnnoise --output_csv outputs/metrics/rnnoise_metrics.csv --pair_mode rnnoise
```

8. Plot training and evaluation figures:

```powershell
python scripts/plot_results.py --history_csvs outputs/metrics/transformer_loss_history.csv --metric_csvs outputs/metrics/rnnoise_metrics.csv --output_dir outputs/plots
```

## Data Naming

`mix_audio.py` saves paired files in this format:

- noisy: `name_snr{snr}.wav`
- clean reference: `name_clean.wav`
- metadata: `metadata.csv`

## Notes

- The default sample rate is `16000 Hz`.
- The Transformer baseline predicts a magnitude mask and reuses the noisy phase for ISTFT reconstruction.
- `train_transformer.py` supports `stft_log_mag`, `log_mel`, and `rnnoise_like` input features.
- The simplified RNNoise-style features use Bark-like band energies and frame-to-frame deltas.
- RNNoise packages usually expect `48000 Hz`; the demo script resamples automatically.
