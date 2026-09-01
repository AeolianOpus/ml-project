"""
Helix end-to-end test using WASAPI device 14.
Records 5s, runs through trained model, reports predictions.
"""
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Config ---
HELIX_DEVICE = 6      # WASAPI, 1 channel — cleanest
HELIX_SR = 48000
DURATION_SEC = 10
CHANNELS = 2

SR = 22050
WINDOW_SAMPLES = 11025
N_BINS = 84
BINS_PER_OCTAVE = 12
CQT_HOP_LENGTH = 512

PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
CLASS_NAMES = (
    [f"note:{p}" for p in PITCH_CLASSES]
    + [f"maj:{p}" for p in PITCH_CLASSES]
    + [f"min:{p}" for p in PITCH_CLASSES]
    + ["silence"]
)
NUM_CLASSES = len(CLASS_NAMES)
DEVICE = torch.device("cpu")


def find_project_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / ".gitignore").exists() and (parent / "data").exists():
            return parent
    raise RuntimeError(f"Could not find project root from {start}")

PROJECT_ROOT = find_project_root(Path(__file__).resolve().parent)
MODEL_PATH = PROJECT_ROOT / "models" / "cnn_augmented_best.pt"


class ChordCNN(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 1))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.global_pool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)


def audio_to_cqt(audio: np.ndarray) -> np.ndarray:
    C = np.abs(librosa.cqt(
        audio, sr=SR, hop_length=CQT_HOP_LENGTH,
        n_bins=N_BINS, bins_per_octave=BINS_PER_OCTAVE,
    ))
    return librosa.amplitude_to_db(C, ref=np.max).astype(np.float32)


def predict_window(model: ChordCNN, audio_window: np.ndarray) -> np.ndarray:
    cqt = audio_to_cqt(audio_window)
    cqt_normalized = (cqt + 80.0) / 80.0
    x = torch.from_numpy(cqt_normalized).unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    return probs


# --- Load model ---
print(f"Loading model...")
model = ChordCNN(num_classes=NUM_CLASSES).to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print(f"Model loaded (val_acc {checkpoint['val_acc']:.3f})\n")

# --- Record ---
print(f"Recording {DURATION_SEC}s from Helix (WASAPI device {HELIX_DEVICE})...")
print("Play something SIMPLE and HOLD IT — e.g., open E, C major, A minor. GO!\n")
time.sleep(0.5)

recording = sd.rec(
    int(DURATION_SEC * HELIX_SR),
    samplerate=HELIX_SR,
    channels=CHANNELS,
    device=HELIX_DEVICE,
    dtype="float32",
)
sd.wait()

audio = recording.mean(axis=1)
max_amp = float(np.abs(audio).max())
rms = float(np.sqrt(np.mean(audio ** 2)))
print(f"Signal: max={max_amp:.4f}, rms={rms:.4f}")

if max_amp < 0.01:
    print("⚠ Signal too weak — did you play? Play harder next time.\n")
    exit(1)

# --- Resample and infer ---
audio_resampled = librosa.resample(audio, orig_sr=HELIX_SR, target_sr=SR)
n_windows = len(audio_resampled) // WINDOW_SAMPLES
print(f"\nRunning inference on {n_windows} × 500ms windows...\n")

all_top = []
for i in range(n_windows):
    start = i * WINDOW_SAMPLES
    window = audio_resampled[start:start + WINDOW_SAMPLES]
    probs = predict_window(model, window)
    top_idx = int(np.argmax(probs))
    top_conf = float(probs[top_idx])
    top3 = np.argsort(probs)[::-1][:3]
    all_top.append((CLASS_NAMES[top_idx], top_conf, [(CLASS_NAMES[j], probs[j]) for j in top3]))

print(f"{'Window':>8} {'Time':>7}  {'Top':>14} {'Conf':>8}  {'Runner-up':>14} {'Third':>14}")
print("-" * 80)
for i, (label, conf, top3) in enumerate(all_top):
    t = i * 0.5
    marker = " ✓" if conf > 0.5 else " ~" if conf > 0.3 else "  "
    r_label, r_conf = top3[1]
    t_label, t_conf = top3[2]
    print(f"{i+1:>8} {t:>5.1f}s   {label:>14} {conf:>7.1%}{marker}  "
          f"{r_label:>14} {r_conf:>7.1%}  {t_label:>14} {t_conf:>7.1%}")

# Summary
from collections import Counter
counts = Counter(t[0] for t in all_top)
most_common, count = counts.most_common(1)[0]
avg_conf = np.mean([t[1] for t in all_top])
print(f"\n{'='*80}")
print(f"Most-predicted: {most_common} ({count}/{n_windows} windows)  |  Avg confidence: {avg_conf:.1%}")
print(f"{'='*80}")