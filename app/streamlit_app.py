"""
Streamlit demo app — Guitar Note & Chord Recognition
Run with: streamlit run app/streamlit_app.py
"""
from pathlib import Path

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==== CONFIG ====

SR = 22050
WINDOW_SEC = 0.5
WINDOW_SAMPLES = int(SR * WINDOW_SEC)
HOP_SEC = 0.25
HOP_SAMPLES = int(SR * HOP_SEC)

N_BINS = 84
BINS_PER_OCTAVE = 12
CQT_HOP_LENGTH = 512

PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
CLASS_NAMES: list[str] = (
    [f"note:{p}" for p in PITCH_CLASSES]
    + [f"maj:{p}" for p in PITCH_CLASSES]
    + [f"min:{p}" for p in PITCH_CLASSES]
    + ["silence"]
)
NUM_CLASSES = len(CLASS_NAMES)

# Robust project root — search upward from this file's location
def find_project_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / ".gitignore").exists() and (parent / "data").exists():
            return parent
    raise RuntimeError(f"Could not find project root from {start}")

PROJECT_ROOT = find_project_root(Path(__file__).resolve().parent)
MODEL_PATH = PROJECT_ROOT / "models" / "cnn_augmented_best.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==== MODEL ====

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


@st.cache_resource
def load_model() -> ChordCNN:
    """Load model once and cache — Streamlit reruns the script on every interaction."""
    model = ChordCNN(num_classes=NUM_CLASSES).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


# ==== AUDIO PROCESSING ====

def audio_to_cqt(audio: np.ndarray) -> np.ndarray:
    """500ms audio window → log-CQT spectrogram (84 x ~22)."""
    C = np.abs(librosa.cqt(
        audio, sr=SR, hop_length=CQT_HOP_LENGTH,
        n_bins=N_BINS, bins_per_octave=BINS_PER_OCTAVE,
    ))
    return librosa.amplitude_to_db(C, ref=np.max).astype(np.float32)


def predict_window(model: ChordCNN, audio_window: np.ndarray) -> np.ndarray:
    """One 500ms window → 37-dim softmax probabilities."""
    cqt = audio_to_cqt(audio_window)
    cqt_normalized = (cqt + 80.0) / 80.0
    x = torch.from_numpy(cqt_normalized).unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    return probs


def predict_all_windows(model: ChordCNN, audio: np.ndarray) -> np.ndarray:
    """Slide 500ms/250ms across audio → (num_windows, 37) probability matrix."""
    if len(audio) < WINDOW_SAMPLES:
        # Pad if audio shorter than one window
        audio = np.pad(audio, (0, WINDOW_SAMPLES - len(audio)))
    all_probs: list[np.ndarray] = []
    pos = 0
    while pos + WINDOW_SAMPLES <= len(audio):
        window = audio[pos : pos + WINDOW_SAMPLES]
        all_probs.append(predict_window(model, window))
        pos += HOP_SAMPLES
    return np.stack(all_probs)


# ==== UI ====

st.set_page_config(
    page_title="Guitar Chord & Note Recognizer",
    page_icon="🎸",
    layout="wide",
)

st.title("🎸 Guitar Chord & Note Recognition")
st.markdown(
    "**Large Individual Project — Jensen YH.** "
    "CNN trained on GuitarSet with pitch-shift augmentation. "
    "37 classes: 12 single notes + 12 major triads + 12 minor triads + silence."
)

# --- Sidebar with info + model status ---
model = load_model()
with st.sidebar:
    st.header("Model")
    st.write(f"**Device:** `{DEVICE}`")
    st.write(f"**Classes:** {NUM_CLASSES}")
    st.write(f"**Test accuracy:** 0.633")
    st.write(f"**Test macro F1:** 0.649")
    st.write(f"**Test ROC-AUC:** 0.946")
    st.markdown("---")
    st.header("Instructions")
    st.write(
        "1. Upload a guitar audio file\n"
        "2. See the predicted chord/note over time\n"
        "3. Best results: mono acoustic-style guitar,\n"
        "   clean tone, isolated chord or note"
    )


# --- File upload + processing ---
uploaded = st.file_uploader(
    "Upload a guitar audio file",
    type=["wav", "mp3", "flac", "ogg"],
)

if uploaded is None:
    st.info("👆 Upload an audio file to get started.")
    st.stop()

# Load audio
with st.spinner("Loading and processing audio..."):
    audio_bytes = uploaded.read()
    # librosa.load supports file-like objects
    import io
    y, _ = librosa.load(io.BytesIO(audio_bytes), sr=SR, mono=True)

duration = len(y) / SR
st.audio(audio_bytes)
st.write(
    f"**Duration:** {duration:.2f} s  |  "
    f"**Samples:** {len(y):,}  |  "
    f"**Sample rate:** {SR} Hz"
)

# Run predictions across all windows
with st.spinner("Running CNN inference..."):
    all_probs = predict_all_windows(model, y)

times = np.arange(len(all_probs)) * HOP_SEC

# --- Mode selector ---
mode = st.radio(
    "Analysis mode",
    ["Whole file (average) — best for held chords", "Segment-by-segment — best for solos and progressions"],
    horizontal=True,
)

if mode.startswith("Whole file"):
    # ---- AVERAGE MODE ----
    aggregate = all_probs.mean(axis=0)
    top_indices = np.argsort(aggregate)[::-1][:5]

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Top-5 predictions (averaged over entire file)")
        fig, ax = plt.subplots(figsize=(7, 5))
        top_names = [CLASS_NAMES[i] for i in top_indices][::-1]
        top_probs = aggregate[top_indices][::-1]
        bars = ax.barh(top_names, top_probs, color="#4caf50")
        bars[-1].set_color("#e91e63")
        ax.set_xlabel("Confidence")
        ax.set_xlim(0, max(1.0, top_probs.max() * 1.15))
        for bar, prob in zip(bars, top_probs):
            ax.text(prob + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{prob:.1%}", va="center", fontsize=10)
        ax.grid(axis="x", alpha=0.3)
        ax.set_title(f"Best guess: {CLASS_NAMES[top_indices[0]]}")
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.subheader("Top-5 class confidences over time")
        fig, ax = plt.subplots(figsize=(7, 5))
        for idx in top_indices:
            ax.plot(times, all_probs[:, idx], label=CLASS_NAMES[idx], linewidth=2)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Confidence")
        ax.set_ylim(0, 1)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)

else:
    # ---- SEGMENT-BY-SEGMENT MODE ----
    # For each window, show the argmax + its confidence
    predictions_per_window = all_probs.argmax(axis=1)
    confidences_per_window = all_probs.max(axis=1)

    # Confidence threshold — below this we call it "unsure"
    threshold = st.slider("Minimum confidence to display", 0.0, 0.9, 0.30, 0.05)

    st.subheader("What the model heard, moment by moment")
    fig, ax = plt.subplots(figsize=(14, 5))

    # Color code: notes=blue, majors=green, minors=orange, silence=gray, low-conf=lightgray
    for i, (t, cid, conf) in enumerate(zip(times, predictions_per_window, confidences_per_window)):
        if conf < threshold:
            color = "#e0e0e0"
        elif cid < 12:
            color = "#2196f3"  # notes
        elif cid < 24:
            color = "#4caf50"  # major
        elif cid < 36:
            color = "#ff9800"  # minor
        else:
            color = "#9e9e9e"  # silence
        ax.bar(t, conf, width=HOP_SEC, color=color, edgecolor="none", align="edge")

    # Annotate high-confidence predictions
    last_labeled_class = -1
    for t, cid, conf in zip(times, predictions_per_window, confidences_per_window):
        if conf >= threshold and cid != last_labeled_class:
            ax.text(t + HOP_SEC / 2, conf + 0.02, CLASS_NAMES[cid],
                    fontsize=8, ha="center", rotation=45, color="#333")
            last_labeled_class = cid

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Confidence")
    ax.set_ylim(0, 1.15)
    ax.set_xlim(0, times[-1] + HOP_SEC)
    ax.set_title(
        f"Per-window predictions  "
        f"(blue = single note, green = major, orange = minor, gray = silence / unsure)"
    )
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

    # Also show a summary count
    from collections import Counter
    high_conf_mask = confidences_per_window >= threshold
    high_conf_classes = predictions_per_window[high_conf_mask]
    class_counter = Counter(int(c) for c in high_conf_classes)
    n_high_conf = int(high_conf_mask.sum())

    st.write(
        f"**{n_high_conf}/{len(all_probs)} windows** above confidence threshold "
        f"({threshold:.0%}). "
        f"**{len(class_counter)}** distinct classes detected."
    )

    if class_counter:
        counts_df = [
            {"Class": CLASS_NAMES[cid],
             "Windows": count,
             "Fraction": f"{count / n_high_conf:.1%}"}
            for cid, count in class_counter.most_common(10)
        ]
        st.dataframe(counts_df, hide_index=True)

# --- Spectrogram at the bottom (full width) ---
st.subheader("CQT spectrogram of your audio")
fig, ax = plt.subplots(figsize=(14, 4))
C = np.abs(librosa.cqt(y, sr=SR, n_bins=N_BINS, bins_per_octave=BINS_PER_OCTAVE))
C_db = librosa.amplitude_to_db(C, ref=np.max)
img = librosa.display.specshow(
    C_db, sr=SR, x_axis="time", y_axis="cqt_note", ax=ax, cmap="magma",
)
fig.colorbar(img, ax=ax, format="%+2.0f dB")
ax.set_title("CQT — this is what the model sees")
plt.tight_layout()
st.pyplot(fig)