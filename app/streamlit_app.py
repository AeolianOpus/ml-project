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
import queue
import time
import sounddevice as sd
import warnings
warnings.filterwarnings("ignore", message="n_fft=.* is too large")

# ==== CONFIG ====
# Live audio config (Helix via DirectSound Primary Capture)
LIVE_DEVICE = 6              # confirmed working — stereo, 0.43 max amplitude
LIVE_SR_NATIVE = 48000
LIVE_CHANNELS = 2            # stereo, averaged to mono internally
LIVE_BLOCK_MS = 100


SR = 22050
WINDOW_SEC = 0.5
WINDOW_SAMPLES = int(SR * WINDOW_SEC)
HOP_SEC = 0.25
HOP_SAMPLES = int(SR * HOP_SEC)

N_BINS = 84
BINS_PER_OCTAVE = 12
CQT_HOP_LENGTH = 512

PITCH_CLASSES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
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


def _start_live_audio() -> None:
    """Start the audio input stream. Stored in session_state to survive reruns."""
    if st.session_state.get("live_stream") is not None:
        return

    audio_queue: queue.Queue = queue.Queue()
    audio_buffer = np.zeros(int(LIVE_SR_NATIVE * WINDOW_SEC), dtype=np.float32)

    def callback(indata, frames, time_info, status):
        audio_queue.put(indata.mean(axis=1).copy())

    block_size = int(LIVE_SR_NATIVE * LIVE_BLOCK_MS / 1000)
    stream = sd.InputStream(
        device=LIVE_DEVICE,
        samplerate=LIVE_SR_NATIVE,
        channels=LIVE_CHANNELS,
        blocksize=block_size,
        dtype="float32",
        callback=callback,
    )
    stream.start()

    st.session_state.live_stream = stream
    st.session_state.live_queue = audio_queue
    st.session_state.live_buffer = audio_buffer


def _stop_live_audio() -> None:
    stream = st.session_state.get("live_stream")
    if stream is not None:
        stream.stop()
        stream.close()
    st.session_state.live_stream = None
    st.session_state.live_queue = None
    st.session_state.live_buffer = None


def render_live_mode(cnn_model) -> None:
    """Live capture from Helix — continuous inference, no file needed."""
    _start_live_audio()

    top1, top2 = st.columns([4, 1])
    with top1:
        st.markdown("## 🔴 LIVE — reading from Helix")
    with top2:
        if st.button("■  STOP", use_container_width=True):
            _stop_live_audio()
            st.session_state.live_mode = False
            st.rerun()

    st.caption(f"Device {LIVE_DEVICE} · {LIVE_SR_NATIVE} Hz · stereo→mono · updates every 500 ms")

    # Init prediction history
    if "live_history" not in st.session_state:
        st.session_state.live_history = []

    # Fragment: only this rerurns, not the whole page → no more gray-out
    _live_inference_fragment(cnn_model)


@st.fragment(run_every=0.5)
def _live_inference_fragment(cnn_model) -> None:
    """Runs every 500ms without triggering a full page rerun."""
    # Drain audio queue into rolling buffer
    if st.session_state.live_queue is not None and st.session_state.live_buffer is not None:
        while not st.session_state.live_queue.empty():
            try:
                chunk = st.session_state.live_queue.get_nowait()
                n = len(chunk)
                buf = st.session_state.live_buffer
                buf = np.roll(buf, -n)
                buf[-n:] = chunk
                st.session_state.live_buffer = buf
            except queue.Empty:
                break

    if st.session_state.live_buffer is not None:
        resampled = librosa.resample(st.session_state.live_buffer, orig_sr=LIVE_SR_NATIVE, target_sr=SR)
        if len(resampled) >= WINDOW_SAMPLES:
            window = resampled[-WINDOW_SAMPLES:]
            signal_amp = float(np.abs(window).max())
            probs = predict_window(cnn_model, window)
            top_idx = int(np.argmax(probs))
            top_conf = float(probs[top_idx])
            top_label = CLASS_NAMES[top_idx]

            st.session_state.live_history.append((top_label, top_conf, signal_amp))
            if len(st.session_state.live_history) > 30:
                st.session_state.live_history = st.session_state.live_history[-30:]

            display = top_label if signal_amp > 0.01 else "— quiet —"
            color = "#4caf50" if top_conf > 0.5 else "#ff9800" if top_conf > 0.3 else "#666"
            st.markdown(
                f"<div style='text-align:center; padding:40px; background:#0f0f0f; "
                f"border:2px solid #333; border-radius:16px; margin:20px 0;'>"
                f"<h1 style='color:{color}; font-size:120px; margin:0; font-family:Georgia,serif;'>"
                f"{display}</h1>"
                f"<p style='color:#888; font-size:14px; margin-top:12px;'>"
                f"signal {signal_amp:.3f} · confidence {top_conf:.1%}</p></div>",
                unsafe_allow_html=True,
            )
            st.progress(min(top_conf, 1.0))

            # Fun message when confidence is low
            recent = st.session_state.live_history[-4:]
            if len(recent) >= 4:
                avg_conf = np.mean([conf for _, conf, _ in recent])
                avg_amp = np.mean([amp for _, _, amp in recent])
                if avg_amp > 0.05 and avg_conf < 0.35:
                    st.markdown(
                        "<div style='text-align:center; font-size:22px; color:#ff9800; "
                        "font-family:Georgia,serif; font-style:italic; margin-top:20px;'>"
                        "🔥 slow down tiger! 🔥</div>",
                        unsafe_allow_html=True,
                    )

# ==== KEY DETECTION ====

# Weight vectors — how much each pitch class is emphasized in a key
# Krumhansl-Schmuckler-inspired: tonic gets highest weight, then perfect 5th, 3rd, etc.
MAJOR_KEY_PROFILE = np.array([
    6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
    2.52, 5.19, 2.39, 3.66, 2.29, 2.88,
])
MINOR_KEY_PROFILE = np.array([
    6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
    2.54, 4.75, 3.98, 2.69, 3.34, 3.17,
])


def detect_key(pitch_class_energy: np.ndarray) -> tuple[str, str, float]:
    """
    Given a 12-dim pitch-class energy vector, return (root, mode, confidence).
    mode is "major" or "minor". confidence is the correlation with the winning key profile.
    """
    best_root = 0
    best_mode = "major"
    best_score = -np.inf

    # Try all 12 major keys and 12 minor keys — rotate the profile
    for root in range(12):
        for mode, profile in [("major", MAJOR_KEY_PROFILE), ("minor", MINOR_KEY_PROFILE)]:
            rotated = np.roll(profile, root)
            # Correlation between rotated profile and observed pitch energy
            score = float(np.corrcoef(rotated, pitch_class_energy)[0, 1])
            if score > best_score:
                best_score = score
                best_root = root
                best_mode = mode

    return PITCH_CLASSES[best_root], best_mode, best_score


def class_predictions_to_pitch_energy(
    predictions: np.ndarray,
    confidences: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """
    Convert per-window class predictions into a 12-dim pitch-class energy vector.
    Each high-confidence prediction contributes to its constituent pitch classes,
    weighted by its confidence.
    """
    energy = np.zeros(12, dtype=np.float32)
    for cid, conf in zip(predictions, confidences):
        if conf < threshold:
            continue
        cid = int(cid)
        if cid == 36:  # silence — contributes nothing
            continue
        if cid < 12:  # single note — full contribution to its pitch class
            energy[cid] += conf
        elif cid < 24:  # major triad — root + major 3rd + perfect 5th
            root = cid - 12
            energy[root] += conf
            energy[(root + 4) % 12] += conf * 0.6
            energy[(root + 7) % 12] += conf * 0.8
        else:  # minor triad — root + minor 3rd + perfect 5th
            root = cid - 24
            energy[root] += conf
            energy[(root + 3) % 12] += conf * 0.6
            energy[(root + 7) % 12] += conf * 0.8

    # Normalize so absolute confidence doesn't matter, just the profile shape
    if energy.sum() > 0:
        energy = energy / energy.sum()
    return energy

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

# Init session state for live mode
if "live_mode" not in st.session_state:
    st.session_state.live_mode = False

# Big red LIVE button — CSS
st.markdown("""
<style>
div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #ff1744 0%, #b71c1c 100%);
    color: white !important;
    font-size: 28px !important;
    font-weight: 900 !important;
    padding: 28px 40px !important;
    border-radius: 16px !important;
    border: 3px solid #ff8080 !important;
    box-shadow: 0 0 30px rgba(255,23,68,0.6), 0 4px 20px rgba(0,0,0,0.3),
                inset 0 -4px 12px rgba(0,0,0,0.2) !important;
    letter-spacing: 4px;
    text-transform: uppercase;
    animation: pulse-live 1.8s ease-in-out infinite;
    transition: all 0.2s ease;
}
div.stButton > button[kind="primary"]:hover {
    transform: translateY(-3px) scale(1.02);
    box-shadow: 0 0 50px rgba(255,23,68,0.95), 0 8px 30px rgba(0,0,0,0.4),
                inset 0 -4px 12px rgba(0,0,0,0.2) !important;
}
@keyframes pulse-live {
    0%, 100% { box-shadow: 0 0 30px rgba(255,23,68,0.6), 0 4px 20px rgba(0,0,0,0.3),
               inset 0 -4px 12px rgba(0,0,0,0.2); }
    50%      { box-shadow: 0 0 50px rgba(255,23,68,1.0), 0 4px 20px rgba(0,0,0,0.3),
               inset 0 -4px 12px rgba(0,0,0,0.2); }
}
</style>
""", unsafe_allow_html=True)

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
    
# Route into live mode if active — hides all file-mode UI
if st.session_state.live_mode:
    render_live_mode(model)
    st.stop()

# GO LIVE button — shown when in file mode
col_left, col_center, col_right = st.columns([1, 2, 1])
with col_center:
    if st.button("⚡  GO LIVE FROM HELIX  ⚡", type="primary", use_container_width=True):
        st.session_state.live_mode = True
        st.rerun()

st.markdown("---")


import io

# --- Sample files for quick demo ---
DEMO_SAMPLES = {
    "🎸 Chord playing (Eb Bossa Nova)": "00_BN1-129-Eb_comp_mic.wav",
    "🎵 Solo melody (C# Rock)": "01_Rock1-90-C#_solo_mic.wav",
    "🎼 Chord playing (C Jazz)": None,  # will find dynamically below
}

# Try to locate a jazz comp file automatically
AUDIO_SAMPLES_DIR = PROJECT_ROOT / "data" / "guitarset" / "audio_mono-mic"
jazz_candidates = list(AUDIO_SAMPLES_DIR.glob("*_Jazz*_comp_mic.wav"))
if jazz_candidates:
    DEMO_SAMPLES["🎼 Chord playing (Jazz)"] = jazz_candidates[0].name
    DEMO_SAMPLES.pop("🎼 Chord playing (C Jazz)")
else:
    DEMO_SAMPLES.pop("🎼 Chord playing (C Jazz)")


st.markdown("### Choose an audio source")

tab_upload, tab_demo = st.tabs(["📁 Upload your own", "⚡ Try a demo file"])

audio_bytes: bytes | None = None
source_name: str = ""

with tab_upload:
    uploaded = st.file_uploader(
        "Drop a .wav / .mp3 / .flac / .ogg file",
        type=["wav", "mp3", "flac", "ogg"],
    )
    if uploaded is not None:
        audio_bytes = uploaded.read()
        source_name = uploaded.name

with tab_demo:
    st.caption("Instantly load a file from GuitarSet — no upload needed.")
    cols = st.columns(len(DEMO_SAMPLES))
    for col, (label, filename) in zip(cols, DEMO_SAMPLES.items()):
        with col:
            if st.button(label, use_container_width=True):
                sample_path = AUDIO_SAMPLES_DIR / filename
                if sample_path.exists():
                    audio_bytes = sample_path.read_bytes()
                    source_name = filename
                    st.success(f"Loaded {filename}")
                else:
                    st.error(f"Sample file not found: {sample_path}")


if audio_bytes is None:
    st.info("👆 Upload a file or click a demo button to get started.")
    st.stop()

# Load audio
with st.spinner(f"Loading and processing {source_name}..."):
    y, _ = librosa.load(io.BytesIO(audio_bytes), sr=SR, mono=True)

duration = len(y) / SR
st.markdown(f"### Now analyzing: `{source_name}`")
st.audio(audio_bytes)
st.caption(
    f"Duration: **{duration:.2f} s**  •  "
    f"Samples: **{len(y):,}**  •  "
    f"Sample rate: **{SR} Hz**"
)

# Run predictions across all windows
with st.spinner("Running CNN inference..."):
    all_probs = predict_all_windows(model, y)

times = np.arange(len(all_probs)) * HOP_SEC

# --- Mode selector ---
mode = st.radio(
    "Analysis mode",
    ["Segment-by-segment — best for solos and progressions", "Whole file (average) — best for held chords"],
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

    # Split into two columns — left: stats, right: key detection
    stat_col, key_col = st.columns([1, 1])

    with stat_col:
        st.metric("High-confidence windows", f"{n_high_conf} / {len(all_probs)}")
        st.metric("Distinct classes detected", f"{len(class_counter)}")

    with key_col:
        # Derive the likely key from all high-confidence predictions
        pitch_energy = class_predictions_to_pitch_energy(
            predictions_per_window, confidences_per_window, threshold
        )

        if pitch_energy.sum() > 0:
            key_root, key_mode, key_conf = detect_key(pitch_energy)
            key_symbol = "🎼"
            st.metric(
                f"{key_symbol}  Likely key",
                f"{key_root} {key_mode}",
                delta=f"correlation {key_conf:+.2f}",
                delta_color="off",
            )
            st.caption(
                "Derived from high-confidence predictions using Krumhansl-style key profiles. "
                "This is *not* a model output — it's downstream analysis."
            )
        else:
            st.metric("🎼 Likely key", "—")
            st.caption("Not enough high-confidence data to detect a key.")

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