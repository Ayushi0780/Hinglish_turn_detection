
"""
Gradio demo for real-time audio turn detection.

Two model tiers:

1. TinyCNNGRU
   - Trained from scratch
   - Log-mel spectrogram input
   - Lightweight / efficient

2. WhisperTurnClassifier
   - Whisper-tiny pretrained encoder
   - Classification head
   - More accurate / multilingual

The demo shows both predictions side-by-side.

Run:
    python src/demo_app.py
"""

import time

import gradio as gr
import librosa
import numpy as np
import torch

from model import WhisperTurnClassifier, TinyCNNGRU


# ============================================================
# SETTINGS
# ============================================================

SAMPLE_RATE = 16000
WINDOW_SEC = 2.5

CHECKPOINT_DIR = "checkpoints"

TINY_CHECKPOINT = (
    f"{CHECKPOINT_DIR}/tiny_best.pt"
)

WHISPER_CHECKPOINT = (
    f"{CHECKPOINT_DIR}/whisper_best.pt"
)

# Thresholds
#
# Tiny model:
# No separate tuned threshold was established for the
# final demo, so keep the standard 0.50.
#
# Whisper:
# Your validation threshold tuning found 0.44.
#
TINY_THRESHOLD = 0.50
WHISPER_THRESHOLD = 0.44


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 60)
print("TURN DETECTION — GRADIO DEMO")
print("=" * 60)

print(
    f"Using device: {device}"
)


# ============================================================
# LOAD MODELS
# ============================================================

tiny_model = None
whisper_model = None
feature_extractor = None


# ------------------------------------------------------------
# Load Tiny CNN-GRU
# ------------------------------------------------------------

print()
print("Loading TinyCNNGRU...")

try:

    tiny_model = TinyCNNGRU().to(device)

    tiny_model.load_state_dict(
        torch.load(
            TINY_CHECKPOINT,
            map_location=device
        )
    )

    tiny_model.eval()

    print(
        "✓ TinyCNNGRU loaded"
    )

    print(
        f"  Checkpoint: {TINY_CHECKPOINT}"
    )

    print(
        f"  Parameters: "
        f"{tiny_model.count_params():,}"
    )

except Exception as e:

    print(
        f"⚠️ TinyCNNGRU could not be loaded:"
    )

    print(e)

    tiny_model = None


# ------------------------------------------------------------
# Load Whisper
# ------------------------------------------------------------

print()
print("Loading Whisper-tiny...")

try:

    from transformers import (
        WhisperFeatureExtractor
    )

    feature_extractor = (
        WhisperFeatureExtractor
        .from_pretrained(
            "openai/whisper-tiny"
        )
    )

    whisper_model = WhisperTurnClassifier(
        freeze_encoder=True,
        unfreeze_last_n=0,
    ).to(device)

    whisper_model.load_state_dict(
        torch.load(
            WHISPER_CHECKPOINT,
            map_location=device
        )
    )

    whisper_model.eval()

    print(
        "✓ Whisper model loaded"
    )

    print(
        f"  Checkpoint: "
        f"{WHISPER_CHECKPOINT}"
    )

except Exception as e:

    print(
        "⚠️ Whisper model could not be loaded:"
    )

    print(e)

    whisper_model = None
    feature_extractor = None


# ============================================================
# AUDIO PREPROCESSING
# ============================================================

def prepare_audio(
    audio,
    sample_rate
):
    """
    Convert incoming Gradio audio into:

        mono float32
        16 kHz
        final 2.5 second window

    Returns:
        audio_16k
    """

    if audio is None:
        return None

    audio = np.asarray(
        audio,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Stereo -> mono
    # --------------------------------------------------------

    if audio.ndim > 1:

        audio = np.mean(
            audio,
            axis=1
        )

    # --------------------------------------------------------
    # Remove NaN / Inf
    # --------------------------------------------------------

    audio = np.nan_to_num(
        audio
    )

    # --------------------------------------------------------
    # Normalize only if necessary
    # --------------------------------------------------------

    max_abs = np.max(
        np.abs(audio)
    )

    if max_abs > 1.0:

        audio = (
            audio /
            max_abs
        )

    # --------------------------------------------------------
    # Resample to 16 kHz
    # --------------------------------------------------------

    if sample_rate != SAMPLE_RATE:

        audio = librosa.resample(
            audio,
            orig_sr=sample_rate,
            target_sr=SAMPLE_RATE
        )

    # --------------------------------------------------------
    # Ensure float32
    # --------------------------------------------------------

    audio = audio.astype(
        np.float32
    )

    # --------------------------------------------------------
    # Keep final 2.5 seconds
    # --------------------------------------------------------

    target_len = int(
        WINDOW_SEC *
        SAMPLE_RATE
    )

    if len(audio) > target_len:

        audio = audio[-target_len:]

    elif len(audio) < target_len:

        audio = np.pad(
            audio,
            (
                target_len - len(audio),
                0
            ),
            mode="constant"
        )

    return audio


# ============================================================
# TINY MODEL FEATURES
# ============================================================

def prepare_tiny_features(
    audio
):
    """
    Create 64-bin normalized log-mel spectrogram.

    This matches the TinyCNNGRU preprocessing used
    during training.
    """

    mel = librosa.feature.melspectrogram(
        y=audio.astype(
            np.float32
        ),
        sr=SAMPLE_RATE,
        n_mels=64,
        hop_length=160,
        n_fft=400,
    )

    log_mel = librosa.power_to_db(
        mel,
        ref=np.max
    ).astype(
        np.float32
    )

    # Same normalization used in demo/training pipeline
    log_mel = (
        log_mel -
        log_mel.mean()
    ) / (
        log_mel.std() +
        1e-6
    )

    tensor = torch.from_numpy(
        log_mel
    ).unsqueeze(0)

    return tensor


# ============================================================
# TINY PREDICTION
# ============================================================

def predict_tiny(
    audio
):
    """
    Run TinyCNNGRU inference.
    """

    if tiny_model is None:

        return {
            "probability": None,
            "prediction": None,
            "latency_ms": None,
        }

    features = prepare_tiny_features(
        audio
    ).to(device)

    start = time.perf_counter()

    with torch.no_grad():

        logits = tiny_model(
            features
        )

        probability = torch.sigmoid(
            logits
        ).item()

    latency_ms = (
        time.perf_counter() -
        start
    ) * 1000

    prediction = (
        probability >=
        TINY_THRESHOLD
    )

    return {
        "probability": probability,
        "prediction": prediction,
        "latency_ms": latency_ms,
    }


# ============================================================
# WHISPER PREDICTION
# ============================================================

def predict_whisper(
    audio
):
    """
    Run Whisper-tiny inference.
    """

    if (
        whisper_model is None
        or feature_extractor is None
    ):

        return {
            "probability": None,
            "prediction": None,
            "latency_ms": None,
        }

    features = feature_extractor(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
    )

    input_features = (
        features["input_features"]
        .to(device)
    )

    start = time.perf_counter()

    with torch.no_grad():

        logits = whisper_model(
            input_features
        )

        probability = torch.sigmoid(
            logits
        ).item()

    latency_ms = (
        time.perf_counter() -
        start
    ) * 1000

    prediction = (
        probability >=
        WHISPER_THRESHOLD
    )

    return {
        "probability": probability,
        "prediction": prediction,
        "latency_ms": latency_ms,
    }


# ============================================================
# FORMAT RESULT
# ============================================================

def format_result(
    result,
    threshold,
    model_name
):

    if result["probability"] is None:

        return (
            f"### {model_name}\n\n"
            "⚠️ Model not loaded."
        )

    probability = (
        result["probability"]
    )

    prediction = (
        result["prediction"]
    )

    latency = (
        result["latency_ms"]
    )

    if prediction:

        decision = (
            "🟢 **TURN COMPLETE**"
        )

    else:

        decision = (
            "🟡 **KEEP LISTENING**"
        )

    return (
        f"### {model_name}\n\n"
        f"**P(turn complete):** "
        f"`{probability:.3f}`\n\n"
        f"**Threshold:** "
        f"`{threshold:.2f}`\n\n"
        f"**Prediction:** "
        f"{decision}\n\n"
        f"**Inference latency:** "
        f"`{latency:.1f} ms`"
    )


# ============================================================
# MAIN PREDICTION FUNCTION
# ============================================================

def predict(audio):
    """
    Main Gradio prediction function.
    """

    if audio is None:

        return (
            "⚠️ Please record or upload audio.",
            "⚠️ Please record or upload audio.",
        )

    try:

        sample_rate, waveform = audio

        # ----------------------------------------------------
        # Prepare audio
        # ----------------------------------------------------

        processed_audio = prepare_audio(
            waveform,
            sample_rate
        )

        if processed_audio is None:

            return (
                "⚠️ Could not process audio.",
                "⚠️ Could not process audio.",
            )

        # ----------------------------------------------------
        # Tiny
        # ----------------------------------------------------

        tiny_result = predict_tiny(
            processed_audio
        )

        # ----------------------------------------------------
        # Whisper
        # ----------------------------------------------------

        whisper_result = predict_whisper(
            processed_audio
        )

        # ----------------------------------------------------
        # Format
        # ----------------------------------------------------

        tiny_text = format_result(
            tiny_result,
            TINY_THRESHOLD,
            "Tiny CNN-GRU — Efficient / Scratch"
        )

        whisper_text = format_result(
            whisper_result,
            WHISPER_THRESHOLD,
            "Whisper-tiny — Accurate / Pretrained"
        )

        return (
            tiny_text,
            whisper_text,
        )

    except Exception as e:

        error = (
            "❌ Error during inference:\n\n"
            f"`{str(e)}`"
        )

        return (
            error,
            error,
        )


# ============================================================
# GRADIO UI
# ============================================================

with gr.Blocks(
    title="Turn Detection Demo"
) as demo:

    gr.Markdown(
        """
# 🎤 Turn Detection

### Is the speaker finished speaking?

Record or upload a short audio clip and compare
two turn-detection models:

| Model | Approach | Goal |
|---|---|---|
| **Tiny CNN-GRU** | Trained from scratch | ⚡ Efficient |
| **Whisper-tiny** | Pretrained multilingual encoder | 🎯 Accurate |

The models analyze the **final 2.5 seconds** of the audio
and estimate the probability that the speaker has completed
their turn.
"""
    )

    gr.Markdown(
        """
### 🎧 Try these examples

**Example 1 — Complete thought**

> "I have completed the task."

Expected result: **TURN COMPLETE**

**Example 2 — Incomplete thought**

> "I was thinking that maybe..."

Expected result: **KEEP LISTENING**

**Example 3 — Hinglish**

> "Actually mujhe lagta hai ki..."

Try both complete and incomplete Hinglish sentences.
"""
    )

    audio_input = gr.Audio(
        sources=[
            "microphone",
            "upload"
        ],
        type="numpy",
        label=(
            "🎤 Record or upload audio"
        ),
    )

    predict_button = gr.Button(
        "🔍 Analyze Turn",
        variant="primary",
    )

    with gr.Row():

        tiny_output = gr.Markdown(
            label="Tiny CNN-GRU"
        )

        whisper_output = gr.Markdown(
            label="Whisper-tiny"
        )

    predict_button.click(
        fn=predict,
        inputs=audio_input,
        outputs=[
            tiny_output,
            whisper_output,
        ],
    )

    gr.Markdown(
        """
---

### Model comparison

**Tiny CNN-GRU**
- Trained completely from scratch
- Uses log-mel spectrograms
- Lightweight architecture
- Designed for CPU-efficient inference

**Whisper-tiny**
- Uses pretrained Whisper-tiny encoder
- Multilingual representation
- Better suited to Hindi/English/Hinglish audio
- Higher accuracy in our current evaluation

**Whisper validation F1:** 0.8421 on the clean real test set.
"""
    )


# ============================================================
# LAUNCH
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("STARTING GRADIO")
    print("=" * 60)

    print(
        f"Tiny threshold   : "
        f"{TINY_THRESHOLD:.2f}"
    )

    print(
        f"Whisper threshold : "
        f"{WHISPER_THRESHOLD:.2f}"
    )

    print()

    demo.launch()

