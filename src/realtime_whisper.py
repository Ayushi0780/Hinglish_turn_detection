import time
import numpy as np
import sounddevice as sd
import torch

from transformers import WhisperFeatureExtractor
from model import WhisperTurnClassifier


# ============================================================
# SETTINGS
# ============================================================

SAMPLE_RATE = 16000
WINDOW_SEC = 2.5

CHECKPOINT = "checkpoints/whisper_best.pt"

# Model threshold from your validation tuning
THRESHOLD = 0.50

# ------------------------------------------------------------
# Voice Activity Detection
# ------------------------------------------------------------

# Automatically calculated after calibration
ENERGY_THRESHOLD = None

CALIBRATION_SEC = 3.0

# Silence required after speech
SILENCE_DURATION = 0.8

# Require several consecutive chunks above threshold before
# declaring that speech has started. This prevents noise spikes.
SPEECH_CONFIRM_CHUNKS = 3

# Minimum speech duration
MIN_SPEECH_DURATION = 0.6

# Maximum recording duration
MAX_RECORDING_DURATION = 12.0

# Small amount of audio before speech starts
PRE_SPEECH_SEC = 0.3


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("REAL-TIME TURN DETECTION")
print("=" * 60)

print(f"Using device: {device}")


# ============================================================
# LOAD WHISPER
# ============================================================

print("\nLoading Whisper feature extractor...")

feature_extractor = WhisperFeatureExtractor.from_pretrained(
    "openai/whisper-tiny"
)

print("Loading Whisper model...")

model = WhisperTurnClassifier(
    freeze_encoder=True,
    unfreeze_last_n=0,
).to(device)

model.load_state_dict(
    torch.load(
        CHECKPOINT,
        map_location=device
    )
)

model.eval()

print("✓ Model loaded")


# ============================================================
# MICROPHONE CALIBRATION
# ============================================================

def calibrate_microphone():

    print()
    print("=" * 60)
    print("🎧 MICROPHONE CALIBRATION")
    print("=" * 60)

    print()
    print(
        f"Please stay SILENT for {CALIBRATION_SEC:.1f} seconds."
    )
    print(
        "Measuring your laptop's background noise..."
    )

    samples = []

    def callback(
        indata,
        frames,
        time_info,
        status
    ):

        if status:
            print(
                f"\nAudio status: {status}"
            )

        samples.extend(
            indata[:, 0].copy()
        )

    blocksize = int(
        SAMPLE_RATE * 0.1
    )

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=blocksize,
        callback=callback,
    ):

        time.sleep(
            CALIBRATION_SEC
        )

    if not samples:

        print(
            "⚠️ Could not capture microphone data."
        )

        return 0.003

    samples = np.asarray(
        samples,
        dtype=np.float32
    )

    # Calculate RMS in small windows
    window = int(
        SAMPLE_RATE * 0.1
    )

    rms_values = []

    for start in range(
        0,
        len(samples) - window,
        window
    ):

        chunk = samples[
            start:start + window
        ]

        rms = np.sqrt(
            np.mean(chunk ** 2)
        )

        rms_values.append(rms)

    if not rms_values:

        return 0.003

    noise_rms = np.median(
        rms_values
    )

    # Threshold comfortably above background noise
    # Use both the typical noise level and the loudest calibration
    # spikes so occasional laptop/mic noise does not trigger speech.
    noise_peak = np.percentile(rms_values, 95)

    threshold = max(
        0.0025,
        noise_rms * 4.0,
        noise_peak * 1.5,
    )

    print()
    print(
        f"Background noise RMS : {noise_rms:.5f}"
    )

    print(
        f"Speech threshold      : {threshold:.5f}"
    )

    print()
    print("✓ Calibration complete.")

    return threshold


# ============================================================
# MODEL PREDICTION
# ============================================================

def predict(audio):

    target_len = int(
        WINDOW_SEC * SAMPLE_RATE
    )

    audio = np.asarray(
        audio,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Remove trailing silence
    # --------------------------------------------------------

    abs_audio = np.abs(audio)

    indices = np.where(
        abs_audio > ENERGY_THRESHOLD
    )[0]

    if len(indices) > 0:

        last_voice = indices[-1]

        keep_after = int(
            0.25 * SAMPLE_RATE
        )

        end = min(
            len(audio),
            last_voice + keep_after
        )

        audio = audio[:end]

    # --------------------------------------------------------
    # Keep last 2.5 seconds
    # --------------------------------------------------------

    if len(audio) > target_len:

        audio = audio[-target_len:]

    # --------------------------------------------------------
    # Left pad
    # --------------------------------------------------------

    elif len(audio) < target_len:

        audio = np.pad(
            audio,
            (
                target_len - len(audio),
                0
            )
        )

    # --------------------------------------------------------
    # Whisper features
    # --------------------------------------------------------

    features = feature_extractor(
        audio.astype(np.float32),
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
    )

    input_features = features[
        "input_features"
    ].to(device)

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    with torch.no_grad():

        logits = model(
            input_features
        )

        probability = torch.sigmoid(
            logits
        ).item()

    prediction = (
        probability >= THRESHOLD
    )

    return prediction, probability


# ============================================================
# RECORD SPEECH
# ============================================================

def record_until_silence():

    global ENERGY_THRESHOLD

    print()
    print("=" * 60)
    print("🎤 LISTENING")
    print("=" * 60)

    print()
    print("Speak normally...")

    recorded_chunks = []

    speech_started = False
    silence_time = 0.0
    total_time = 0.0
    consecutive_speech_chunks = 0

    chunk_duration = 0.1

    blocksize = int(
        SAMPLE_RATE * chunk_duration
    )

    # Keep a small rolling buffer before speech begins
    pre_speech_chunks = []

    max_pre_chunks = int(
        PRE_SPEECH_SEC / chunk_duration
    )

    def callback(
        indata,
        frames,
        time_info,
        status
    ):

        if status:
            print(
                f"\nAudio status: {status}"
            )

        recorded_chunks.append(
            indata[:, 0].copy()
        )

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=blocksize,
        callback=callback,
    )

    with stream:

        while True:

            time.sleep(
                chunk_duration
            )

            total_time += chunk_duration

            if not recorded_chunks:
                continue

            latest = recorded_chunks[-1]

            rms = np.sqrt(
                np.mean(
                    latest ** 2
                )
            )

            # =================================================
            # WAITING FOR SPEECH
            # =================================================

            if not speech_started:

                pre_speech_chunks.append(
                    latest
                )

                if len(
                    pre_speech_chunks
                ) > max_pre_chunks:

                    pre_speech_chunks.pop(
                        0
                    )

                if rms > ENERGY_THRESHOLD:

                    consecutive_speech_chunks += 1

                else:

                    # A noise spike should not count as speech.
                    consecutive_speech_chunks = 0

                if (
                    consecutive_speech_chunks
                    >= SPEECH_CONFIRM_CHUNKS
                ):

                    speech_started = True

                    # Include a little audio before speech.
                    recorded_chunks = (
                        pre_speech_chunks
                        + recorded_chunks[
                            -1:
                        ]
                    )

                    print()
                    print(
                        "🗣️ SPEECH DETECTED"
                    )

                    print(
                        f"Volume: {rms:.5f}"
                    )

                else:

                    print(
                        f"Waiting... "
                        f"background={rms:.5f} "
                        f"speech_hits={consecutive_speech_chunks}/{SPEECH_CONFIRM_CHUNKS}",
                        end="\r"
                    )

            # =================================================
            # SPEAKING
            # =================================================

            else:

                if rms > ENERGY_THRESHOLD:

                    silence_time = 0.0

                    print(
                        f"🗣️ Speaking... "
                        f"volume={rms:.5f}",
                        end="\r"
                    )

                else:

                    silence_time += (
                        chunk_duration
                    )

                    print(
                        f"🤫 Silence: "
                        f"{silence_time:.1f}s",
                        end="\r"
                    )

                    # -----------------------------------------
                    # END OF SPEECH
                    # -----------------------------------------

                    if (
                        silence_time
                        >= SILENCE_DURATION
                    ):

                        print()
                        print(
                            "✓ END OF SPEECH"
                        )

                        break

            # =================================================
            # MAXIMUM RECORDING
            # =================================================

            if (
                total_time
                >= MAX_RECORDING_DURATION
            ):

                print()

                if not speech_started:
                    print(
                        "⚠️ No speech detected within "
                        f"{MAX_RECORDING_DURATION:.0f} seconds."
                    )
                    print(
                        "Returning to listening..."
                    )
                else:
                    print(
                        "⚠️ Maximum recording "
                        "duration reached."
                    )

                break

    if not recorded_chunks or not speech_started:

        return np.array(
            [],
            dtype=np.float32
        )

    audio = np.concatenate(
        recorded_chunks
    )

    return audio


# ============================================================
# START
# ============================================================

print()
print("First we need to calibrate your microphone.")

ENERGY_THRESHOLD = calibrate_microphone()


# ============================================================
# READY
# ============================================================

print()
print("=" * 60)
print("🚀 REAL-TIME DEMO READY")
print("=" * 60)

print()
print(
    "You do NOT need to press Enter."
)

print(
    "Just speak when you see:"
)

print(
    "🎤 LISTENING"
)

print()
print(
    "The app automatically detects:"
)

print(
    "  1. Speech start"
)

print(
    "  2. Speech end"
)

print(
    "  3. Model prediction"
)

print()
print(
    "Press Ctrl+C to stop."
)


# ============================================================
# MAIN LOOP
# ============================================================

try:

    while True:

        audio = record_until_silence()

        duration = (
            len(audio)
            / SAMPLE_RATE
        )

        print()
        print(
            f"Recorded: {duration:.2f} seconds"
        )

        # ----------------------------------------------------
        # Too short
        # ----------------------------------------------------

        if duration < MIN_SPEECH_DURATION:

            print(
                "⚠️ No usable speech recording."
            )
            print(
                "Waiting for the next speech..."
            )

            continue

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        print()
        print(
            "🧠 Analyzing..."
        )

        prediction, probability = predict(
            audio
        )

        print()
        print("=" * 60)

        print(
            f"Probability: {probability:.3f}"
        )

        print(
            f"Threshold  : {THRESHOLD:.2f}"
        )

        print()

        if prediction:

            print(
                "🟢 TURN COMPLETE"
            )

        else:

            print(
                "🟡 KEEP LISTENING"
            )

        print("=" * 60)

        # Small pause before next listening cycle
        time.sleep(0.5)


except KeyboardInterrupt:

    print()
    print()
    print("=" * 60)
    print("Stopped.")
    print("=" * 60)