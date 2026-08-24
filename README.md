# Hinglish Turn Detection

A lightweight audio-based **End-of-Turn (EOT) detection system** for Hindi-English (Hinglish) conversational speech.

The system predicts whether a speaker has:

* **0 — KEEP LISTENING:** the speaker may still be speaking, pausing, or trailing off.
* **1 — TURN COMPLETE:** the speaker has likely finished their turn.

The project implements two complementary model tiers:

1. **TinyCNNGRU** — a lightweight model trained completely from scratch on log-mel spectrograms.
2. **Whisper-tiny + classification head** — a pretrained multilingual Whisper encoder with a lightweight classification head.

The Gradio application in `src/demo_app.py` allows both models to be tested side-by-side on the same audio input.

---

## Project Overview

The goal is to build a turn detector for conversational speech containing:

* English
* Hindi
* Hinglish / code-switched speech
* Pauses
* Filler words
* Incomplete thoughts
* Complete utterances

For example:

> "Kal mujhe office jaana hai..."

→ **KEEP LISTENING**

> "Kal mujhe office jaana hai, meeting hai."

→ **TURN COMPLETE**

The system does **not** perform speech-to-text. It directly analyzes the audio signal and predicts whether the current speaker turn is complete.

---

## Model Architecture

### 1. TinyCNNGRU — Efficient Tier

TinyCNNGRU is trained **from scratch**, without using a pretrained speech encoder.

Pipeline:

```text
Audio
  ↓
2.5-second trailing window
  ↓
64-bin Mel Spectrogram
  ↓
Log-Mel normalization
  ↓
CNN layers
  ↓
GRU
  ↓
Attention pooling
  ↓
MLP classification head
  ↓
P(Turn Complete)
```

The architecture contains:

* 2 convolutional blocks
* Batch normalization
* ReLU activation
* Max pooling
* Bidirectional GRU
* Attention pooling
* Fully connected classification head

It is designed as the lightweight and CPU-friendly model tier.

---

### 2. Whisper-tiny + Classification Head — Accuracy Tier

The second model uses the pretrained `openai/whisper-tiny` encoder.

Pipeline:

```text
Audio
  ↓
Whisper Feature Extractor
  ↓
Whisper-tiny Encoder
  ↓
Attention Pooling
  ↓
MLP Classification Head
  ↓
P(Turn Complete)
```

The Whisper encoder is frozen in the trained checkpoint and a lightweight classification head is trained for the turn-detection task.

This provides a stronger multilingual speech representation and is useful for Hindi/English conversational speech.

---

## Why Two Models?

The two models demonstrate an accuracy-versus-efficiency trade-off:

| Model        | Training          | Encoder            | Main Advantage                 |
| ------------ | ----------------- | ------------------ | ------------------------------ |
| TinyCNNGRU   | From scratch      | CNN + GRU          | Lightweight and CPU-friendly   |
| Whisper-tiny | Transfer learning | Pretrained Whisper | Stronger speech representation |

The project therefore demonstrates:

**Efficient model → TinyCNNGRU**

**Accuracy-oriented model → Whisper-tiny**

The Gradio demo displays predictions and inference latency for both models on the same audio input.

---

## Dataset

The project uses the Smart Turn dataset as the base dataset.

The available Hindi portion used in the project is synthetic, and additional synthetic Hinglish samples were generated to increase exposure to Hindi-English conversational patterns.

The training pipeline combines:

```text
Smart Turn data
      +
Synthetic Hinglish augmentation
      ↓
Training / Validation / Test
```

### Important evaluation design

The final evaluation uses a **clean real test subset** rather than relying only on synthetic augmentation.

Synthetic test samples are excluded when calculating the final real-test performance.

This provides a more meaningful estimate of performance on real speech.

---

## Audio Processing

Audio is processed at:

```text
Sample rate : 16 kHz
Window      : 2.5 seconds
```

### TinyCNNGRU

```text
Mel bins    : 64
Hop length : 160
FFT size   : 400
```

### Whisper

```text
Feature extractor : WhisperFeatureExtractor
Model             : openai/whisper-tiny
```

For variable-length recordings, the final 2.5 seconds are used for prediction.

---

## Included Model Weights

**The repository already contains the trained model checkpoints.**

You do **not** need to train the models before running the Gradio demo.

```text
checkpoints/
├── tiny_best.pt       (~948 KB)
└── whisper_best.pt    (~32 MB)
```

Approximate total model-weight size:

```text
~33 MB
```

These checkpoints are loaded automatically by `src/demo_app.py`.

Therefore, the evaluator can directly clone the repository, install the dependencies, and run the demo without retraining.

---

## Running the Demo

### 1. Clone the repository

```bash
git clone <repository-url>
cd Hinglish_turn_detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Gradio demo

```bash
python src/demo_app.py
```

Gradio will provide a local URL in the terminal.

Open the URL in a browser and use the microphone or upload an audio file.

### The demo provides

1. Microphone recording
2. Audio-file upload
3. Prediction from TinyCNNGRU
4. Prediction from Whisper-tiny
5. Turn-complete probability
6. Final classification
7. Inference latency for both models

Both models receive the same audio input, making the comparison easy to observe.

---

## Prediction Interpretation

The models output:

```text
P(Turn Complete)
```

Using the default threshold:

```text
probability >= 0.50 → TURN COMPLETE

probability < 0.50 → KEEP LISTENING
```

The threshold was selected/tuned using validation data during experimentation. The clean test set was kept separate from threshold optimization.

---

## Evaluation

The project evaluates:

* Accuracy
* F1 score
* Precision
* Recall
* Confusion matrix

### Whisper real-test result

Using the clean real English test subset:

```text
Samples   : 24
Accuracy  : 0.8750
F1        : 0.8421
Precision : 1.0000
Recall    : 0.7273

Confusion Matrix:

[[13  0]
 [ 3  8]]
```

The clean real test set is relatively small, so these metrics should be interpreted cautiously.

---

## Repository Structure

```text
Hinglish_turn_detection/
│
├── checkpoints/
│   ├── tiny_best.pt
│   └── whisper_best.pt
│
├── src/
│   ├── analyze_real_test.py
│   ├── baseline_classical.py
│   ├── baseline_pause.py
│   ├── data_prep.py
│   ├── dataset.py
│   ├── demo_app.py
│   ├── evaluate.py
│   ├── evaluate_hinglish.py
│   ├── evaluate_whisper.py
│   ├── export_onnx.py
│   ├── generate_synthetic_hinglish.py
│   ├── model.py
│   ├── realtime_whisper.py
│   ├── test_microphone.py
│   └── train.py
│
├── requirements.txt
├── real_test.txt
├── .gitignore
└── README.md
```

### Important source files

**`src/model.py`**

Contains both model architectures:

* `TinyCNNGRU`
* `WhisperTurnClassifier`

**`src/dataset.py`**

Contains dataset loading and audio dataset classes.

**`src/data_prep.py`**

Prepares the Smart Turn data and creates the processed manifest.

**`src/generate_synthetic_hinglish.py`**

Creates synthetic Hinglish augmentation data.

**`src/train.py`**

Contains the training pipeline used to train both model variants.

Training is **not required to run the submitted demo**, because the trained checkpoints are already included in `checkpoints/`.

**`src/evaluate.py`**

Evaluates the trained model.

**`src/evaluate_hinglish.py`**

Evaluates performance on Hinglish evaluation data.

**`src/evaluate_whisper.py`**

Evaluates the Whisper model, including clean real-test evaluation.

**`src/analyze_real_test.py`**

Analyzes the real test set.

**`src/demo_app.py`**

The main Gradio application for comparing TinyCNNGRU and Whisper-tiny.

**`src/realtime_whisper.py`**

Local microphone-based real-time Whisper turn-detection experiment.

**`src/test_microphone.py`**

Checks whether the laptop microphone is capturing usable audio.

**`src/export_onnx.py`**

Provides the ONNX export path for the lightweight model.

**`src/baseline_classical.py`**

Classical audio-feature baseline.

**`src/baseline_pause.py`**

Pause/silence-based baseline.

---

## Training — Optional

Training is **not required for the submitted demo** because trained checkpoints are already included.

The training scripts are retained for reproducibility and further experimentation.

### Train TinyCNNGRU

```bash
python src/train.py --model tiny --epochs 15 --manifest data/processed_test/manifest.csv --extra_manifest hinglish_data/manifest.csv --hinglish_oversample 1
```

Output:

```text
checkpoints/tiny_best.pt
```

### Train Whisper

```bash
python src/train.py --model whisper --epochs 6 --freeze_encoder --manifest data/processed_test/manifest.csv --extra_manifest hinglish_data/manifest.csv --hinglish_oversample 1
```

Output:

```text
checkpoints/whisper_best.pt
```

These commands are provided only for reproduction or further model development.

---

## Experimental Components

The repository also contains several experimental and evaluation components used during development:

```text
baseline_pause.py
baseline_classical.py
evaluate.py
evaluate_hinglish.py
evaluate_whisper.py
analyze_real_test.py
export_onnx.py
realtime_whisper.py
test_microphone.py
```

These were used to understand the problem from multiple perspectives, establish baselines, evaluate the models, test microphone input, and explore lightweight deployment options.

The **submitted interactive demo** is `src/demo_app.py`.

---

## Reproducibility

The training pipeline uses a fixed random seed by default:

```text
seed = 42
```

The train/validation/test split is kept separate.

Validation data is used for model selection and threshold tuning.

The clean real test set is reserved for final evaluation.

---

## Limitations

The current implementation is a prototype rather than a production-scale turn detector.

Important limitations include:

* The available Hindi data is synthetic.
* The clean real test set is relatively small.
* Hinglish conversational coverage is limited.
* Real-world microphones and background noise can affect performance.
* Turn detection depends strongly on the final audio context/window.
* The current model predicts from a trailing audio window rather than maintaining a long conversational state.

These limitations should be considered when interpreting the reported metrics.

---

## Future Improvements

Potential improvements include:

1. Collecting more real Hindi and Hinglish conversational recordings.
2. Increasing speaker diversity.
3. Adding real-world background-noise augmentation.
4. Training with more incomplete-turn examples containing fillers such as:

   * "umm..."
   * "matlab..."
   * "actually..."
   * "woh..."
   * "haan but..."
5. Fine-tuning selected Whisper encoder layers.
6. Knowledge distillation from Whisper to TinyCNNGRU.
7. ONNX/int8 optimization for faster CPU inference.
8. Larger real-world evaluation with unseen speakers.
9. Streaming inference with overlapping audio windows.
10. More robust voice-activity detection before turn classification.

---

## Summary

This project explores two complementary approaches to audio turn detection for Hinglish conversational speech:

```text
                         Audio
                           │
                ┌──────────┴──────────┐
                │                     │
           TinyCNNGRU            Whisper-tiny
           From scratch        Pretrained encoder
                │                     │
             Efficient          Accuracy-oriented
                │                     │
                └──────────┬──────────┘
                           │
                    Turn prediction
                           │
              ┌────────────┴────────────┐
              │                         │
        KEEP LISTENING            TURN COMPLETE
```

The final repository includes the trained model weights, source code, evaluation utilities, and Gradio demo.

**To run the submitted demo, training is not required.**

```bash
# Create virtual environment
python -m venv venv

# Activate it — Windows
venv\Scripts\activate


pip install -r requirements.txt
python src/demo_app.py
```

The system therefore provides both a lightweight model suitable for CPU-oriented inference and a stronger pretrained model for comparison.
