# Hinglish_turn_detection
# Hinglish Turn Detection

A lightweight audio-based **End-of-Turn (EOT) detection system** for Hindi-English (Hinglish) speech.

The system predicts whether a speaker has:

* **0 — KEEP LISTENING:** the speaker may still be speaking, pausing, or trailing off.
* **1 — TURN COMPLETE:** the speaker has likely finished their turn.

The project implements **two model tiers**:

1. **TinyCNNGRU** — a lightweight model trained from scratch on log-mel spectrograms.
2. **Whisper-tiny + classification head** — a pretrained multilingual Whisper encoder with a lightweight classification head.

The Gradio application in `src/demo_app.py` allows both models to be tested side-by-side on the same audio input.

---

## Project Overview

The goal is to build a turn detector that can work with conversational speech, including:

* English
* Hindi
* Hinglish / code-switched speech
* Pauses
* Filler words
* Incomplete thoughts
* Complete utterances

For example:

> "Kal mujhe office jaana hai..."
> → KEEP LISTENING

> "Kal mujhe office jaana hai, meeting hai."
> → TURN COMPLETE

The model does not perform speech-to-text. It directly analyzes the audio signal and predicts whether the current speaker turn is complete.

---

## Model Architecture

### 1. TinyCNNGRU — Efficient Tier

The TinyCNNGRU model is trained **from scratch**, without using a pretrained speech encoder.

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

The model is designed for CPU-friendly inference and provides the lightweight/efficient model tier.

The architecture contains:

* 2 convolutional blocks
* Batch normalization
* ReLU activation
* Max pooling
* Bidirectional GRU
* Attention pooling
* Fully connected classification head

The parameter count can be obtained directly using:

```bash
python -c "import sys; sys.path.insert(0,'src'); from model import TinyCNNGRU; print(f'{TinyCNNGRU().count_params():,}')"
```

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

The Whisper encoder is initially frozen and the classification head is trained for the turn-detection task.

This provides a stronger multilingual representation and is particularly useful for Hindi/English conversational speech.

---

## Why Two Models?

The two models represent a practical accuracy-vs-efficiency trade-off.

| Model        | Training          | Encoder            | Main Advantage                 |
| ------------ | ----------------- | ------------------ | ------------------------------ |
| TinyCNNGRU   | From scratch      | CNN + GRU          | Lightweight and CPU-friendly   |
| Whisper-tiny | Transfer learning | Pretrained Whisper | Stronger speech representation |

The project therefore demonstrates both:

**Efficient model → TinyCNNGRU**

and

**Higher-accuracy model → Whisper-tiny**

The Gradio demo displays predictions and inference latency for both models.

---

## Dataset

The project uses the Smart Turn dataset as the base dataset.

The dataset contains English and Hindi-labelled samples, with the Hindi portion used in the project being synthetic.

Additional synthetic Hinglish samples were generated to increase exposure to Hindi-English conversational patterns.

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

This is important because evaluating only on synthetic samples could give an overly optimistic estimate of real-world performance.

---

## Audio Processing

Audio is processed at:

```text
Sample rate : 16 kHz
Window      : 2.5 seconds
```

For the TinyCNNGRU:

```text
64 Mel-frequency bins
Hop length = 160
FFT size   = 400
```

For Whisper:

```text
WhisperFeatureExtractor
openai/whisper-tiny
```

For variable-length recordings, the final 2.5 seconds are used for prediction.

---

## Training

### Train TinyCNNGRU

From the project root:

```bash
python src/train.py --model tiny --epochs 15 --manifest data/processed_test/manifest.csv --extra_manifest hinglish_data/manifest.csv --hinglish_oversample 1
```

The best model is saved as:

```text
checkpoints/tiny_best.pt
```

### Train Whisper

```bash
python src/train.py --model whisper --epochs 6 --freeze_encoder --manifest data/processed_test/manifest.csv --extra_manifest hinglish_data/manifest.csv --hinglish_oversample 1
```

The best Whisper checkpoint is saved as:

```text
checkpoints/whisper_best.pt
```

---

## Threshold Tuning

The classifier produces a probability:

```text
P(Turn Complete)
```

A default threshold of `0.50` corresponds to:

```text
probability >= 0.50 → TURN COMPLETE
probability <  0.50 → KEEP LISTENING
```

The training pipeline also performs validation-based threshold tuning.

The threshold is selected using the **validation set only** and is then applied to the test set.

This prevents the test set from being used to optimize the decision threshold.

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

These results should be interpreted cautiously because the clean real test set used in this experiment is small.

---

## Gradio Demo

The main interactive demo is:

```text
src/demo_app.py
```

Run it with:

```bash
python src/demo_app.py
```

The application allows the user to:

1. Record audio using the microphone.
2. Upload an audio file.
3. Send the same audio to both models.
4. View the predicted probability.
5. View whether the model predicts `TURN COMPLETE` or `STILL SPEAKING`.
6. Compare inference latency between the two model tiers.

The demo displays:

```text
Tiny CNN-GRU (efficient tier)
              vs.
Whisper-tiny + head (accurate tier)
```

---

## Repository Structure

```text
Hinglish_turn_detection/
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

Prepares the Smart Turn dataset and creates the processed manifest.

**`src/generate_synthetic_hinglish.py`**

Creates synthetic Hinglish augmentation data.

**`src/train.py`**

Contains the complete training pipeline for both TinyCNNGRU and Whisper.

**`src/evaluate.py`**

Evaluates the trained model.

**`src/evaluate_hinglish.py`**

Evaluates performance on the Hinglish evaluation data.

**`src/evaluate_whisper.py`**

Evaluates the Whisper model, including the clean real test evaluation.

**`src/analyze_real_test.py`**

Analyzes the real test set.

**`src/demo_app.py`**

Gradio application for comparing the TinyCNNGRU and Whisper models.

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

## Running the Demo

Install dependencies:

```bash
pip install -r requirements.txt
```

Make sure the trained checkpoints exist:

```text
checkpoints/
├── tiny_best.pt
└── whisper_best.pt
```

Then run:

```bash
python src/demo_app.py
```

Gradio will provide a local URL in the terminal.

---

## Reproducibility

The training pipeline uses a fixed random seed by default:

```text
seed = 42
```

The train/validation/test split is kept separate, and validation data is used for model selection and threshold tuning.

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
3. Adding real-world background noise augmentation.
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
             ┌────────┴────────┐
             │                 │
       TinyCNNGRU          Whisper-tiny
       From scratch       Pretrained encoder
             │                 │
        Efficient          More accurate
             │                 │
             └────────┬────────┘
                      │
              Turn prediction
                      │
          ┌───────────┴───────────┐
          │                       │
   KEEP LISTENING          TURN COMPLETE
```

The final system provides both a lightweight model suitable for CPU-oriented deployment and a stronger pretrained baseline for comparison.
