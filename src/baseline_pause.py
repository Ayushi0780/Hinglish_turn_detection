"""
Baseline 0: the thing every naive voice-AI system does — trailing silence
duration crosses a fixed threshold means "user is done". This exists purely
to show, quantitatively, why it fails on filler-word pauses and thinking
gaps (which is the whole reason a learned model is needed).

Run:
    python src/baseline_pause.py --manifest data/processed/manifest.csv --threshold_ms 700
"""
import argparse
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from sklearn.metrics import f1_score, precision_score, recall_score

SAMPLE_RATE = 16000


def trailing_silence_ms(audio: np.ndarray, sr: int, top_db: float = 30) -> float:
    """Duration of silence at the very end of the clip, in milliseconds."""
    non_silent = librosa.effects.split(audio, top_db=top_db)
    if len(non_silent) == 0:
        return len(audio) / sr * 1000
    last_voiced_end = non_silent[-1][1]
    trailing_samples = len(audio) - last_voiced_end
    return trailing_samples / sr * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/processed/manifest.csv")
    ap.add_argument("--threshold_ms", type=float, default=700)
    ap.add_argument("--split", default="val")
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    df = df[df["split"] == args.split]

    preds, labels = [], []
    for _, row in df.iterrows():
        audio, sr = sf.read(row["path"])
        if sr != SAMPLE_RATE:
            audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=SAMPLE_RATE)
        silence_ms = trailing_silence_ms(audio, SAMPLE_RATE)
        pred = 1 if silence_ms >= args.threshold_ms else 0
        preds.append(pred)
        labels.append(row["label"])

    print(f"Naive silence-threshold baseline ({args.threshold_ms}ms):")
    print(f"  F1:        {f1_score(labels, preds):.4f}")
    print(f"  Precision: {precision_score(labels, preds, zero_division=0):.4f}")
    print(f"  Recall:    {recall_score(labels, preds, zero_division=0):.4f}")
    print("This baseline should struggle most on 'incomplete' clips with a long "
          "pause after a filler word — check those rows specifically in your report.")


if __name__ == "__main__":
    main()
