"""
Baseline 2: hand-engineered acoustic features + XGBoost. Much stronger than
the naive silence threshold, and useful in the report as evidence you
understand what actually distinguishes complete vs incomplete turns before
reaching for a neural model. Also extremely fast at inference (microseconds).

Features per clip:
    - trailing silence duration
    - pitch (F0) slope over the last 300ms — falling pitch usually signals
      completion, flat/rising often signals continuation
    - RMS energy slope over the last 300ms — energy decay vs sustain
    - overall clip duration
    - speaking rate proxy (voiced-frame ratio)

Run:
    python src/baseline_classical.py --manifest data/processed/manifest.csv
"""
import argparse
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from xgboost import XGBClassifier

SAMPLE_RATE = 16000


def extract_features(audio: np.ndarray, sr: int) -> dict:
    non_silent = librosa.effects.split(audio, top_db=30)
    if len(non_silent) > 0:
        trailing_silence = (len(audio) - non_silent[-1][1]) / sr * 1000
        voiced_ratio = sum(e - s for s, e in non_silent) / len(audio)
    else:
        trailing_silence = len(audio) / sr * 1000
        voiced_ratio = 0.0

    tail = audio[-int(0.3 * sr):] if len(audio) > int(0.3 * sr) else audio

    f0, voiced_flag, _ = librosa.pyin(
        tail, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr
    )
    f0_valid = f0[~np.isnan(f0)] if f0 is not None else np.array([])
    pitch_slope = 0.0
    if len(f0_valid) >= 2:
        pitch_slope = (f0_valid[-1] - f0_valid[0]) / len(f0_valid)

    rms = librosa.feature.rms(y=tail)[0]
    energy_slope = (rms[-1] - rms[0]) / len(rms) if len(rms) >= 2 else 0.0

    return {
        "trailing_silence_ms": trailing_silence,
        "voiced_ratio": voiced_ratio,
        "pitch_slope": pitch_slope,
        "energy_slope": energy_slope,
        "duration_sec": len(audio) / sr,
    }


def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        audio, sr = sf.read(row["path"])
        if sr != SAMPLE_RATE:
            audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=SAMPLE_RATE)
        feats = extract_features(audio, SAMPLE_RATE)
        feats["label"] = row["label"]
        feats["split"] = row["split"]
        rows.append(feats)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/processed/manifest.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    feat_df = build_feature_table(df)

    feature_cols = ["trailing_silence_ms", "voiced_ratio", "pitch_slope", "energy_slope", "duration_sec"]
    train = feat_df[feat_df["split"] == "train"]
    val = feat_df[feat_df["split"] == "val"]

    clf = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, eval_metric="logloss")
    clf.fit(train[feature_cols], train["label"])

    preds = clf.predict(val[feature_cols])
    print("Classical baseline (XGBoost on acoustic features):")
    print(f"  F1:        {f1_score(val['label'], preds):.4f}")
    print(f"  Precision: {precision_score(val['label'], preds, zero_division=0):.4f}")
    print(f"  Recall:    {recall_score(val['label'], preds, zero_division=0):.4f}")
    print()
    print("Feature importances:")
    for name, imp in sorted(zip(feature_cols, clf.feature_importances_), key=lambda x: -x[1]):
        print(f"  {name}: {imp:.3f}")

    clf.save_model("checkpoints/xgboost_baseline.json")
    print("\nSaved to checkpoints/xgboost_baseline.json")


if __name__ == "__main__":
    main()
