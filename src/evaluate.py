"""
Full evaluation: F1/precision/recall, latency benchmark (ms/inference and
model size), plus a targeted error breakdown using the dataset's own
`endfiller` flag (trails off after a filler word) and `language` field —
this is the section your report should lean on, since it's what the
assignment explicitly asked about.

Run:
    python src/evaluate.py --model whisper --checkpoint checkpoints/whisper_best.pt \
        --manifest data/processed/manifest.csv
    python src/evaluate.py --model tiny --checkpoint checkpoints/tiny_best.pt \
        --manifest data/processed/manifest.csv
"""
import argparse
import time
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

from dataset import load_manifest, TurnDataset, TurnDatasetRaw
from model import WhisperTurnClassifier, TinyCNNGRU


def benchmark_latency(model, sample_input, n_runs=50):
    model.eval()
    with torch.no_grad():
        for _ in range(5):  # warmup
            model(sample_input)
        start = time.perf_counter()
        for _ in range(n_runs):
            model(sample_input)
        elapsed = time.perf_counter() - start
    return (elapsed / n_runs) * 1000  # ms


def search_best_threshold(probs, labels, steps=25):
    """
    0.5 is an arbitrary default cutoff on the sigmoid output. Sweeping it
    and picking whatever maximizes val F1 is a free accuracy gain — report
    both the default and the tuned threshold in your write-up.
    """
    best_t, best_f1 = 0.5, -1
    for t in np.linspace(0.05, 0.95, steps):
        preds = (probs > t).astype(float)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


def get_probs_labels(model, ds, batch_size=16):
    loader = DataLoader(ds, batch_size=batch_size)
    model.eval()
    all_probs, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            p = torch.sigmoid(logits)
            all_probs.extend(p.numpy())
            labels.extend(y.numpy())
    return np.array(all_probs), np.array(labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["whisper", "tiny"], required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", default="data/processed/manifest.csv")
    ap.add_argument("--split", default="test",
                     help="split to report final metrics on — should be 'test'")
    ap.add_argument("--tune_split", default="val",
                     help="split to tune the decision threshold on — must be "
                          "different from --split, never tune on your report split")
    args = ap.parse_args()

    if args.split == args.tune_split:
        print(f"WARNING: --split and --tune_split are both '{args.split}'. "
              f"Tuning the threshold on the same data you report metrics on is "
              f"leakage — it will make your numbers look better than they really "
              f"are. Use --tune_split val --split test.")

    device = torch.device("cpu")  # benchmark on CPU since that's the deployment target
    manifest = load_manifest(args.manifest)

    def build_ds(split):
        if args.model == "whisper":
            from transformers import WhisperFeatureExtractor
            fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")
            return TurnDataset(manifest, split, fe)
        return TurnDatasetRaw(manifest, split)

    if args.model == "whisper":
        model = WhisperTurnClassifier().to(device)
    else:
        model = TinyCNNGRU().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    report_ds = build_ds(args.split)

    # Tune the threshold ONLY on tune_split (default: val), never on the
    # split you're reporting final numbers for.
    if args.tune_split != args.split:
        tune_ds = build_ds(args.tune_split)
        tune_probs, tune_labels = get_probs_labels(model, tune_ds)
        best_t, tune_f1 = search_best_threshold(tune_probs, tune_labels)
        print(f"Tuned threshold on '{args.tune_split}' split: {best_t:.3f} "
              f"(F1={tune_f1:.4f} on {args.tune_split})")
    else:
        best_t = 0.5
        print("No separate tune_split given — using default threshold 0.5.")

    all_probs, labels = get_probs_labels(model, report_ds)
    preds_default = (all_probs > 0.5).astype(float)
    preds_tuned = (all_probs > best_t).astype(float)

    # derive the benchmark input shape from a real sample rather than guessing
    sample_x, _ = report_ds[0]
    sample_shape = tuple([1] + list(sample_x.shape))

    print(f"\n=== {args.model} model on '{args.split}' split (held out, "
          f"not used for tuning) ===")
    print(f"F1 (threshold=0.5):         {f1_score(labels, preds_default):.4f}")
    print(f"F1 (tuned threshold={best_t:.3f}): {f1_score(labels, preds_tuned):.4f}")
    print(f"Precision (tuned): {precision_score(labels, preds_tuned, zero_division=0):.4f}")
    print(f"Recall (tuned):    {recall_score(labels, preds_tuned, zero_division=0):.4f}")
    print(f"Confusion matrix (tuned threshold):\n{confusion_matrix(labels, preds_tuned)}")

    n_params = sum(p.numel() for p in model.parameters())
    dummy = torch.randn(sample_shape)
    latency_ms = benchmark_latency(model, dummy)
    print(f"\nParams: {n_params:,}")
    print(f"CPU latency: {latency_ms:.2f} ms/inference (batch size 1)")

    # Error breakdown using the dataset's own endfiller / language columns —
    # uses the tuned threshold's predictions, computed only on the report split
    sub_df = manifest[manifest["split"] == args.split].reset_index(drop=True)
    preds = preds_tuned

    if "endfiller" in sub_df.columns:
        filler_mask = sub_df["endfiller"].astype(bool)
        if filler_mask.sum() > 0:
            filler_labels = [labels[i] for i in range(len(labels)) if filler_mask.iloc[i]]
            filler_preds = [preds[i] for i in range(len(preds)) if filler_mask.iloc[i]]
            print(f"\n=== endfiller=True clips (trails off after a filler word, n={filler_mask.sum()}) ===")
            print(f"F1 on this subset: {f1_score(filler_labels, filler_preds):.4f}")
            print("(compare against overall F1 above — a gap here is exactly the "
                  "filler-word weakness worth writing up)")

    if "language" in sub_df.columns:
        print("\n=== F1 by language (top 8 by count) ===")
        for lang, group in sub_df.groupby("language"):
            if len(group) < 5:
                continue
            idx = group.index.tolist()
            lang_labels = [labels[i] for i in idx]
            lang_preds = [preds[i] for i in idx]
            print(f"  {lang} (n={len(idx)}): F1={f1_score(lang_labels, lang_preds, zero_division=0):.4f}")


if __name__ == "__main__":
    main()
