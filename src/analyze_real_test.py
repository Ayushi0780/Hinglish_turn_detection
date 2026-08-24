import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from dataset import load_manifest, TurnDatasetRaw
from model import TinyCNNGRU


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MANIFEST = "data/processed_test/manifest.csv"
HINGLISH = "hinglish_data/manifest.csv"
CHECKPOINT = "checkpoints/tiny_best.pt"

THRESHOLD = 0.57


print("=" * 60)
print("REAL TEST ANALYSIS")
print("=" * 60)

# Load exactly the same manifest used during training
df = load_manifest(
    MANIFEST,
    HINGLISH,
    oversample=1
)

test_df = df[df["split"] == "test"].reset_index(drop=True)

print(f"\nReal test samples: {len(test_df)}")

print("\nDistribution:")
print(test_df.groupby(["language", "label"]).size())


dataset = TurnDatasetRaw(
    manifest_df=df,
    split="test",
    window_sec=2.5,
    n_mels=64,
    cache_dir="cache/mel"
)

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)

model = TinyCNNGRU().to(DEVICE)

model.load_state_dict(
    torch.load(CHECKPOINT, map_location=DEVICE)
)

model.eval()

all_probs = []
all_labels = []

with torch.no_grad():

    for x, y in loader:

        x = x.to(DEVICE)

        logits = model(x)

        probs = torch.sigmoid(logits)

        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(y.numpy())


all_probs = np.array(all_probs)
all_labels = np.array(all_labels)

preds = (all_probs >= THRESHOLD).astype(int)

test_df["probability"] = all_probs
test_df["prediction"] = preds

print("\nOverall Metrics")
print("-" * 40)

print("Accuracy :", round(accuracy_score(all_labels, preds),4))
print("F1       :", round(f1_score(all_labels, preds),4))

print("\nConfusion Matrix")
print(confusion_matrix(all_labels, preds))


print("\nPer Language Analysis")
print("-" * 40)

for lang in test_df["language"].unique():

    sub = test_df[test_df["language"] == lang]

    acc = accuracy_score(sub["label"], sub["prediction"])

    f1 = f1_score(
        sub["label"],
        sub["prediction"],
        zero_division=0
    )

    print(f"\nLanguage: {lang}")
    print(f"Samples : {len(sub)}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1      : {f1:.4f}")

    print("Label distribution:")
    print(sub.groupby("label").size())


print("\nPrediction Details")
print("-" * 40)

print(
    test_df[
        [
            "language",
            "label",
            "prediction",
            "probability"
        ]
    ].to_string(index=False)
)

print("\nDone.")