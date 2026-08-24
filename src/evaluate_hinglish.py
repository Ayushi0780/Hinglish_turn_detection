import os
import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

from dataset import TurnDatasetRaw
from model import TinyCNNGRU


# ============================================================
# Configuration
# ============================================================

MANIFEST = "hinglish_data/manifest.csv"
CHECKPOINT = "checkpoints/tiny_best.pt"

WINDOW_SEC = 2.5
BATCH_SIZE = 16

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Load Hinglish manifest
# ============================================================

print("=" * 60)
print("HINGLISH SYNTHETIC EVALUATION")
print("=" * 60)

df = pd.read_csv(MANIFEST)

print("\nOriginal Hinglish dataset:")
print(df.groupby(["split", "label"]).size())

# We evaluate only the samples that were NOT used for training.
# Your Hinglish manifest has:
# train = 256
# val   = 44
#
# Therefore we evaluate validation samples separately.

eval_df = df[df["split"] == "val"].copy()

print("\nEvaluation split:")
print(eval_df.groupby(["split", "label"]).size())

print(
    f"\nNumber of Hinglish evaluation samples: "
    f"{len(eval_df)}"
)


# ============================================================
# Dataset
# ============================================================

dataset = TurnDatasetRaw(
    manifest_df=eval_df,
    split="val",
    window_sec=WINDOW_SEC,
    n_mels=64,
    cache_dir="cache/mel",
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


# ============================================================
# Load model
# ============================================================

print("\nLoading model...")

model = TinyCNNGRU().to(DEVICE)

model.load_state_dict(
    torch.load(
        CHECKPOINT,
        map_location=DEVICE
    )
)

model.eval()

print(
    f"Using device: {DEVICE}"
)

print(
    f"Model parameters: "
    f"{model.count_params():,}"
)


# ============================================================
# Prediction
# ============================================================

all_probs = []
all_labels = []

with torch.no_grad():

    for x, y in loader:

        x = x.to(DEVICE)

        logits = model(x)

        probs = torch.sigmoid(logits)

        all_probs.extend(
            probs.cpu().numpy()
        )

        all_labels.extend(
            y.numpy()
        )


all_probs = np.asarray(all_probs)
all_labels = np.asarray(all_labels)


# ============================================================
# Use the same threshold selected from real validation
# ============================================================

THRESHOLD = 0.57

predictions = (
    all_probs >= THRESHOLD
).astype(int)


# ============================================================
# Metrics
# ============================================================

accuracy = accuracy_score(
    all_labels,
    predictions
)

f1 = f1_score(
    all_labels,
    predictions,
    zero_division=0
)

precision = precision_score(
    all_labels,
    predictions,
    zero_division=0
)

recall = recall_score(
    all_labels,
    predictions,
    zero_division=0
)

cm = confusion_matrix(
    all_labels,
    predictions
)


# ============================================================
# Results
# ============================================================

print("\n" + "=" * 60)
print("HINGLISH VALIDATION RESULTS")
print("=" * 60)

print(
    f"Samples   : {len(all_labels)}"
)

print(
    f"Threshold : {THRESHOLD:.2f}"
)

print(
    f"Accuracy  : {accuracy:.4f}"
)

print(
    f"F1        : {f1:.4f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    "\nConfusion Matrix:"
)

print(cm)

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)