
import os
import torch
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
from transformers import WhisperFeatureExtractor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

from dataset import TurnDataset
from model import WhisperTurnClassifier


MANIFEST = "data/processed_test/manifest.csv"
CHECKPOINT = "checkpoints/whisper_best.pt"
BATCH_SIZE = 8


def get_predictions(model, loader, device):
    model.eval()

    all_labels = []
    all_probs = []

    with torch.no_grad():

        for x, y in loader:

            x = x.to(device)

            logits = model(x)

            probs = torch.sigmoid(logits)

            all_probs.extend(
                probs.cpu().numpy()
            )

            all_labels.extend(
                y.numpy()
            )

    return (
        np.array(all_labels).astype(int),
        np.array(all_probs),
    )


def metrics_at_threshold(labels, probs, threshold):

    preds = (
        probs >= threshold
    ).astype(int)

    return {
        "accuracy": accuracy_score(
            labels,
            preds
        ),
        "f1": f1_score(
            labels,
            preds,
            zero_division=0
        ),
        "precision": precision_score(
            labels,
            preds,
            zero_division=0
        ),
        "recall": recall_score(
            labels,
            preds,
            zero_division=0
        ),
        "cm": confusion_matrix(
            labels,
            preds
        ),
    }


def main():

    print("=" * 60)
    print("WHISPER THRESHOLD TUNING + TEST EVALUATION")
    print("=" * 60)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Using device: {device}"
    )

    # ---------------------------------------------------------
    # LOAD MANIFEST
    # ---------------------------------------------------------

    manifest = pd.read_csv(
        MANIFEST
    )

    # ---------------------------------------------------------
    # VALIDATION SET
    # ---------------------------------------------------------

    val_df = manifest[
        manifest["split"] == "val"
    ].copy()

    # Validation may contain synthetic Hinglish.
    # We keep it because this is the validation set used
    # during model selection.
    print(
        f"\nValidation samples: {len(val_df)}"
    )

    # ---------------------------------------------------------
    # REAL TEST SET
    # ---------------------------------------------------------

    test_df = manifest[
        manifest["split"] == "test"
    ].copy()

    if "synthetic" in test_df.columns:

        before = len(test_df)

        test_df = test_df[
            test_df["synthetic"] == False
        ].copy()

        print(
            f"Removed {before - len(test_df)} "
            "synthetic test samples."
        )

    print(
        f"Real test samples: {len(test_df)}"
    )

    print("\nTest distribution:")

    print(
        test_df.groupby(
            ["language", "label"]
        ).size()
    )

    # ---------------------------------------------------------
    # FEATURE EXTRACTOR
    # ---------------------------------------------------------

    print(
        "\nLoading Whisper feature extractor..."
    )

    feature_extractor = (
        WhisperFeatureExtractor.from_pretrained(
            "openai/whisper-tiny"
        )
    )

    # ---------------------------------------------------------
    # DATASETS
    # ---------------------------------------------------------

    val_dataset = TurnDataset(
        val_df.assign(split="val"),
        "val",
        feature_extractor,
    )

    test_dataset = TurnDataset(
        test_df.assign(split="test"),
        "test",
        feature_extractor,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # ---------------------------------------------------------
    # MODEL
    # ---------------------------------------------------------

    print(
        "\nLoading Whisper model..."
    )

    model = WhisperTurnClassifier(
        freeze_encoder=True,
        unfreeze_last_n=0,
    ).to(device)

    if not os.path.exists(
        CHECKPOINT
    ):
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT}"
        )

    state_dict = torch.load(
        CHECKPOINT,
        map_location=device,
    )

    model.load_state_dict(
        state_dict
    )

    model.eval()

    print(
        f"Loaded checkpoint: {CHECKPOINT}"
    )

    # ---------------------------------------------------------
    # VALIDATION PREDICTIONS
    # ---------------------------------------------------------

    print(
        "\nRunning validation inference..."
    )

    val_labels, val_probs = get_predictions(
        model,
        val_loader,
        device,
    )

    # ---------------------------------------------------------
    # THRESHOLD SEARCH
    # ---------------------------------------------------------

    print()
    print("=" * 60)
    print("VALIDATION THRESHOLD TUNING")
    print("=" * 60)

    best_threshold = 0.50
    best_f1 = -1

    for threshold in np.arange(
        0.30,
        0.71,
        0.01
    ):

        result = metrics_at_threshold(
            val_labels,
            val_probs,
            threshold,
        )

        if result["f1"] > best_f1:

            best_f1 = result["f1"]
            best_threshold = threshold

    best_threshold = round(
        float(best_threshold),
        2
    )

    best_val = metrics_at_threshold(
        val_labels,
        val_probs,
        best_threshold,
    )

    print(
        f"Best threshold : {best_threshold:.2f}"
    )

    print(
        f"Validation accuracy : "
        f"{best_val['accuracy']:.4f}"
    )

    print(
        f"Validation F1       : "
        f"{best_val['f1']:.4f}"
    )

    print(
        f"Validation precision : "
        f"{best_val['precision']:.4f}"
    )

    print(
        f"Validation recall    : "
        f"{best_val['recall']:.4f}"
    )

    # ---------------------------------------------------------
    # TEST PREDICTIONS
    # ---------------------------------------------------------

    print()
    print(
        "Running REAL TEST inference..."
    )

    test_labels, test_probs = get_predictions(
        model,
        test_loader,
        device,
    )

    # ---------------------------------------------------------
    # DEFAULT 0.50
    # ---------------------------------------------------------

    default_result = metrics_at_threshold(
        test_labels,
        test_probs,
        0.50,
    )

    # ---------------------------------------------------------
    # TUNED THRESHOLD
    # ---------------------------------------------------------

    tuned_result = metrics_at_threshold(
        test_labels,
        test_probs,
        best_threshold,
    )

    # ---------------------------------------------------------
    # RESULTS
    # ---------------------------------------------------------

    print()
    print("=" * 60)
    print("REAL TEST — DEFAULT THRESHOLD 0.50")
    print("=" * 60)

    print(
        f"Accuracy  : "
        f"{default_result['accuracy']:.4f}"
    )

    print(
        f"F1        : "
        f"{default_result['f1']:.4f}"
    )

    print(
        f"Precision : "
        f"{default_result['precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{default_result['recall']:.4f}"
    )

    print(
        "\nConfusion Matrix:"
    )

    print(
        default_result["cm"]
    )

    print()
    print("=" * 60)
    print(
        f"REAL TEST — TUNED THRESHOLD {best_threshold:.2f}"
    )
    print("=" * 60)

    print(
        f"Accuracy  : "
        f"{tuned_result['accuracy']:.4f}"
    )

    print(
        f"F1        : "
        f"{tuned_result['f1']:.4f}"
    )

    print(
        f"Precision : "
        f"{tuned_result['precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{tuned_result['recall']:.4f}"
    )

    print(
        "\nConfusion Matrix:"
    )

    print(
        tuned_result["cm"]
    )

    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------

    print()
    print("=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    print(
        f"Default threshold : 0.50"
    )

    print(
        f"Default F1        : "
        f"{default_result['f1']:.4f}"
    )

    print(
        f"Tuned threshold   : "
        f"{best_threshold:.2f}"
    )

    print(
        f"Tuned F1          : "
        f"{tuned_result['f1']:.4f}"
    )

    print()
    print("Done.")


if __name__ == "__main__":
    main()

