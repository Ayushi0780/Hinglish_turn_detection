"""
CPU-friendly training pipeline for tiny audio turn detection.

Main model:
    TinyCNNGRU

Pipeline:
    audio
      -> 2.5 sec trailing window
      -> 64-bin log-mel spectrogram
      -> CNN
      -> GRU
      -> turn-complete probability

Dataset:
    Smart Turn real/synthetic base data
    +
    synthetic Hinglish augmentation

Evaluation:
    validation F1 during training
    validation threshold tuning
    clean real test F1/precision/recall after training
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
)

from tqdm import tqdm

from dataset import (
    load_manifest,
    TurnDataset,
    TurnDatasetRaw,
)

from model import (
    WhisperTurnClassifier,
    TinyCNNGRU,
)


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Class weighting
# ============================================================

def compute_pos_weight(train_df) -> float:

    n_pos = int(
        (train_df["label"] == 1).sum()
    )

    n_neg = int(
        (train_df["label"] == 0).sum()
    )

    if n_pos == 0:
        return 1.0

    return n_neg / n_pos


# ============================================================
# Basic evaluation
# ============================================================

def evaluate_split(model, loader, device, threshold=0.5):

    model.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():

        for x, y in loader:

            x = x.to(device)
            y = y.to(device)

            logits = model(x)

            probs = torch.sigmoid(logits)

            all_probs.extend(
                probs.cpu().numpy()
            )

            all_labels.extend(
                y.cpu().numpy()
            )

    if len(all_labels) == 0:

        return {
            "accuracy": 0.0,
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "confusion_matrix": None,
        }

    all_probs = np.asarray(all_probs)
    all_labels = np.asarray(all_labels)

    all_preds = (
        all_probs >= threshold
    ).astype(float)

    return {
        "accuracy": accuracy_score(
            all_labels,
            all_preds
        ),

        "f1": f1_score(
            all_labels,
            all_preds,
            zero_division=0
        ),

        "precision": precision_score(
            all_labels,
            all_preds,
            zero_division=0
        ),

        "recall": recall_score(
            all_labels,
            all_preds,
            zero_division=0
        ),

        "confusion_matrix": confusion_matrix(
            all_labels,
            all_preds
        ),
    }


# ============================================================
# Get probabilities
# ============================================================

def get_probabilities(model, loader, device):

    model.eval()

    all_probs = []
    all_labels = []

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
        np.asarray(all_probs),
        np.asarray(all_labels)
    )


# ============================================================
# Find best threshold using validation data
# ============================================================

def find_best_threshold(model, val_loader, device):

    probs, labels = get_probabilities(
        model,
        val_loader,
        device
    )

    best_threshold = 0.5
    best_f1 = -1.0

    # Try thresholds from 0.10 to 0.90
    # This avoids extreme thresholds and gives
    # enough resolution for this small dataset.
    thresholds = np.arange(
        0.10,
        0.91,
        0.01
    )

    for threshold in thresholds:

        preds = (
            probs >= threshold
        ).astype(float)

        score = f1_score(
            labels,
            preds,
            zero_division=0
        )

        if score > best_f1:

            best_f1 = score
            best_threshold = float(
                threshold
            )

    return best_threshold, best_f1


# ============================================================
# Training loop
# ============================================================

def run_training_loop(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs,
    patience,
    checkpoint_path,
    tag,
):

    best_f1 = -1.0
    epochs_without_improvement = 0

    for epoch in range(epochs):

        model.train()

        total_loss = 0.0

        progress = tqdm(
            train_loader,
            desc=f"[{tag}] "
                 f"Epoch {epoch + 1}/{epochs}"
        )

        for x, y in progress:

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            logits = model(x)

            loss = criterion(
                logits,
                y
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        train_loss = (
            total_loss /
            max(len(train_loader), 1)
        )

        # ----------------------------------------------------
        # Validation at default threshold 0.5
        # ----------------------------------------------------

        val_metrics = evaluate_split(
            model,
            val_loader,
            device,
            threshold=0.5
        )

        print(
            f"\nEpoch {epoch + 1}/{epochs}"
        )

        print(
            f"Train loss : {train_loss:.4f}"
        )

        print(
            f"Val accuracy : "
            f"{val_metrics['accuracy']:.4f}"
        )

        print(
            f"Val F1       : "
            f"{val_metrics['f1']:.4f}"
        )

        print(
            f"Val precision: "
            f"{val_metrics['precision']:.4f}"
        )

        print(
            f"Val recall   : "
            f"{val_metrics['recall']:.4f}"
        )

        # ----------------------------------------------------
        # Early stopping is still based on validation F1
        # ----------------------------------------------------

        if (
            val_metrics["f1"]
            > best_f1
        ):

            best_f1 = (
                val_metrics["f1"]
            )

            epochs_without_improvement = 0

            torch.save(
                model.state_dict(),
                checkpoint_path
            )

            print(
                f"✓ Saved best model "
                f"(val F1={best_f1:.4f})"
            )

        else:

            epochs_without_improvement += 1

            print(
                f"No improvement "
                f"({epochs_without_improvement}/"
                f"{patience})"
            )

            if (
                epochs_without_improvement
                >= patience
            ):

                print(
                    "\nEarly stopping."
                )

                break

    return best_f1


# ============================================================
# Tiny CNN-GRU training
# ============================================================

def train_tiny(
    args,
    device
):

    print("\n" + "=" * 60)
    print("TRAINING TINY CNN-GRU")
    print("=" * 60)

    manifest = load_manifest(
        args.manifest,
        args.extra_manifest,
        args.hinglish_oversample
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    train_ds = TurnDatasetRaw(
        manifest_df=manifest,
        split="train",
        window_sec=args.window_sec,
        n_mels=args.n_mels,
        cache_dir=args.cache_dir,
    )

    val_ds = TurnDatasetRaw(
        manifest_df=manifest,
        split="val",
        window_sec=args.window_sec,
        n_mels=args.n_mels,
        cache_dir=args.cache_dir,
    )

    test_ds = TurnDatasetRaw(
        manifest_df=manifest,
        split="test",
        window_sec=args.window_sec,
        n_mels=args.n_mels,
        cache_dir=args.cache_dir,
    )

    print(
        "\nDataset sizes:"
    )

    print(
        f"Train: {len(train_ds)}"
    )

    print(
        f"Val  : {len(val_ds)}"
    )

    print(
        f"Test : {len(test_ds)}"
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = TinyCNNGRU().to(device)

    print(
        f"\nTinyCNNGRU parameters: "
        f"{model.count_params():,}"
    )

    # --------------------------------------------------------
    # Class weighting
    # --------------------------------------------------------

    train_df = manifest[
        manifest["split"] == "train"
    ]

    pos_weight = compute_pos_weight(
        train_df
    )

    print(
        f"Positive class weight: "
        f"{pos_weight:.4f}"
    )

    criterion = (
        nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(
                pos_weight,
                dtype=torch.float32,
                device=device
            )
        )
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    # --------------------------------------------------------
    # Checkpoint
    # --------------------------------------------------------

    os.makedirs(
        args.checkpoint_dir,
        exist_ok=True
    )

    checkpoint_path = os.path.join(
        args.checkpoint_dir,
        "tiny_best.pt"
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_val_f1 = run_training_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=args.epochs,
        patience=args.patience,
        checkpoint_path=checkpoint_path,
        tag="tiny",
    )

    # --------------------------------------------------------
    # Load best checkpoint
    # --------------------------------------------------------

    print(
        "\nLoading best checkpoint..."
    )

    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=device
        )
    )

    # --------------------------------------------------------
    # Find best threshold using VALIDATION ONLY
    # --------------------------------------------------------

    print(
        "\n" + "=" * 60
    )

    print(
        "VALIDATION THRESHOLD TUNING"
    )

    print(
        "=" * 60
    )

    best_threshold, threshold_f1 = (
        find_best_threshold(
            model,
            val_loader,
            device
        )
    )

    print(
        f"Best threshold : "
        f"{best_threshold:.2f}"
    )

    print(
        f"Validation F1 at threshold: "
        f"{threshold_f1:.4f}"
    )

    # --------------------------------------------------------
    # Final test evaluation
    #
    # IMPORTANT:
    # The test set is NOT used to choose the threshold.
    # --------------------------------------------------------

    print(
        "\n" + "=" * 60
    )

    print(
        "FINAL TEST EVALUATION"
    )

    print(
        "=" * 60
    )

    test_metrics = evaluate_split(
        model,
        test_loader,
        device,
        threshold=best_threshold
    )

    print(
        f"Threshold: {best_threshold:.2f}"
    )

    print(
        f"Accuracy : "
        f"{test_metrics['accuracy']:.4f}"
    )

    print(
        f"F1       : "
        f"{test_metrics['f1']:.4f}"
    )

    print(
        f"Precision: "
        f"{test_metrics['precision']:.4f}"
    )

    print(
        f"Recall   : "
        f"{test_metrics['recall']:.4f}"
    )

    print(
        "\nConfusion Matrix:"
    )

    print(
        test_metrics["confusion_matrix"]
    )

    print(
        f"\nBest validation F1: "
        f"{best_val_f1:.4f}"
    )

    print(
        f"Model saved to: "
        f"{checkpoint_path}"
    )


# ============================================================
# Optional Whisper training
# ============================================================

def train_whisper(
    args,
    device
):

    print(
        "\nWARNING:"
    )

    print(
        "Whisper training is not recommended "
        "for the CPU-first solution."
    )

    from transformers import (
        WhisperFeatureExtractor
    )

    fe = (
        WhisperFeatureExtractor
        .from_pretrained(
            "openai/whisper-tiny"
        )
    )

    manifest = load_manifest(
        args.manifest,
        args.extra_manifest,
        args.hinglish_oversample
    )

    train_ds = TurnDataset(
        manifest,
        "train",
        fe
    )

    val_ds = TurnDataset(
        manifest,
        "val",
        fe
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = WhisperTurnClassifier(
        freeze_encoder=args.freeze_encoder,
        unfreeze_last_n=args.unfreeze_last_n,
    ).to(device)

    optimizer = torch.optim.AdamW(
        filter(
            lambda p: p.requires_grad,
            model.parameters()
        ),
        lr=args.lr
    )

    train_df = manifest[
        manifest["split"] == "train"
    ]

    pos_weight = compute_pos_weight(
        train_df
    )

    criterion = (
        nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(
                pos_weight,
                device=device
            )
        )
    )

    os.makedirs(
        args.checkpoint_dir,
        exist_ok=True
    )

    checkpoint_path = os.path.join(
        args.checkpoint_dir,
        "whisper_best.pt"
    )

    run_training_loop(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        args.epochs,
        args.patience,
        checkpoint_path,
        "whisper"
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=[
            "tiny",
            "whisper"
        ],
        default="tiny"
    )

    parser.add_argument(
        "--manifest",
        default="data/processed_test/manifest.csv"
    )

    parser.add_argument(
        "--extra_manifest",
        default="hinglish_data/manifest.csv"
    )

    parser.add_argument(
        "--hinglish_oversample",
        type=int,
        default=1
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=15
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=4
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3
    )

    parser.add_argument(
        "--window_sec",
        type=float,
        default=2.5
    )

    parser.add_argument(
        "--n_mels",
        type=int,
        default=64
    )

    parser.add_argument(
        "--checkpoint_dir",
        default="checkpoints"
    )

    parser.add_argument(
        "--cache_dir",
        default="cache/mel"
    )

    parser.add_argument(
        "--freeze_encoder",
        action="store_true"
    )

    parser.add_argument(
        "--unfreeze_last_n",
        type=int,
        default=0
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    set_seed(
        args.seed
    )

    # --------------------------------------------------------
    # Cache handling
    # --------------------------------------------------------

    if args.cache_dir == "":
        args.cache_dir = None

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\nUsing device: {device}"
    )

    print(
        f"PyTorch version: "
        f"{torch.__version__}"
    )

    # --------------------------------------------------------
    # Model selection
    # --------------------------------------------------------

    if args.model == "tiny":

        train_tiny(
            args,
            device
        )

    else:

        train_whisper(
            args,
            device
        )


if __name__ == "__main__":
    main()