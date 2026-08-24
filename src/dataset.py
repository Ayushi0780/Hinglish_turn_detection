"""
Dataset classes for turn detection.

TurnDataset:
    raw waveform -> WhisperFeatureExtractor log-mel features.

TurnDatasetRaw:
    raw waveform -> fixed-length log-mel spectrogram for the lightweight
    CNN-GRU model.

Dataset strategy:
    - Smart Turn manifest provides the base dataset.
    - Hinglish manifest can be added as targeted augmentation.
    - Hinglish train samples are used only for training.
    - Hinglish validation samples are used only for validation.
    - Synthetic samples are NEVER allowed in the final test split.
    - Test data comes only from the real Smart Turn manifest.
"""

import os
import hashlib
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import torch
from torch.utils.data import Dataset

SAMPLE_RATE = 16000


def load_manifest(
    manifest_path: str,
    extra_manifest: str = None,
    oversample: int = 1
) -> pd.DataFrame:
    """
    Load the base Smart Turn manifest and optionally add Hinglish data.

    Dataset rules:

    BASE MANIFEST:
        Used as the main Smart Turn dataset.

    EXTRA MANIFEST:
        Expected to be the Hinglish dataset.

        train -> added to train
        val   -> added to val
        test  -> NEVER added

    IMPORTANT:
        The final test set must contain only REAL audio.

        Therefore, any synthetic row belonging to the test split is removed
        from the base manifest as well.
    """

    df = pd.read_csv(manifest_path)

    # ---------------------------------------------------------
    # 1. Clean the base manifest
    # ---------------------------------------------------------

    required_columns = {
        "path",
        "label",
        "language",
        "synthetic",
        "split"
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Manifest is missing required columns: {sorted(missing)}"
        )

    # Make synthetic column robust to CSV bool/string values.
    df["synthetic"] = (
        df["synthetic"]
        .astype(str)
        .str.lower()
        .isin(["true", "1", "yes"])
    )

    # ---------------------------------------------------------
    # 2. NEVER allow synthetic data into the test set
    # ---------------------------------------------------------

    before = len(df)

    synthetic_test = (
        (df["split"] == "test") &
        (df["synthetic"] == True)
    )

    n_removed = int(synthetic_test.sum())

    if n_removed > 0:
        print(
            f"[load_manifest] Removing {n_removed} synthetic "
            f"samples from TEST."
        )

        df = df.loc[~synthetic_test].copy()

    # ---------------------------------------------------------
    # 3. Add Hinglish dataset
    # ---------------------------------------------------------

    if extra_manifest:

        extra = pd.read_csv(extra_manifest)

        missing_extra = required_columns - set(extra.columns)

        if missing_extra:
            raise ValueError(
                f"Extra manifest is missing required columns: "
                f"{sorted(missing_extra)}"
            )

        extra["synthetic"] = (
            extra["synthetic"]
            .astype(str)
            .str.lower()
            .isin(["true", "1", "yes"])
        )

        # -----------------------------------------------------
        # Hinglish TRAIN
        # -----------------------------------------------------

        extra_train = extra[
            extra["split"] == "train"
        ].copy()

        # Optional oversampling of Hinglish training data.
        if oversample > 1 and len(extra_train) > 0:
            extra_train = pd.concat(
                [extra_train] * oversample,
                ignore_index=True
            )

        # -----------------------------------------------------
        # Hinglish VALIDATION
        # -----------------------------------------------------

        extra_val = extra[
            extra["split"] == "val"
        ].copy()

        # -----------------------------------------------------
        # NEVER add Hinglish TEST
        # -----------------------------------------------------

        extra_test_count = int(
            (extra["split"] == "test").sum()
        )

        if extra_test_count > 0:
            print(
                f"[load_manifest] Ignoring {extra_test_count} "
                f"extra-manifest TEST samples."
            )

        # Add Hinglish train + validation.
        if len(extra_train) > 0:
            df = pd.concat(
                [df, extra_train],
                ignore_index=True
            )

        if len(extra_val) > 0:
            df = pd.concat(
                [df, extra_val],
                ignore_index=True
            )

        print(
            f"[load_manifest] Added {len(extra_train)} "
            f"Hinglish TRAIN samples."
        )

        print(
            f"[load_manifest] Added {len(extra_val)} "
            f"Hinglish VALIDATION samples."
        )

    # ---------------------------------------------------------
    # 4. Final safety check
    # ---------------------------------------------------------

    synthetic_test_after = (
        (df["split"] == "test") &
        (df["synthetic"] == True)
    ).sum()

    if synthetic_test_after > 0:
        raise RuntimeError(
            "Synthetic samples are present in the final TEST set."
        )

    df = df.reset_index(drop=True)

    # ---------------------------------------------------------
    # 5. Print useful dataset statistics
    # ---------------------------------------------------------

    print("\nDataset composition:")
    print("----------------------------------------")

    print(
        df.groupby(
            ["split", "synthetic"]
        ).size()
    )

    print("\nLanguage × label:")
    print("----------------------------------------")

    print(
        df.groupby(
            ["split", "language", "label"]
        ).size()
    )

    print("----------------------------------------\n")

    return df


class TurnDataset(Dataset):
    """
    Dataset for the Whisper-tiny backbone.

    Audio -> WhisperFeatureExtractor -> log-mel features.
    """

    def __init__(
        self,
        manifest_df: pd.DataFrame,
        split: str,
        feature_extractor
    ):

        self.df = (
            manifest_df[
                manifest_df["split"] == split
            ]
            .reset_index(drop=True)
        )

        self.fe = feature_extractor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        audio, sr = sf.read(row["path"])

        # Convert stereo -> mono if necessary.
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        audio = audio.astype(np.float32)

        if sr != SAMPLE_RATE:
            audio = librosa.resample(
                audio,
                orig_sr=sr,
                target_sr=SAMPLE_RATE
            )

        features = self.fe(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt"
        )

        input_features = features[
            "input_features"
        ].squeeze(0)

        label = torch.tensor(
            row["label"],
            dtype=torch.float32
        )

        return input_features, label


class TurnDatasetRaw(Dataset):
    """
    Lightweight CPU-friendly dataset.

    Audio -> 64-bin log-mel spectrogram -> CNN-GRU.

    No Whisper dependency is required.

    The spectrogram is cached on disk when cache_dir is provided,
    avoiding expensive librosa computation every epoch.
    """

    def __init__(
        self,
        manifest_df: pd.DataFrame,
        split: str,
        window_sec: float = 2.5,
        n_mels: int = 64,
        cache_dir: str = None
    ):

        self.df = (
            manifest_df[
                manifest_df["split"] == split
            ]
            .reset_index(drop=True)
        )

        self.window_sec = window_sec
        self.n_mels = n_mels

        self.target_len = int(
            window_sec * SAMPLE_RATE
        )

        self.cache_dir = cache_dir

        if cache_dir:
            os.makedirs(
                cache_dir,
                exist_ok=True
            )

    def __len__(self):
        return len(self.df)

    def _cache_path(self, audio_path: str) -> str:

        key = (
            f"{audio_path}_"
            f"{self.window_sec}_"
            f"{self.n_mels}"
        )

        h = hashlib.md5(
            key.encode()
        ).hexdigest()

        return os.path.join(
            self.cache_dir,
            f"{h}.npy"
        )

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        label = torch.tensor(
            row["label"],
            dtype=torch.float32
        )

        # -----------------------------------------------------
        # Load cached feature if available
        # -----------------------------------------------------

        if self.cache_dir:

            cpath = self._cache_path(
                row["path"]
            )

            if os.path.exists(cpath):

                log_mel = np.load(
                    cpath
                ).astype(np.float32)

                return (
                    torch.from_numpy(log_mel),
                    label
                )

        # -----------------------------------------------------
        # Load audio
        # -----------------------------------------------------

        audio, sr = sf.read(
            row["path"]
        )

        # Convert stereo -> mono.
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        audio = audio.astype(
            np.float32
        )

        # -----------------------------------------------------
        # Resample
        # -----------------------------------------------------

        if sr != SAMPLE_RATE:

            audio = librosa.resample(
                audio,
                orig_sr=sr,
                target_sr=SAMPLE_RATE
            )

        # -----------------------------------------------------
        # Fixed 2.5 second trailing window
        # -----------------------------------------------------

        if len(audio) < self.target_len:

            audio = np.pad(
                audio,
                (
                    self.target_len - len(audio),
                    0
                )
            )

        else:

            audio = audio[
                -self.target_len:
            ]

        # -----------------------------------------------------
        # Log-mel spectrogram
        # -----------------------------------------------------

        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=SAMPLE_RATE,
            n_mels=self.n_mels,
            hop_length=160,
            n_fft=400
        )

        log_mel = librosa.power_to_db(
            mel,
            ref=np.max
        ).astype(np.float32)

        # -----------------------------------------------------
        # Per-example normalization
        # -----------------------------------------------------

        log_mel = (
            log_mel - log_mel.mean()
        ) / (
            log_mel.std() + 1e-6
        )

        # -----------------------------------------------------
        # Cache
        # -----------------------------------------------------

        if self.cache_dir:

            np.save(
                self._cache_path(
                    row["path"]
                ),
                log_mel
            )

        return (
            torch.from_numpy(log_mel),
            label
        )