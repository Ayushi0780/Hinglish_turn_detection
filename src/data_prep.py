# """
# Data preparation for turn detection.

# Dataset:
#     pipecat-ai/smart-turn-data-v3.2-train

# What this script does:
#     1. Streams the Smart Turn dataset from Hugging Face.
#     2. Filters to Hindi (hin) and English (eng).
#     3. Balances:
#            Hindi + complete
#            Hindi + incomplete
#            English + complete
#            English + incomplete
#     4. Splits each group into train / validation / test.
#     5. Keeps the test set 100% real Smart Turn data.
#     6. Resamples audio to 16 kHz.
#     7. Keeps the last 2.5 seconds of every clip.
#     8. Saves individual WAV files.
#     9. Writes manifest.csv.

# IMPORTANT:
#     We use Audio(decode=False), so Hugging Face does not try to decode
#     audio using TorchCodec. Audio is decoded locally with soundfile.

# Default dataset size:

#     TRAIN:
#         Hindi complete       1000
#         Hindi incomplete     1000
#         English complete     1000
#         English incomplete   1000
#         Total                4000

#     VALIDATION:
#         125 per group
#         Total = 500

#     TEST:
#         125 per group
#         Total = 500

#     TOTAL REAL DATA = 5000

# Quick test:

#     python src/data_prep.py --out_dir data/processed_test \
#         --train_per_group 100 \
#         --val_per_group 20 \
#         --test_per_group 20

# Full run:

#     python src/data_prep.py --out_dir data/processed

# Requirements:

#     pip install datasets soundfile librosa pandas numpy tqdm
# """

# import argparse
# import io
# import os
# from collections import defaultdict

# # Hugging Face can occasionally need longer than the default read timeout
# # when downloading Parquet shards. These settings do not change model/data
# # behavior; they only make interrupted/slow downloads less likely to fail.
# os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
# os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")

# import numpy as np
# import pandas as pd
# import soundfile as sf
# import librosa
# from tqdm import tqdm


# SAMPLE_RATE = 16000

# DATASET_NAME = "pipecat-ai/smart-turn-data-v3.2-train"

# GROUPS = [
#     ("hin", 1),  # Hindi complete
#     ("hin", 0),  # Hindi incomplete
#     ("eng", 1),  # English complete
#     ("eng", 0),  # English incomplete
# ]


# # ---------------------------------------------------------------------
# # LOAD DATASET
# # ---------------------------------------------------------------------

# def load_raw_dataset(
#     split="train",
#     languages=None,
#     shuffle_buffer=2000,
# ):
#     """
#     Load Smart Turn using streaming.

#     Audio decoding is explicitly disabled so that Hugging Face does not
#     require TorchCodec.

#     The audio is decoded later using soundfile.
#     """

#     from datasets import load_dataset, Audio

#     print(f"Loading {DATASET_NAME}...")
#     print("Streaming mode: ON")
#     print("Audio automatic decoding: OFF")

#     ds = load_dataset(
#         DATASET_NAME,
#         split=split,
#         streaming=True,
#     )

#     # IMPORTANT:
#     # Prevent Hugging Face from automatically decoding Audio.
#     ds = ds.cast_column(
#         "audio",
#         Audio(decode=False),
#     )

#     if languages:
#         ds = ds.filter(
#             lambda ex: ex.get("language") in languages
#         )

#     # Shuffle the streaming dataset so we don't depend on the original
#     # ordering of the parquet shards.
#     # A small streaming buffer is intentional here. A very large buffer
#     # (e.g. 2000+) can make the first example take a long time to arrive
#     # because Hugging Face must read many examples before yielding data.
#     ds = ds.shuffle(
#         seed=42,
#         buffer_size=shuffle_buffer,
#     )

#     return ds


# # ---------------------------------------------------------------------
# # AUDIO DECODING
# # ---------------------------------------------------------------------

# def decode_audio(audio_data):
#     """
#     Decode an audio example returned with Audio(decode=False).

#     Depending on the Hugging Face dataset representation, audio_data
#     can contain:
#         - bytes
#         - path
#         - a dictionary containing bytes/path

#     Returns:
#         audio: np.ndarray
#         sample_rate: int
#     """

#     if audio_data is None:
#         raise ValueError("Audio field is None.")

#     # Audio(decode=False) normally returns a dictionary.
#     if isinstance(audio_data, dict):

#         audio_bytes = audio_data.get("bytes")
#         audio_path = audio_data.get("path")

#         # Case 1: audio bytes are available.
#         if audio_bytes is not None:

#             if isinstance(audio_bytes, memoryview):
#                 audio_bytes = audio_bytes.tobytes()

#             elif isinstance(audio_bytes, bytearray):
#                 audio_bytes = bytes(audio_bytes)

#             audio, sr = sf.read(
#                 io.BytesIO(audio_bytes),
#                 dtype="float32",
#             )

#             return audio, sr

#         # Case 2: a local/path reference is available.
#         if audio_path:

#             audio, sr = sf.read(
#                 audio_path,
#                 dtype="float32",
#             )

#             return audio, sr

#         raise ValueError(
#             "Audio dictionary contains neither "
#             "'bytes' nor 'path'."
#         )

#     # Sometimes the value may directly be bytes.
#     if isinstance(
#         audio_data,
#         (bytes, bytearray, memoryview),
#     ):

#         if isinstance(audio_data, memoryview):
#             audio_data = audio_data.tobytes()

#         elif isinstance(audio_data, bytearray):
#             audio_data = bytes(audio_data)

#         audio, sr = sf.read(
#             io.BytesIO(audio_data),
#             dtype="float32",
#         )

#         return audio, sr

#     raise TypeError(
#         f"Unsupported audio type: {type(audio_data)}"
#     )


# # ---------------------------------------------------------------------
# # AUDIO PREPROCESSING
# # ---------------------------------------------------------------------

# def convert_to_mono(audio):
#     """Convert stereo/multi-channel audio to mono."""

#     if audio.ndim == 1:
#         return audio

#     return np.mean(
#         audio,
#         axis=1,
#     ).astype(np.float32)


# def trim_to_trailing_window(
#     audio,
#     sr,
#     window_sec,
# ):
#     """
#     Keep only the last `window_sec` seconds.

#     If the clip is shorter than the required window, left-pad it with
#     silence.

#     Example:

#         clip = 1.5 seconds
#         window = 2.5 seconds

#         [silence][1.5 sec audio]
#         <------ 2.5 sec ------>
#     """

#     target_len = int(
#         window_sec * sr
#     )

#     if len(audio) >= target_len:
#         return audio[-target_len:]

#     pad_len = (
#         target_len - len(audio)
#     )

#     padding = np.zeros(
#         pad_len,
#         dtype=np.float32,
#     )

#     return np.concatenate(
#         [padding, audio]
#     )


# def preprocess_audio(
#     audio,
#     sr,
#     window_sec,
# ):
#     """
#     Convert audio to:
#         mono
#         16 kHz
#         fixed 2.5-second trailing window
#     """

#     audio = np.asarray(
#         audio,
#         dtype=np.float32,
#     )

#     audio = convert_to_mono(
#         audio
#     )

#     if sr != SAMPLE_RATE:

#         audio = librosa.resample(
#             audio,
#             orig_sr=sr,
#             target_sr=SAMPLE_RATE,
#         )

#         sr = SAMPLE_RATE

#     audio = trim_to_trailing_window(
#         audio,
#         sr,
#         window_sec,
#     )

#     # Avoid NaN/Inf values entering the model.
#     audio = np.nan_to_num(
#         audio,
#         nan=0.0,
#         posinf=0.0,
#         neginf=0.0,
#     )

#     return audio.astype(
#         np.float32
#     )


# # ---------------------------------------------------------------------
# # BALANCED SAMPLE COLLECTION
# # ---------------------------------------------------------------------

# def collect_balanced_samples(
#     ds,
#     train_per_group,
#     val_per_group,
#     test_per_group,
# ):
#     """
#     Collect equal numbers from:

#         Hindi + complete
#         Hindi + incomplete
#         English + complete
#         English + incomplete

#     Each group is independently split into train/validation/test.

#     This prevents random splitting from creating an imbalanced dataset.
#     """

#     total_per_group = (
#         train_per_group
#         + val_per_group
#         + test_per_group
#     )

#     buckets = defaultdict(list)

#     print()
#     print("=" * 60)
#     print("COLLECTING BALANCED REAL DATA")
#     print("=" * 60)

#     print(
#         f"Need {total_per_group} examples "
#         "per language/label group."
#     )

#     for ex in tqdm(
#         ds,
#         desc="streaming Smart Turn",
#     ):

#         language = ex.get(
#             "language"
#         )

#         endpoint = ex.get(
#             "endpoint_bool"
#         )
#         synthetic = bool(ex.get("synthetic") or False)

#         if language not in {
#             "hin",
#             "eng",
#         }:
#             continue

#         if endpoint is None:
#             continue

#         if synthetic:
#             continue

#         label = int(endpoint)

#         group = (
#             language,
#             label,
#         )

#         if group not in GROUPS:
#             continue

#         if len(
#             buckets[group]
#         ) >= total_per_group:
#             continue

#         buckets[group].append(
#             ex
#         )

#         # Stop as soon as all four groups
#         # contain enough examples.
#         if all(
#             len(
#                 buckets[group]
#             ) >= total_per_group
#             for group in GROUPS
#         ):
#             break

#     print()
#     print("Collected:")

#     for language, label in GROUPS:

#         count = len(
#             buckets[
#                 (language, label)
#             ]
#         )

#         print(
#             f"  {language} | "
#             f"label={label}: "
#             f"{count}"
#         )

#         if count < total_per_group:

#             raise RuntimeError(
#                 f"Not enough data for "
#                 f"{language}, label={label}. "
#                 f"Required={total_per_group}, "
#                 f"found={count}."
#             )

#     # Deterministic shuffle.
#     rng = np.random.default_rng(
#         42
#     )

#     train_rows = []
#     val_rows = []
#     test_rows = []

#     for group in GROUPS:

#         samples = list(
#             buckets[group]
#         )

#         rng.shuffle(
#             samples
#         )

#         train_end = (
#             train_per_group
#         )

#         val_end = (
#             train_per_group
#             + val_per_group
#         )

#         train_rows.extend(
#             samples[
#                 :train_end
#             ]
#         )

#         val_rows.extend(
#             samples[
#                 train_end:val_end
#             ]
#         )

#         test_rows.extend(
#             samples[
#                 val_end:
#             ]
#         )

#     rng.shuffle(
#         train_rows
#     )

#     rng.shuffle(
#         val_rows
#     )

#     rng.shuffle(
#         test_rows
#     )

#     return (
#         train_rows,
#         val_rows,
#         test_rows,
#     )


# # ---------------------------------------------------------------------
# # PROCESS AUDIO
# # ---------------------------------------------------------------------

# def process_examples(
#     examples,
#     out_dir,
#     window_sec,
#     split_name,
# ):
#     """
#     Decode, preprocess and save examples as WAV files.
#     """

#     audio_dir = os.path.join(
#         out_dir,
#         "audio",
#         split_name,
#     )

#     os.makedirs(
#         audio_dir,
#         exist_ok=True,
#     )

#     rows = []

#     for i, ex in enumerate(
#         tqdm(
#             examples,
#             desc=f"processing {split_name}",
#         )
#     ):

#         try:

#             audio, sr = decode_audio(
#                 ex["audio"]
#             )

#             audio = preprocess_audio(
#                 audio,
#                 sr,
#                 window_sec,
#             )

#         except Exception as e:

#             print(
#                 f"\nWARNING: Could not decode "
#                 f"sample {i}: {e}"
#             )

#             continue

#         label = int(
#             ex["endpoint_bool"]
#         )

#         language = ex.get(
#             "language",
#             "unk",
#         )

#         midfiller = bool(
#             ex.get(
#                 "midfiller"
#             ) or False
#         )

#         endfiller = bool(
#             ex.get(
#                 "endfiller"
#             ) or False
#         )

#         synthetic = bool(
#             ex.get(
#                 "synthetic"
#             ) or False
#         )

#         source = ex.get(
#             "dataset",
#             "smart_turn",
#         )

#         original_id = ex.get(
#             "id",
#             f"{split_name}_{i}",
#         )

#         filename = (
#             f"{split_name}_{i:06d}.wav"
#         )

#         filepath = os.path.join(
#             audio_dir,
#             filename,
#         )

#         sf.write(
#             filepath,
#             audio,
#             SAMPLE_RATE,
#         )

#         rows.append(
#             {
#                 "path": filepath,
#                 "label": label,
#                 "language": language,
#                 "midfiller": midfiller,
#                 "endfiller": endfiller,
#                 "synthetic": synthetic,
#                 "source": source,
#                 "original_id": original_id,
#                 "split": split_name,
#             }
#         )

#     return pd.DataFrame(
#         rows
#     )


# # ---------------------------------------------------------------------
# # DATASET SUMMARY
# # ---------------------------------------------------------------------

# def print_dataset_summary(
#     manifest,
# ):
#     """Print useful sanity checks."""

#     print()
#     print("=" * 60)
#     print("DATASET SUMMARY")
#     print("=" * 60)

#     print(
#         f"\nTotal samples: "
#         f"{len(manifest)}"
#     )

#     print(
#         "\nSplit distribution:"
#     )

#     print(
#         manifest[
#             "split"
#         ].value_counts()
#         .sort_index()
#     )

#     print(
#         "\nLanguage × label distribution:"
#     )

#     distribution = pd.crosstab(
#         [
#             manifest["split"],
#             manifest["language"],
#         ],
#         manifest["label"],
#     )

#     print(
#         distribution
#     )

#     print(
#         "\nSynthetic samples:"
#     )

#     print(
#         manifest.groupby(
#             "split"
#         )["synthetic"].sum()
#     )

#     # Test must be real.
#     test_df = manifest[
#         manifest["split"] == "test"
#     ]

#     if len(test_df) > 0:

#         if test_df[
#             "synthetic"
#         ].any():

#             raise RuntimeError(
#                 "ERROR: Synthetic data "
#                 "was found in the test set."
#             )

#     print(
#         "\n✓ Test set contains only real "
#         "Smart Turn data."
#     )


# # ---------------------------------------------------------------------
# # MAIN
# # ---------------------------------------------------------------------

# def main():

#     parser = argparse.ArgumentParser()

#     parser.add_argument(
#         "--out_dir",
#         default="data/processed",
#     )

#     parser.add_argument(
#         "--window_sec",
#         type=float,
#         default=2.5,
#     )

#     parser.add_argument(
#         "--languages",
#         nargs="*",
#         default=[
#             "hin",
#             "eng",
#         ],
#         help=(
#             "Languages to use. "
#             "Default: hin eng."
#         ),
#     )

#     parser.add_argument(
#         "--train_per_group",
#         type=int,
#         default=1000,
#     )

#     parser.add_argument(
#         "--val_per_group",
#         type=int,
#         default=125,
#     )

#     parser.add_argument(
#         "--test_per_group",
#         type=int,
#         default=125,
#     )

#     parser.add_argument(
#         "--shuffle_buffer",
#         type=int,
#         default=200,
#         help=(
#             "Streaming shuffle buffer. 200 is recommended for a CPU/time-"
#             "constrained run; increase only if the connection is fast."
#         ),
#     )

#     args = parser.parse_args()

#     if any(
#         value < 0
#         for value in (
#             args.train_per_group,
#             args.val_per_group,
#             args.test_per_group,
#             args.shuffle_buffer,
#         )
#     ):
#         parser.error("train/val/test counts and shuffle_buffer must be >= 0")

#     if args.shuffle_buffer == 0:
#         parser.error("shuffle_buffer must be > 0")

#     os.makedirs(
#         args.out_dir,
#         exist_ok=True,
#     )

#     total_train = (
#         args.train_per_group * 4
#     )

#     total_val = (
#         args.val_per_group * 4
#     )

#     total_test = (
#         args.test_per_group * 4
#     )

#     total = (
#         total_train
#         + total_val
#         + total_test
#     )

#     print("=" * 60)
#     print("SMART TURN DATA PREPARATION")
#     print("=" * 60)

#     print(
#         f"\nLanguages: "
#         f"{args.languages}"
#     )

#     print(
#         f"Audio window: "
#         f"{args.window_sec} sec"
#     )

#     print(
#         "\nExpected dataset:"
#     )

#     print(
#         f"  Train: "
#         f"{total_train}"
#     )

#     print(
#         f"  Validation: "
#         f"{total_val}"
#     )

#     print(
#         f"  Test: "
#         f"{total_test}"
#     )

#     print(
#         f"  Total: "
#         f"{total}"
#     )

#     # -------------------------------------------------------------
#     # LOAD DATA
#     # -------------------------------------------------------------

#     ds = load_raw_dataset(
#         split="train",
#         languages=args.languages,
#         shuffle_buffer=args.shuffle_buffer,
#     )

#     # -------------------------------------------------------------
#     # COLLECT BALANCED DATA
#     # -------------------------------------------------------------

#     print()
#     print("Note: Smart Turn is a large multi-shard dataset.")
#     print("The collector downloads only enough streamed examples to fill")
#     print("the four requested REAL Hindi/English groups; it does not")
#     print("download the complete dataset intentionally.")
#     print("Synthetic Smart Turn records are excluded before collection.")
#     print()

#     (
#         train_rows,
#         val_rows,
#         test_rows,
#     ) = collect_balanced_samples(
#         ds,
#         train_per_group=args.train_per_group,
#         val_per_group=args.val_per_group,
#         test_per_group=args.test_per_group,
#     )

#     # -------------------------------------------------------------
#     # PROCESS AUDIO
#     # -------------------------------------------------------------

#     train_df = process_examples(
#         train_rows,
#         args.out_dir,
#         args.window_sec,
#         "train",
#     )

#     val_df = process_examples(
#         val_rows,
#         args.out_dir,
#         args.window_sec,
#         "val",
#     )

#     test_df = process_examples(
#         test_rows,
#         args.out_dir,
#         args.window_sec,
#         "test",
#     )

#     # -------------------------------------------------------------
#     # MANIFEST
#     # -------------------------------------------------------------

#     manifest = pd.concat(
#         [
#             train_df,
#             val_df,
#             test_df,
#         ],
#         ignore_index=True,
#     )

#     manifest_path = os.path.join(
#         args.out_dir,
#         "manifest.csv",
#     )

#     manifest.to_csv(
#         manifest_path,
#         index=False,
#     )

#     # -------------------------------------------------------------
#     # SUMMARY
#     # -------------------------------------------------------------

#     print_dataset_summary(
#         manifest
#     )

#     print()
#     print(
#         f"✓ Manifest saved to:"
#     )
#     print(
#         f"  {manifest_path}"
#     )

#     print()
#     print(
#         f"✓ Audio saved to:"
#     )
#     print(
#         f"  {os.path.join(args.out_dir, 'audio')}"
#     )

#     print()
#     print(
#         "Synthetic Hinglish data is NOT included "
#         "in this real-data preprocessing step."
#     )

#     print(
#         "It should only be added to train/validation "
#         "through the existing extra_manifest mechanism."
#     )


# if __name__ == "__main__":
#     main()


############################# for hindi

"""
Real Hindi data preparation for turn detection.

Dataset:
    pipecat-ai/smart-turn-data-v3.2-train

Purpose:
    Collect additional REAL Hindi Smart Turn samples so we can improve
    the real-data performance of the TinyCNNGRU turn detector.

What this script does:
    1. Streams Smart Turn from Hugging Face.
    2. Keeps Hindi (hin) only.
    3. Removes synthetic samples.
    4. Balances Hindi label 0 and label 1.
    5. Creates train and validation splits.
    6. Does NOT create a new test set.
    7. Resamples audio to 16 kHz.
    8. Keeps the trailing 2.5 seconds.
    9. Saves WAV files and manifest.csv.

IMPORTANT:
    Existing data/processed_test is NOT touched.

Default:
    Train:
        Hindi label 0 = 100
        Hindi label 1 = 100

    Validation:
        Hindi label 0 = 30
        Hindi label 1 = 30

Output:
    data/processed_hindi/

Run:
    python src/data_prep.py

Or a smaller test first:
    python src/data_prep.py \
        --out_dir data/processed_hindi_test \
        --train_per_label 20 \
        --val_per_label 10

Requirements:
    pip install datasets soundfile librosa pandas numpy tqdm
"""

import argparse
import io
import os
from collections import defaultdict

# Hugging Face download settings.
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from tqdm import tqdm


SAMPLE_RATE = 16000
DATASET_NAME = "pipecat-ai/smart-turn-data-v3.2-train"

# We only want real Hindi.
LANGUAGE = "hin"

# Turn-detection groups.
LABELS = [0, 1]


# ---------------------------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------------------------

def load_raw_dataset(shuffle_buffer=200):
    """
    Load Smart Turn in streaming mode.

    Only Hindi is collected later during filtering.

    Audio decoding is disabled because we decode locally using soundfile.
    """

    from datasets import load_dataset, Audio

    print(f"Loading {DATASET_NAME}...")
    print("Streaming mode: ON")
    print("Audio automatic decoding: OFF")
    print("Target language: Hindi (hin)")
    print("Synthetic samples: EXCLUDED")

    ds = load_dataset(
        DATASET_NAME,
        split="train",
        streaming=True,
    )

    # Prevent Hugging Face from automatically decoding audio.
    ds = ds.cast_column(
        "audio",
        Audio(decode=False),
    )

    # Shuffle the stream so we are not dependent on shard ordering.
    ds = ds.shuffle(
        seed=42,
        buffer_size=shuffle_buffer,
    )

    return ds


# ---------------------------------------------------------------------
# AUDIO DECODING
# ---------------------------------------------------------------------

def decode_audio(audio_data):
    """
    Decode audio returned by Hugging Face Audio(decode=False).
    """

    if audio_data is None:
        raise ValueError("Audio field is None.")

    if isinstance(audio_data, dict):

        audio_bytes = audio_data.get("bytes")
        audio_path = audio_data.get("path")

        if audio_bytes is not None:

            if isinstance(audio_bytes, memoryview):
                audio_bytes = audio_bytes.tobytes()

            elif isinstance(audio_bytes, bytearray):
                audio_bytes = bytes(audio_bytes)

            audio, sr = sf.read(
                io.BytesIO(audio_bytes),
                dtype="float32",
            )

            return audio, sr

        if audio_path:

            audio, sr = sf.read(
                audio_path,
                dtype="float32",
            )

            return audio, sr

        raise ValueError(
            "Audio dictionary contains neither bytes nor path."
        )

    if isinstance(
        audio_data,
        (bytes, bytearray, memoryview),
    ):

        if isinstance(audio_data, memoryview):
            audio_data = audio_data.tobytes()

        elif isinstance(audio_data, bytearray):
            audio_data = bytes(audio_data)

        audio, sr = sf.read(
            io.BytesIO(audio_data),
            dtype="float32",
        )

        return audio, sr

    raise TypeError(
        f"Unsupported audio type: {type(audio_data)}"
    )


# ---------------------------------------------------------------------
# AUDIO PREPROCESSING
# ---------------------------------------------------------------------

def convert_to_mono(audio):
    """Convert stereo/multi-channel audio to mono."""

    if audio.ndim == 1:
        return audio

    return np.mean(
        audio,
        axis=1,
    ).astype(np.float32)


def trim_to_trailing_window(
    audio,
    sr,
    window_sec,
):
    """
    Keep the last `window_sec` seconds.

    Shorter clips are left-padded with silence.
    """

    target_len = int(
        window_sec * sr
    )

    if len(audio) >= target_len:
        return audio[-target_len:]

    pad_len = target_len - len(audio)

    padding = np.zeros(
        pad_len,
        dtype=np.float32,
    )

    return np.concatenate(
        [padding, audio]
    )


def preprocess_audio(
    audio,
    sr,
    window_sec,
):
    """
    Convert audio to:
        mono
        16 kHz
        fixed 2.5-second trailing window
    """

    audio = np.asarray(
        audio,
        dtype=np.float32,
    )

    audio = convert_to_mono(audio)

    if sr != SAMPLE_RATE:

        audio = librosa.resample(
            audio,
            orig_sr=sr,
            target_sr=SAMPLE_RATE,
        )

        sr = SAMPLE_RATE

    audio = trim_to_trailing_window(
        audio,
        sr,
        window_sec,
    )

    audio = np.nan_to_num(
        audio,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return audio.astype(np.float32)


# ---------------------------------------------------------------------
# COLLECT REAL HINDI DATA
# ---------------------------------------------------------------------

def collect_hindi_samples(
    ds,
    train_per_label,
    val_per_label,
):
    """
    Collect balanced REAL Hindi samples.

    Groups:
        Hindi label 0
        Hindi label 1

    Synthetic examples are explicitly excluded.

    No test samples are created here because the existing real test set
    should remain untouched for fair comparison.
    """

    required_per_label = (
        train_per_label + val_per_label
    )

    buckets = defaultdict(list)

    print()
    print("=" * 60)
    print("COLLECTING REAL HINDI DATA")
    print("=" * 60)

    print(
        f"Need {required_per_label} samples "
        "for each Hindi label."
    )

    print(
        f"  Hindi label 0 -> "
        f"{train_per_label} train + {val_per_label} val"
    )

    print(
        f"  Hindi label 1 -> "
        f"{train_per_label} train + {val_per_label} val"
    )

    print()
    print(
        "Streaming will stop automatically once both "
        "Hindi label groups are complete."
    )
    print()

    for ex in tqdm(
        ds,
        desc="streaming Smart Turn",
    ):

        language = ex.get("language")

        # ---------------------------------------------------------
        # Hindi only
        # ---------------------------------------------------------

        if language != LANGUAGE:
            continue

        # ---------------------------------------------------------
        # Ignore examples without endpoint label
        # ---------------------------------------------------------

        endpoint = ex.get("endpoint_bool")

        if endpoint is None:
            continue

        # ---------------------------------------------------------
        # IMPORTANT: real data only
        # ---------------------------------------------------------

        synthetic = bool(
            ex.get("synthetic") or False
        )

        if synthetic:
            continue

        label = int(endpoint)

        if label not in LABELS:
            continue

        # ---------------------------------------------------------
        # Stop collecting this label once enough examples exist
        # ---------------------------------------------------------

        if len(buckets[label]) >= required_per_label:
            continue

        buckets[label].append(ex)

        # ---------------------------------------------------------
        # Stop the entire streaming process once BOTH groups
        # have enough samples.
        # ---------------------------------------------------------

        if all(
            len(buckets[label]) >= required_per_label
            for label in LABELS
        ):
            break

    print()
    print("=" * 60)
    print("COLLECTION RESULT")
    print("=" * 60)

    for label in LABELS:

        count = len(
            buckets[label]
        )

        print(
            f"Hindi label={label}: {count}"
        )

        if count < required_per_label:

            raise RuntimeError(
                f"Not enough REAL Hindi data for label {label}. "
                f"Required={required_per_label}, found={count}."
            )

    # -------------------------------------------------------------
    # Deterministic split
    # -------------------------------------------------------------

    rng = np.random.default_rng(42)

    train_rows = []
    val_rows = []

    for label in LABELS:

        samples = list(
            buckets[label]
        )

        rng.shuffle(samples)

        train_rows.extend(
            samples[:train_per_label]
        )

        val_rows.extend(
            samples[
                train_per_label:
                train_per_label + val_per_label
            ]
        )

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)

    return train_rows, val_rows


# ---------------------------------------------------------------------
# PROCESS AUDIO
# ---------------------------------------------------------------------

def process_examples(
    examples,
    out_dir,
    window_sec,
    split_name,
):
    """
    Decode, preprocess and save examples as WAV files.
    """

    audio_dir = os.path.join(
        out_dir,
        "audio",
        split_name,
    )

    os.makedirs(
        audio_dir,
        exist_ok=True,
    )

    rows = []

    for i, ex in enumerate(
        tqdm(
            examples,
            desc=f"processing {split_name}",
        )
    ):

        try:

            audio, sr = decode_audio(
                ex["audio"]
            )

            audio = preprocess_audio(
                audio,
                sr,
                window_sec,
            )

        except Exception as e:

            print(
                f"\nWARNING: Could not decode "
                f"sample {i}: {e}"
            )

            continue

        label = int(
            ex["endpoint_bool"]
        )

        language = ex.get(
            "language",
            "unk",
        )

        midfiller = bool(
            ex.get("midfiller") or False
        )

        endfiller = bool(
            ex.get("endfiller") or False
        )

        synthetic = bool(
            ex.get("synthetic") or False
        )

        source = ex.get(
            "dataset",
            "smart_turn",
        )

        original_id = ex.get(
            "id",
            f"{split_name}_{i}",
        )

        filename = (
            f"{split_name}_{i:06d}.wav"
        )

        filepath = os.path.join(
            audio_dir,
            filename,
        )

        sf.write(
            filepath,
            audio,
            SAMPLE_RATE,
        )

        rows.append(
            {
                "path": filepath,
                "label": label,
                "language": language,
                "midfiller": midfiller,
                "endfiller": endfiller,
                "synthetic": synthetic,
                "source": source,
                "original_id": original_id,
                "split": split_name,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------

def print_dataset_summary(manifest):

    print()
    print("=" * 60)
    print("REAL HINDI DATASET SUMMARY")
    print("=" * 60)

    print(
        f"\nTotal samples: {len(manifest)}"
    )

    print(
        "\nSplit distribution:"
    )

    print(
        manifest["split"]
        .value_counts()
        .sort_index()
    )

    print(
        "\nLanguage × label distribution:"
    )

    distribution = pd.crosstab(
        [
            manifest["split"],
            manifest["language"],
        ],
        manifest["label"],
    )

    print(distribution)

    print(
        "\nSynthetic samples:"
    )

    print(
        manifest.groupby("split")["synthetic"].sum()
    )

    # Safety check.
    if manifest["synthetic"].any():

        raise RuntimeError(
            "ERROR: Synthetic data found. "
            "This script should contain REAL Hindi only."
        )

    print(
        "\n✓ All collected samples are REAL Smart Turn data."
    )

    print(
        "✓ Language is Hindi only."
    )


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--out_dir",
        default="data/processed_hindi",
        help="Output directory for real Hindi data.",
    )

    parser.add_argument(
        "--window_sec",
        type=float,
        default=2.5,
        help="Audio window length in seconds.",
    )

    parser.add_argument(
        "--train_per_label",
        type=int,
        default=100,
        help="Number of real Hindi samples per label for training.",
    )

    parser.add_argument(
        "--val_per_label",
        type=int,
        default=30,
        help="Number of real Hindi samples per label for validation.",
    )

    parser.add_argument(
        "--shuffle_buffer",
        type=int,
        default=200,
        help="Streaming shuffle buffer.",
    )

    args = parser.parse_args()

    if args.train_per_label <= 0:
        parser.error(
            "train_per_label must be > 0"
        )

    if args.val_per_label < 0:
        parser.error(
            "val_per_label must be >= 0"
        )

    if args.shuffle_buffer <= 0:
        parser.error(
            "shuffle_buffer must be > 0"
        )

    os.makedirs(
        args.out_dir,
        exist_ok=True,
    )

    total_train = (
        args.train_per_label * 2
    )

    total_val = (
        args.val_per_label * 2
    )

    total = (
        total_train + total_val
    )

    print("=" * 60)
    print("SMART TURN — REAL HINDI DATA PREPARATION")
    print("=" * 60)

    print()
    print("Language: Hindi (hin)")
    print("Synthetic: EXCLUDED")
    print(f"Audio window: {args.window_sec} sec")

    print()
    print("Expected dataset:")
    print(
        f"  Train: {total_train} "
        f"({args.train_per_label} per label)"
    )
    print(
        f"  Validation: {total_val} "
        f"({args.val_per_label} per label)"
    )
    print(
        f"  Total: {total}"
    )

    print()
    print(
        f"Output: {args.out_dir}"
    )

    # -------------------------------------------------------------
    # LOAD DATASET
    # -------------------------------------------------------------

    ds = load_raw_dataset(
        shuffle_buffer=args.shuffle_buffer
    )

    # -------------------------------------------------------------
    # COLLECT REAL HINDI
    # -------------------------------------------------------------

    train_rows, val_rows = collect_hindi_samples(
        ds,
        train_per_label=args.train_per_label,
        val_per_label=args.val_per_label,
    )

    # -------------------------------------------------------------
    # PROCESS AUDIO
    # -------------------------------------------------------------

    train_df = process_examples(
        train_rows,
        args.out_dir,
        args.window_sec,
        "train",
    )

    val_df = process_examples(
        val_rows,
        args.out_dir,
        args.window_sec,
        "val",
    )

    # -------------------------------------------------------------
    # MANIFEST
    # -------------------------------------------------------------

    manifest = pd.concat(
        [
            train_df,
            val_df,
        ],
        ignore_index=True,
    )

    manifest_path = os.path.join(
        args.out_dir,
        "manifest.csv",
    )

    manifest.to_csv(
        manifest_path,
        index=False,
    )

    # -------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------

    print_dataset_summary(
        manifest
    )

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)

    print()
    print(
        f"Manifest saved to:"
    )
    print(
        f"  {manifest_path}"
    )

    print()
    print(
        "Audio saved to:"
    )
    print(
        f"  {os.path.join(args.out_dir, 'audio')}"
    )

    print()
    print(
        "IMPORTANT:"
    )
    print(
        "This dataset contains TRAIN + VALIDATION only."
    )
    print(
        "Your existing real test set remains untouched."
    )


if __name__ == "__main__":
    main()

