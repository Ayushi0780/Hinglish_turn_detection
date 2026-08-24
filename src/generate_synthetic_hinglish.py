"""
Generates a synthetic Hinglish turn-detection dataset using espeak-ng
(open source, offline, apt-get install espeak-ng).

Why synthetic TTS is a legitimate approach here (not a shortcut): the
Pipecat reference dataset itself has a `synthetic` boolean column, and
several of its sources (chirp3_1, chirp3_2, rime_2) are TTS-generated, not
human recordings. This script does the same thing for the Hinglish gap.

Method: each utterance is built from tagged segments — Hindi segments
(Devanagari script, `hi` voice) and English segments (`en-us` voice) —
synthesized separately and concatenated with a short natural pause, to
approximate code-switching. This is imperfect (espeak-ng is a formant
synthesizer, not neural — flatter pitch/prosody than natural speech, which
matters because pitch slope is one of the real acoustic cues for turn
completion) but it gives correctly labeled, structurally realistic training
data at zero cost and zero recording time. Treat it as a starting corpus to
validate the pipeline and get a first accuracy read — real recordings
(hinglish_data/README.md) will still generalize better and should replace
or augment this before a final submission if time allows.

Run:
    python src/generate_synthetic_hinglish.py --out_dir hinglish_data --n_per_class 150

Requires: apt-get install espeak-ng ; pip install librosa soundfile
"""
import argparse
import os
import random
import subprocess
import tempfile
import numpy as np
import librosa
import soundfile as sf
import pandas as pd

SAMPLE_RATE = 16000

# (language_tag, text) segments. "hi" = Devanagari, synthesized with the
# espeak-ng `hi` voice. "en" = English, synthesized with `en-us`.
# label 1 = complete turn. label 0 = incomplete, trails off after a filler.

COMPLETE_TEMPLATES = [
    [("hi", "मुझे दो कॉफ़ी चाहिए"), ("en", "thank you")],
    [("hi", "हाँ वो ठीक है, बस कर देना")],
    [("en", "I think that's it for today")],
    [("hi", "मुझे कल तक चाहिए"), ("en", "please")],
    [("hi", "ठीक है, मैं आ रहा हूँ")],
    [("en", "yes that works for me")],
    [("hi", "अच्छा चलिए फिर मिलते हैं")],
    [("hi", "मुझे यह वाला पसंद है"), ("en", "this one")],
    [("en", "okay sounds good, see you then")],
    [("hi", "हाँ बिल्कुल सही बात है")],
    [("hi", "मैंने काम पूरा कर दिया है")],
    [("en", "sure, I'll send it right away")],
    [("hi", "ठीक है धन्यवाद")],
    [("hi", "मुझे समझ आ गया")],
    [("en", "perfect, let's go with that")],
]

INCOMPLETE_TEMPLATES = [
    [("hi", "मुझे दो कॉफ़ी चाहिए"), ("hi", "मतलब")],
    [("hi", "वो"), ("hi", "actually".join([]) or "मतलब")],  # placeholder replaced below
    [("hi", "हाँ तो")],
    [("en", "I think that's"), ("hi", "मतलब")],
    [("hi", "मुझे कल तक चाहिए"), ("hi", "वो")],
    [("en", "actually")],
    [("hi", "हाँ वो")],
    [("hi", "मुझे लगता है कि"), ("hi", "वो")],
    [("en", "so basically")],
    [("hi", "ठीक है तो")],
    [("hi", "मैं सोच रहा था कि"), ("hi", "मतलब")],
    [("en", "well, I was thinking")],
    [("hi", "एक मिनट"), ("hi", "वो")],
    [("hi", "हाँ हाँ बिल्कुल"), ("hi", "बस")],
    [("en", "so what happened was")],
]
# fix the placeholder row above (kept simple templates only)
INCOMPLETE_TEMPLATES[1] = [("hi", "वो"), ("hi", "मतलब")]


def synth_segment(lang: str, text: str, tmpdir: str) -> np.ndarray:
    voice = "hi" if lang == "hi" else "en-us"
    out_path = os.path.join(tmpdir, f"seg_{random.randint(0, 10**9)}.wav")
    subprocess.run(
        ["espeak-ng", "-v", voice, "-s", str(random.randint(140, 175)), "-w", out_path, text],
        check=True, capture_output=True,
    )
    audio, sr = sf.read(out_path)
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=SAMPLE_RATE)
    os.remove(out_path)
    return audio.astype(np.float32)


def build_utterance(segments, tmpdir: str) -> np.ndarray:
    pieces = []
    for i, (lang, text) in enumerate(segments):
        pieces.append(synth_segment(lang, text, tmpdir))
        if i < len(segments) - 1:
            gap = np.zeros(int(random.uniform(0.08, 0.18) * SAMPLE_RATE), dtype=np.float32)
            pieces.append(gap)
    audio = np.concatenate(pieces)
    # light amplitude normalization + tiny noise floor, so clips aren't
    # bit-perfect silence outside speech (more realistic for the model)
    audio = audio / (np.abs(audio).max() + 1e-6) * 0.7
    noise = np.random.randn(len(audio)).astype(np.float32) * 0.001
    return audio + noise


def split_templates(templates, val_frac: float, seed: int = 123):
    """
    Splits templates themselves (not samples) into train/val pools, so the
    same sentence never appears in both splits. Without this, the model can
    partly memorize a phrase's exact prosody instead of generalizing to new
    Hinglish phrasing — inflating val accuracy in a way that won't hold up
    on real speech.
    """
    templates = templates[:]
    rng = random.Random(seed)
    rng.shuffle(templates)
    n_val = max(1, round(len(templates) * val_frac)) if len(templates) > 2 else 0
    val_templates = templates[:n_val] if n_val > 0 else []
    train_templates = templates[n_val:] if n_val > 0 else templates
    if not train_templates:  # guard against tiny template lists
        train_templates = templates
    return train_templates, val_templates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="hinglish_data")
    ap.add_argument("--n_per_class", type=int, default=150)
    ap.add_argument("--val_frac", type=float, default=0.15)
    args = ap.parse_args()

    audio_dir = os.path.join(args.out_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    rows = []
    with tempfile.TemporaryDirectory() as tmpdir:
        idx = 0
        for label, templates, tag in [(1, COMPLETE_TEMPLATES, "complete"),
                                       (0, INCOMPLETE_TEMPLATES, "incomplete")]:
            train_templates, val_templates = split_templates(templates, args.val_frac)
            n_val = int(round(args.n_per_class * args.val_frac)) if val_templates else 0
            n_train = args.n_per_class - n_val

            for split_name, pool, count in [("train", train_templates, n_train),
                                             ("val", val_templates, n_val)]:
                n_made = 0
                while n_made < count:
                    segments = random.choice(pool)
                    try:
                        audio = build_utterance(segments, tmpdir)
                    except subprocess.CalledProcessError as e:
                        print(f"espeak-ng failed on {segments}: {e}")
                        continue

                    fname = f"{tag}_{idx:04d}.wav"
                    fpath = os.path.join(audio_dir, fname)
                    sf.write(fpath, audio, SAMPLE_RATE)

                    rows.append({
                        "path": fpath,
                        "label": label,
                        "language": "hin-eng-codeswitch",
                        "midfiller": False,
                        "endfiller": (label == 0),
                        "synthetic": True,
                        "source": "espeak_ng_synthetic",
                        "split": split_name,
                    })
                    idx += 1
                    n_made += 1

    manifest = pd.DataFrame(rows)
    manifest_path = os.path.join(args.out_dir, "manifest.csv")
    manifest.to_csv(manifest_path, index=False)
    print(f"Wrote {len(manifest)} synthetic clips to {audio_dir}")
    print(f"Manifest: {manifest_path}")
    print(manifest["label"].value_counts())
    print(manifest["split"].value_counts())
    print("\nTemplates are split-disjoint: no sentence appears in both train "
          "and val, so val performance reflects generalization, not memorization.")


if __name__ == "__main__":
    main()
