"""
Two model tiers, matching the "accurate AND efficient" brief:

1. WhisperTurnClassifier — Whisper-tiny encoder (pretrained, multilingual)
   + attention-pooling + small MLP head. Higher accuracy, especially on
   Hinglish, because Whisper's pretraining already covers Hindi/English
   phonetics. ~39M params, heavier.

2. TinyCNNGRU — trained fully from scratch on log-mel spectrograms, no
   Whisper dependency. ~1-2M params, much faster, good for the "tiny + fast"
   production target. Train it standalone, or distill it from model 1
   (see train.py --distill_from).
"""
import torch
import torch.nn as nn


class WhisperTurnClassifier(nn.Module):
    def __init__(self, whisper_name="openai/whisper-tiny", freeze_encoder=True,
                 unfreeze_last_n=0, hidden_dim=128, dropout=0.2):
        super().__init__()
        from transformers import WhisperModel
        full = WhisperModel.from_pretrained(whisper_name)
        self.encoder = full.encoder
        enc_dim = self.encoder.config.d_model  # 384 for whisper-tiny

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            if unfreeze_last_n > 0:
                for layer in self.encoder.layers[-unfreeze_last_n:]:
                    for p in layer.parameters():
                        p.requires_grad = True

        self.pool_attn = nn.Linear(enc_dim, 1)
        self.head = nn.Sequential(
            nn.Linear(enc_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, input_features):
        # input_features: (B, 80, 3000) log-mel from WhisperFeatureExtractor
        enc_out = self.encoder(input_features).last_hidden_state  # (B, T, D)
        attn_weights = torch.softmax(self.pool_attn(enc_out), dim=1)
        pooled = (enc_out * attn_weights).sum(dim=1)
        logits = self.head(pooled).squeeze(-1)
        return logits


class TinyCNNGRU(nn.Module):
    """
    Small conv stack (downsamples the mel spectrogram in time+freq) feeding
    a single-layer GRU, then a linear head. Designed for CPU real-time use:
    a couple hundred thousand params, sub-10ms inference on a laptop CPU
    once exported to ONNX/int8.
    """

    def __init__(self, n_mels=64, conv_channels=32, gru_hidden=64, dropout=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_channels),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),  # halve freq and time
            nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_channels),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        )
        freq_after = n_mels // 4
        self.gru = nn.GRU(
            input_size=conv_channels * freq_after,
            hidden_size=gru_hidden,
            batch_first=True,
            bidirectional=True,
        )
        self.pool_attn = nn.Linear(gru_hidden * 2, 1)
        self.head = nn.Sequential(
            nn.Linear(gru_hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, log_mel):
        # log_mel: (B, n_mels, T)
        x = log_mel.unsqueeze(1)               # (B, 1, n_mels, T)
        x = self.conv(x)                       # (B, C, n_mels/4, T/4)
        B, C, F, T = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, T, C * F)  # (B, T, C*F)
        out, _ = self.gru(x)                   # (B, T, 2*hidden)
        attn_weights = torch.softmax(self.pool_attn(out), dim=1)  # (B, T, 1)
        pooled = (out * attn_weights).sum(dim=1)  # attention pool, not plain mean
        logits = self.head(pooled).squeeze(-1)
        return logits

    def count_params(self):
        return sum(p.numel() for p in self.parameters())
