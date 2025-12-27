#!/usr/bin/env python3
"""
Minimal Wav2Vec2 audio embedder.

Given an audio file path, returns embeddings of shape:
    [T, 12, 768]

- Uses ONLY transformer hidden states (no CNN embedding layer)
- Optionally interpolates time dimension to match target video frames
"""

import os
import torch
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

MODEL_ID = "facebook/wav2vec2-base-960h"

class Wav2VecEmbedder:
    def __init__(self, device="cuda"):
        self.device = torch.device(device)
        print(f"Loading Wav2Vec2 model: {MODEL_ID}")
        self.processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
        self.model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID).to(self.device)
        self.model.eval()
        print(f"Model loaded on device: {self.device}")

    @torch.no_grad()
    def embed(
        self,
        audio_path: str,
        target_frames: int | None = None,
        fps: int = 24,
    ) -> torch.Tensor:
        """
        Args:
            audio_path: path to audio file (.wav / .mp3)
            target_frames: if provided, interpolate to this length
            fps: used only if target_frames is None

        Returns:
            feat: Tensor of shape [T, 12, 768]
        """
        # Load audio @ 16kHz
        audio, sr = librosa.load(audio_path, sr=16000)

        # default 24 fps and one feature per frame
        if target_frames is None:
            duration_sec = len(audio) / sr
            target_frames = int(round(duration_sec * fps))

        print(f"embed audio, duration_sec:{duration_sec}, target_frames:{target_frames}")

        # Tokenize
        input_values = self.processor(
            audio,
            sampling_rate=sr,
            return_tensors="pt",
        ).input_values.to(self.device)

        # Forward
        outputs = self.model(
            input_values,
            output_hidden_states=True,
        )

        # hidden_states:
        #   [0]   CNN feature encoder
        #   [1:]  12 transformer layers
        layers = outputs.hidden_states[1:]  # list of 12 × [1, T, 768]

        # Stack → [12, T, 768]
        feat = torch.cat(layers)

        # Interpolate time → target_frames
        # [12, T, 768] -> [12, 768, T]
        feat = feat.transpose(1, 2)
        feat = torch.nn.functional.interpolate(
            feat, # [12, 768, T]
            size=target_frames,
            mode="linear",
            align_corners=True,
        )       # [12, 768, target_frames]
        # -> [12, target_frames, 768]
        feat = feat.permute(0, 2, 1)

        # Final: [target_frames, 12, 768]
        feat = feat.permute(1, 0, 2).contiguous()

        return feat
