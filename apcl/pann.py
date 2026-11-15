from __future__ import annotations

import os
import sys
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import AutoModel
except ImportError:  # pragma: no cover - optional dependency
    AutoModel = None

try:
    import torchaudio
except ImportError:  # pragma: no cover - optional dependency
    torchaudio = None


class PannCnn14Encoder(nn.Module):
    def __init__(
        self,
        repo_root: str,
        checkpoint_path: str | None = None,
        sample_rate: int = 32000,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        pytorch_dir = os.path.join(repo_root, "pytorch")
        for path in (repo_root, pytorch_dir):
            if path not in sys.path:
                sys.path.insert(0, path)
        from models import Cnn14  # type: ignore

        self.model = Cnn14(
            sample_rate=sample_rate,
            window_size=1024,
            hop_size=320,
            mel_bins=64,
            fmin=50,
            fmax=14000,
            classes_num=527,
        )
        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
        self.embedding_dim = 2048
        self.freeze = freeze

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint: Any = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(checkpoint, dict):
            state = checkpoint.get("model") or checkpoint.get("state_dict") or checkpoint
        else:
            state = checkpoint
        cleaned = {}
        for key, value in state.items():
            key = key.replace("module.", "")
            cleaned[key] = value
        missing, unexpected = self.model.load_state_dict(cleaned, strict=False)
        if unexpected:
            print(f"PANN checkpoint unexpected keys: {unexpected[:5]}")
        if missing:
            print(f"PANN checkpoint missing keys: {missing[:5]}")

    def train(self, mode: bool = True) -> "PannCnn14Encoder":
        super().train(mode)
        if self.freeze:
            self.model.eval()
        return self

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        if self.freeze:
            with torch.no_grad():
                output = self.model(waveforms)
        else:
            output = self.model(waveforms)
        return output["embedding"]


class PretrainedFbankAudioEncoder(nn.Module):
    """
    Hugging Face audio SSL encoder wrapper for EAT/SSLAM-style models.

    The APCL dataset provides raw waveforms. This wrapper converts them to
    128-bin Kaldi fbank features, runs the pretrained encoder, and returns a
    fixed-dimensional embedding compatible with APCL's clip projection.
    """

    def __init__(
        self,
        model_name: str,
        input_sample_rate: int,
        sample_rate: int = 16000,
        target_length: int = 1024,
        norm_mean: float = -4.268,
        norm_std: float = 4.569,
        pooling: str = "mean",
        freeze: bool = True,
        trust_remote_code: bool = True,
    ) -> None:
        super().__init__()
        if AutoModel is None:
            raise ImportError("transformers with AutoModel is required for EAT/SSLAM encoders.")
        if torchaudio is None:
            raise ImportError("torchaudio is required for EAT/SSLAM fbank preprocessing.")
        if not model_name:
            raise ValueError("audio_encoder.pretrained_model_name must be set for EAT/SSLAM encoders.")
        if input_sample_rate <= 0 or sample_rate <= 0:
            raise ValueError("Audio sample rates must be positive.")
        if target_length <= 0:
            raise ValueError("audio_encoder.target_length must be positive.")
        if norm_std <= 0:
            raise ValueError("audio_encoder.norm_std must be positive.")
        if pooling not in {"mean", "cls"}:
            raise ValueError("audio_encoder.pooling must be 'mean' or 'cls'.")

        self.model_name = model_name
        self.input_sample_rate = int(input_sample_rate)
        self.sample_rate = int(sample_rate)
        self.target_length = int(target_length)
        self.norm_mean = float(norm_mean)
        self.norm_std = float(norm_std)
        self.pooling = pooling
        self.freeze = bool(freeze)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        self.embedding_dim = self._infer_hidden_dim(self.model)

        if self.freeze:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad_(False)

    @staticmethod
    def _infer_hidden_dim(encoder: nn.Module) -> int:
        config = getattr(encoder, "config", None)
        for name in ("hidden_size", "encoder_embed_dim", "embed_dim", "d_model"):
            value = getattr(config, name, None)
            if value is not None:
                return int(value)
        return 768

    def train(self, mode: bool = True) -> "PretrainedFbankAudioEncoder":
        super().train(mode)
        if self.freeze:
            self.model.eval()
        return self

    def _waveform_to_fbank(self, waveforms: torch.Tensor) -> torch.Tensor:
        if waveforms.ndim != 2:
            waveforms = waveforms.reshape(waveforms.shape[0], -1)
        waveforms = waveforms.float()
        fbanks = []
        for waveform in waveforms:
            sample = waveform - waveform.mean()
            if self.input_sample_rate != self.sample_rate:
                sample = torchaudio.functional.resample(sample, self.input_sample_rate, self.sample_rate)
            min_fbank_samples = max(2, int(round(0.025 * self.sample_rate)))
            if sample.numel() < min_fbank_samples:
                sample = F.pad(sample, (0, min_fbank_samples - sample.numel()))
            fbank = torchaudio.compliance.kaldi.fbank(
                sample.unsqueeze(0),
                htk_compat=True,
                sample_frequency=self.sample_rate,
                use_energy=False,
                window_type="hanning",
                num_mel_bins=128,
                dither=0.0,
                frame_shift=10,
            )
            if fbank.shape[0] < self.target_length:
                fbank = F.pad(fbank, (0, 0, 0, self.target_length - fbank.shape[0]))
            else:
                fbank = fbank[: self.target_length]
            fbank = (fbank - self.norm_mean) / (self.norm_std * 2.0)
            fbanks.append(fbank)
        return torch.stack(fbanks, dim=0).unsqueeze(1)

    @staticmethod
    def _extract_tensor_features(output) -> torch.Tensor:
        if isinstance(output, torch.Tensor):
            return output
        if isinstance(output, (tuple, list)) and output:
            return output[0]
        for attr in ("last_hidden_state", "hidden_states", "pooler_output"):
            value = getattr(output, attr, None)
            if value is None:
                continue
            if attr == "hidden_states" and isinstance(value, (tuple, list)):
                return value[-1]
            return value
        raise TypeError(f"Could not extract tensor features from output type {type(output)!r}")

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        fbanks = self._waveform_to_fbank(waveforms)
        grad_context = torch.no_grad() if self.freeze else torch.enable_grad()
        with grad_context:
            if hasattr(self.model, "extract_features"):
                features = self.model.extract_features(fbanks)
            else:
                features = self.model(fbanks)
            features = self._extract_tensor_features(features)
        if features.ndim == 3:
            return features[:, 0] if self.pooling == "cls" else features.mean(dim=1)
        if features.ndim == 2:
            return features
        return features.reshape(features.shape[0], -1)
