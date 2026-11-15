from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pann import PannCnn14Encoder, PretrainedFbankAudioEncoder


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PrecomputedEmbeddingEncoder(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("PrecomputedEmbeddingEncoder expects batch['clip_embeddings'], not waveforms.")


class APCLModel(nn.Module):
    def __init__(
        self,
        pann_repo_root: str,
        pann_checkpoint_path: str | None,
        anchor_table_norm: torch.Tensor,
        sample_rate: int = 32000,
        audio_dim: int = 512,
        projection_dim: int = 128,
        audio_projection_hidden_dim: int = 256,
        param_encoder_hidden_dim: int = 64,
        freeze_audio_encoder: bool = True,
        audio_encoder_config: dict | None = None,
        anchor_candidate_mask: torch.Tensor | None = None,
        dropout: float = 0.1,
        predict_uncertainty: bool = False,
        predict_vae: bool = False,
        vae_latent_dim: int | None = None,
        vae_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.predict_uncertainty = bool(predict_uncertainty)
        self.predict_vae = bool(predict_vae)
        theta_dim = int(anchor_table_norm.shape[1])
        vae_latent_dim = int(vae_latent_dim or theta_dim)
        vae_hidden_dim = int(vae_hidden_dim or audio_dim)
        self.audio_encoder = self._build_audio_encoder(
            pann_repo_root=pann_repo_root,
            pann_checkpoint_path=pann_checkpoint_path,
            sample_rate=sample_rate,
            freeze_audio_encoder=freeze_audio_encoder,
            audio_encoder_config=audio_encoder_config,
        )
        self.clip_proj = MLP(self.audio_encoder.embedding_dim, audio_dim, audio_dim, dropout)
        self.material_head = nn.Linear(audio_dim * 2, anchor_table_norm.shape[0])
        self.residual_head = MLP(audio_dim * 2, audio_dim, anchor_table_norm.shape[1], dropout)
        self.direct_head = MLP(audio_dim * 2, audio_dim, anchor_table_norm.shape[1], dropout)
        if self.predict_uncertainty:
            self.uncertainty_head = MLP(audio_dim * 2, audio_dim, anchor_table_norm.shape[1], dropout)
        if self.predict_vae:
            self.vae_mu_head = MLP(audio_dim * 2, vae_hidden_dim, vae_latent_dim, dropout)
            self.vae_logvar_head = MLP(audio_dim * 2, vae_hidden_dim, vae_latent_dim, dropout)
            self.vae_decoder = nn.Sequential(
                nn.Linear(vae_latent_dim, vae_hidden_dim),
                nn.SiLU(),
                nn.LayerNorm(vae_hidden_dim),
                nn.Dropout(dropout),
                nn.Linear(vae_hidden_dim, theta_dim),
            )
        self.audio_projection = nn.Sequential(
            nn.Linear(audio_dim * 2, audio_projection_hidden_dim),
            nn.ReLU(),
            nn.Linear(audio_projection_hidden_dim, projection_dim),
        )
        self.param_encoder = nn.Sequential(
            nn.Linear(anchor_table_norm.shape[1], param_encoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(param_encoder_hidden_dim, projection_dim),
        )
        self.register_buffer("anchor_table_norm", anchor_table_norm.float())
        if anchor_candidate_mask is None:
            anchor_candidate_mask = torch.ones(anchor_table_norm.shape[0], dtype=torch.bool)
        self.register_buffer("anchor_candidate_mask", anchor_candidate_mask.bool(), persistent=False)

    @staticmethod
    def _build_audio_encoder(
        pann_repo_root: str,
        pann_checkpoint_path: str | None,
        sample_rate: int,
        freeze_audio_encoder: bool,
        audio_encoder_config: dict | None,
    ) -> nn.Module:
        encoder_cfg = audio_encoder_config or {}
        encoder_type = str(encoder_cfg.get("type", "pann")).lower()
        freeze = bool(encoder_cfg.get("freeze", freeze_audio_encoder))
        if encoder_type in {"precomputed", "cached"}:
            return PrecomputedEmbeddingEncoder(int(encoder_cfg["embedding_dim"]))
        if encoder_type in {"pann", "panns", "cnn14"}:
            return PannCnn14Encoder(
                repo_root=str(encoder_cfg.get("repo_root", pann_repo_root)),
                checkpoint_path=encoder_cfg.get("checkpoint_path", pann_checkpoint_path),
                sample_rate=sample_rate,
                freeze=freeze,
            )
        if encoder_type in {"eat", "sslam"}:
            return PretrainedFbankAudioEncoder(
                model_name=str(encoder_cfg.get("pretrained_model_name", "")),
                input_sample_rate=sample_rate,
                sample_rate=int(encoder_cfg.get("sample_rate", 16000)),
                target_length=int(encoder_cfg.get("target_length", 1024)),
                norm_mean=float(encoder_cfg.get("norm_mean", -4.268)),
                norm_std=float(encoder_cfg.get("norm_std", 4.569)),
                pooling=str(encoder_cfg.get("pooling", "mean")).lower(),
                freeze=freeze,
                trust_remote_code=bool(encoder_cfg.get("trust_remote_code", True)),
            )
        raise ValueError("model.audio_encoder.type must be one of: pann, eat, sslam.")

    def encode_clips(self, waveforms: torch.Tensor) -> torch.Tensor:
        bsz, hits, samples = waveforms.shape
        clip_embeddings = self.audio_encoder(waveforms.reshape(bsz * hits, samples))
        clip_embeddings = self.clip_proj(clip_embeddings)
        return clip_embeddings.reshape(bsz, hits, -1)

    @staticmethod
    def pool_hits(clip_embeddings: torch.Tensor, hit_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask = hit_mask.unsqueeze(-1).to(clip_embeddings.dtype)
        count = mask.sum(dim=1).clamp_min(1.0)
        mean = (clip_embeddings * mask).sum(dim=1) / count
        centered = (clip_embeddings - mean.unsqueeze(1)) * mask
        var = centered.pow(2).sum(dim=1) / count
        std = torch.sqrt(var.clamp_min(1e-8))
        return torch.cat([mean, std], dim=-1), mean

    def predict_from_set_embedding(self, set_embedding: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = self.material_head(set_embedding)
        if not bool(self.anchor_candidate_mask.any()):
            raise ValueError("anchor_candidate_mask must keep at least one material candidate.")
        masked_logits = logits.masked_fill(~self.anchor_candidate_mask.unsqueeze(0), torch.finfo(logits.dtype).min)
        probs = torch.softmax(masked_logits, dim=-1)
        anchor_theta = probs @ self.anchor_table_norm
        residual = self.residual_head(set_embedding)
        theta = anchor_theta + residual
        q_audio = F.normalize(self.audio_projection(set_embedding), dim=-1)
        out = {
            "material_logits": masked_logits,
            "material_logits_raw": logits,
            "material_probs": probs,
            "theta_anchor": anchor_theta,
            "theta_residual": residual,
            "theta": theta,
            "theta_direct": self.direct_head(set_embedding),
            "q_audio": q_audio,
        }
        if self.predict_uncertainty:
            out["theta_log_var"] = self.uncertainty_head(set_embedding)
        if self.predict_vae:
            vae_mu = self.vae_mu_head(set_embedding)
            vae_logvar = self.vae_logvar_head(set_embedding)
            if self.training:
                std = torch.exp(0.5 * vae_logvar)
                z = vae_mu + torch.randn_like(std) * std
            else:
                z = vae_mu
            out["vae_mu"] = vae_mu
            out["vae_logvar"] = vae_logvar
            out["vae_z"] = z
            out["theta_vae"] = self.vae_decoder(z)
        return out

    def forward(
        self,
        waveforms: torch.Tensor | None,
        hit_mask: torch.Tensor,
        clip_embeddings: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if clip_embeddings is None:
            if waveforms is None:
                raise ValueError("Either waveforms or clip_embeddings must be provided.")
            clip_embeddings = self.encode_clips(waveforms)
        else:
            clip_embeddings = self.clip_proj(clip_embeddings)
        set_embedding, clip_mean = self.pool_hits(clip_embeddings, hit_mask)
        out = self.predict_from_set_embedding(set_embedding)
        out["clip_embeddings"] = clip_embeddings
        out["set_embedding"] = set_embedding
        out["clip_mean"] = clip_mean
        return out

    def encode_params(self, theta_norm: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.param_encoder(theta_norm), dim=-1)
