from __future__ import annotations

import torch
import torch.nn.functional as F


def soft_parameter_contrastive(q_audio: torch.Tensor, theta: torch.Tensor, sigma: float, temperature: float) -> torch.Tensor:
    distances = torch.cdist(theta, theta).pow(2)
    target = torch.softmax(-distances / max(sigma * sigma, 1e-8), dim=1)
    logits = q_audio @ q_audio.t() / max(temperature, 1e-8)
    log_probs = torch.log_softmax(logits, dim=1)
    return F.kl_div(log_probs, target, reduction="batchmean")


def multi_positive_cross_modal_loss(
    q_audio: torch.Tensor,
    q_param: torch.Tensor,
    theta: torch.Tensor,
    sigma: float,
    temperature: float,
) -> torch.Tensor:
    q_audio = F.normalize(q_audio, dim=-1)
    q_param = F.normalize(q_param, dim=-1)

    logits = q_audio @ q_param.t() / max(temperature, 1e-8)
    distances = torch.cdist(theta, theta, p=2)
    target_logits = -distances.pow(2) / max(sigma * sigma, 1e-8)
    target_prob = torch.softmax(target_logits, dim=-1).detach()

    log_prob_a2p = F.log_softmax(logits, dim=-1)
    loss_a2p = -(target_prob * log_prob_a2p).sum(dim=-1).mean()

    log_prob_p2a = F.log_softmax(logits.t(), dim=-1)
    loss_p2a = -(target_prob.t() * log_prob_p2a).sum(dim=-1).mean()
    return 0.5 * (loss_a2p + loss_p2a)


audio_parameter_alignment = multi_positive_cross_modal_loss


def multi_impact_consistency(clip_embeddings: torch.Tensor, hit_mask: torch.Tensor) -> torch.Tensor:
    mask = hit_mask.unsqueeze(-1).to(clip_embeddings.dtype)
    count = mask.sum(dim=1).clamp_min(1.0)
    mean = (clip_embeddings * mask).sum(dim=1, keepdim=True) / count.unsqueeze(1)
    return (((clip_embeddings - mean) ** 2) * mask).sum() / mask.sum().clamp_min(1.0)


def latent_interpolation_loss(
    model,
    set_embedding: torch.Tensor,
    theta: torch.Tensor,
    prediction_key: str = "theta",
) -> torch.Tensor:
    if set_embedding.shape[0] < 2:
        return set_embedding.new_tensor(0.0)
    perm = torch.randperm(set_embedding.shape[0], device=set_embedding.device)
    lam = torch.rand(set_embedding.shape[0], 1, device=set_embedding.device)
    z_mix = lam * set_embedding + (1.0 - lam) * set_embedding[perm]
    theta_mix = lam * theta + (1.0 - lam) * theta[perm]
    pred = model.predict_from_set_embedding(z_mix)[prediction_key]
    return F.smooth_l1_loss(pred, theta_mix)
