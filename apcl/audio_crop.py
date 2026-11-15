from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .data import _resample_linear, read_wav_mono


def _event_samples(waveform: torch.Tensor, sample_rate: int) -> dict[str, Any]:
    if waveform.numel() == 0:
        return {
            "peak_sample": 0,
            "rms_peak_sample": 0,
            "onset_sample": 0,
            "peak_time_sec": 0.0,
            "rms_peak_time_sec": 0.0,
            "onset_time_sec": 0.0,
            "peak_abs_amplitude": 0.0,
            "rms_peak": 0.0,
            "noise_floor_rms": 0.0,
            "onset_threshold_rms": 0.0,
        }

    abs_waveform = waveform.abs()
    peak_sample = int(abs_waveform.argmax().item())
    peak_value = float(abs_waveform[peak_sample].item())

    frame = max(int(round(sample_rate * 0.005)), 1)
    hop = max(int(round(sample_rate * 0.001)), 1)
    padded = waveform
    if padded.numel() < frame:
        padded = F.pad(padded, (0, frame - padded.numel()))
    frames = padded.unfold(0, frame, hop)
    if frames.numel() == 0:
        rms = torch.sqrt((padded[:frame] * padded[:frame]).mean()).reshape(1)
    else:
        rms = torch.sqrt((frames * frames).mean(dim=1).clamp_min(0.0))
    starts = torch.arange(rms.numel(), device=waveform.device) * hop
    rms_peak_idx = int(rms.argmax().item())
    rms_peak_sample = int(starts[rms_peak_idx].item())
    rms_peak = float(rms[rms_peak_idx].item())

    baseline_frames = max(1, min(int(round(0.1 * sample_rate / hop)), int(rms.numel())))
    baseline = rms[:baseline_frames]
    noise_floor = float(baseline.median().item())
    noise_mad = float((baseline - noise_floor).abs().median().item())
    threshold = max(noise_floor + 6.0 * noise_mad, 0.1 * rms_peak)
    search_end = max(rms_peak_idx + 1, 1)
    above = torch.nonzero(rms[:search_end] >= threshold, as_tuple=False).flatten()
    onset_idx = int(above[0].item()) if above.numel() else rms_peak_idx
    onset_sample = int(starts[onset_idx].item())

    return {
        "peak_sample": peak_sample,
        "rms_peak_sample": rms_peak_sample,
        "onset_sample": onset_sample,
        "peak_time_sec": float(peak_sample / sample_rate),
        "rms_peak_time_sec": float(rms_peak_sample / sample_rate),
        "onset_time_sec": float(onset_sample / sample_rate),
        "peak_abs_amplitude": peak_value,
        "rms_peak": rms_peak,
        "noise_floor_rms": noise_floor,
        "onset_threshold_rms": float(threshold),
    }


def crop_waveform(
    waveform: torch.Tensor,
    sample_rate: int,
    clip_seconds: float,
    crop_mode: str = "first",
    crop_pre_seconds: float = 0.1,
    external_event_time_sec: float | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    clip_samples = int(round(float(clip_seconds) * sample_rate))
    timing = _event_samples(waveform, sample_rate)
    if crop_mode == "first":
        start = 0
        event_sample = 0
    elif crop_mode in {"onset", "peak", "rms_peak"}:
        event_sample = int(timing[f"{crop_mode}_sample"])
        start = event_sample - int(round(float(crop_pre_seconds) * sample_rate))
        start = max(0, min(start, max(int(waveform.numel()) - clip_samples, 0)))
    elif crop_mode == "external_start":
        if external_event_time_sec is None:
            raise ValueError("external_event_time_sec is required when crop_mode='external_start'.")
        event_sample = int(round(float(external_event_time_sec) * sample_rate))
        event_sample = max(0, min(event_sample, max(int(waveform.numel()) - 1, 0)))
        start = event_sample - int(round(float(crop_pre_seconds) * sample_rate))
        start = max(0, min(start, max(int(waveform.numel()) - clip_samples, 0)))
    else:
        raise ValueError(f"Unsupported crop_mode={crop_mode!r}")

    end = start + clip_samples
    clip = waveform[start:end]
    if clip.numel() < clip_samples:
        clip = F.pad(clip, (0, clip_samples - clip.numel()))
    info = {
        "crop_mode": crop_mode,
        "crop_pre_seconds": float(crop_pre_seconds),
        "crop_start_sample": int(start),
        "crop_end_sample": int(end),
        "crop_start_time_sec": float(start / sample_rate),
        "crop_end_time_sec": float(end / sample_rate),
        "crop_event_sample": int(event_sample),
        "crop_event_time_sec": float(event_sample / sample_rate),
        "external_event_time_sec": None if external_event_time_sec is None else float(external_event_time_sec),
        **timing,
    }
    return clip, info


def load_cropped_clip(
    path: str,
    sample_rate: int,
    clip_seconds: float,
    crop_mode: str = "first",
    crop_pre_seconds: float = 0.1,
    external_event_time_sec: float | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    src_sr, waveform_np = read_wav_mono(path)
    waveform = torch.from_numpy(waveform_np).float()
    waveform = waveform / waveform.abs().max().clamp_min(1e-8)
    waveform = _resample_linear(waveform, src_sr, sample_rate)
    clip, info = crop_waveform(
        waveform=waveform,
        sample_rate=sample_rate,
        clip_seconds=clip_seconds,
        crop_mode=crop_mode,
        crop_pre_seconds=crop_pre_seconds,
        external_event_time_sec=external_event_time_sec,
    )
    info["source_sample_rate"] = int(src_sr)
    info["model_sample_rate"] = int(sample_rate)
    info["source_path"] = str(path)
    return clip, info
