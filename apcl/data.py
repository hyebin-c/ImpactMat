from __future__ import annotations

import os
import random
import re
import wave
from dataclasses import dataclass
from glob import glob
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .materials import MATERIAL_NAMES, MATERIAL_TO_INDEX, TARGET_NAMES, build_anchor_table


RAND_RE = re.compile(r"^rand_(?P<material>[A-Za-z]+)_(?P<hit>\d+)\.wav$")
BLEND_WAV_RE = re.compile(r"^blend_(?P<m1>[A-Za-z]+)_(?P<m2>[A-Za-z]+)_w(?P<w>\d+)_(?P<hit>\d+)\.wav$")
BLEND_FEAT_RE = re.compile(r"^feat_(?P<m1>[A-Za-z]+)_(?P<m2>[A-Za-z]+)_w(?P<w>\d+)\.npz$")


@dataclass(frozen=True)
class ImpactGroup:
    object_id: str
    material: str
    material_index: int
    wav_paths: tuple[str, ...]


@dataclass(frozen=True)
class DatasetBImpactGroup:
    object_id: str
    source: str
    material: str
    material_index: int
    wav_paths: tuple[str, ...]
    theta_raw: tuple[float, ...]
    class_probs: tuple[float, ...]
    blend_m1: str | None = None
    blend_m2: str | None = None
    blend_w1: int | None = None
    feat_path: str | None = None


def _iter_object_dirs(data_root: str) -> Iterable[tuple[str, str]]:
    root = os.path.abspath(data_root)
    roots = [root]
    training_root = os.path.join(root, "training_dataset")
    if os.path.isdir(training_root):
        roots.append(training_root)

    seen: set[tuple[str, str]] = set()
    for scan_root in roots:
        for name in sorted(os.listdir(scan_root), key=lambda x: (not x.isdigit(), x)):
            path = os.path.join(scan_root, name)
            key = (name, os.path.realpath(path))
            if os.path.isdir(path) and key not in seen:
                seen.add(key)
                yield name, path


def _iter_root_object_dirs(data_root: str) -> Iterable[tuple[str, str]]:
    root = os.path.abspath(data_root)
    for name in sorted(os.listdir(root), key=lambda x: (not x.isdigit(), x)):
        path = os.path.join(root, name)
        if os.path.isdir(path) and name.isdigit():
            yield name, path


def _iter_training_object_dirs(data_root: str) -> Iterable[tuple[str, str]]:
    root = os.path.abspath(data_root)
    training_root = os.path.join(root, "training_dataset")
    scan_root = training_root if os.path.isdir(training_root) else root
    for name in sorted(os.listdir(scan_root), key=lambda x: (not x.isdigit(), x)):
        path = os.path.join(scan_root, name)
        if os.path.isdir(path) and name.isdigit():
            yield name, path


def _target_values_from_params(
    rho: float,
    youngs: float,
    alpha: float,
    beta: float,
    nu: float | None = None,
    eta_reference_hz: float = 1000.0,
    target_names: Sequence[str] = TARGET_NAMES,
) -> tuple[float, ...]:
    import math

    omega = 2.0 * math.pi * float(eta_reference_hz)
    eta = float(alpha) / omega + float(beta) * omega
    values = {
        "log_E": math.log(float(youngs)),
        "log_rho": math.log(float(rho)),
        "log_E_over_rho": math.log(float(youngs) / float(rho)),
        "nu": float(0.0 if nu is None else nu),
        "log_eta": math.log(max(eta, 1e-12)),
        "alpha": float(alpha),
        "beta": float(beta),
        "log_alpha": math.log(max(float(alpha), 1e-12)),
        "log_beta": math.log(max(float(beta), 1e-12)),
    }
    return tuple(float(values[name]) for name in target_names)


def discover_dataset_a_groups(data_root: str) -> list[ImpactGroup]:
    groups: list[ImpactGroup] = []
    wanted = set(MATERIAL_NAMES)
    for object_id, object_dir in _iter_object_dirs(data_root):
        wav_dir = os.path.join(object_dir, "wav")
        if not os.path.isdir(wav_dir):
            sibling = os.path.join(os.path.dirname(os.path.dirname(object_dir)), object_id, "wav")
            wav_dir = sibling if os.path.isdir(sibling) else wav_dir
        if not os.path.isdir(wav_dir):
            continue

        by_material: dict[str, list[tuple[int, str]]] = {name: [] for name in MATERIAL_NAMES}
        for filename in os.listdir(wav_dir):
            match = RAND_RE.match(filename)
            if not match:
                continue
            material = match.group("material")
            if material not in wanted:
                continue
            by_material[material].append((int(match.group("hit")), os.path.join(wav_dir, filename)))

        for material in MATERIAL_NAMES:
            paths = tuple(path for _, path in sorted(by_material[material]))
            if paths:
                groups.append(
                    ImpactGroup(
                        object_id=object_id,
                        material=material,
                        material_index=MATERIAL_TO_INDEX[material],
                        wav_paths=paths,
                    )
                )
    return groups


def discover_dataset_b_groups(
    data_root: str,
    include_rand: bool = True,
    include_blend: bool = True,
    complete_object_ids: Sequence[str] | None = None,
    eta_reference_hz: float = 1000.0,
    target_names: Sequence[str] = TARGET_NAMES,
) -> list[DatasetBImpactGroup]:
    groups: list[DatasetBImpactGroup] = []
    allowed_objects = set(str(object_id) for object_id in complete_object_ids) if complete_object_ids is not None else None

    if include_rand:
        pure_targets = build_anchor_table(target_names=target_names, eta_reference_hz=eta_reference_hz).tolist()
        for object_id, object_dir in _iter_root_object_dirs(data_root):
            if allowed_objects is not None and object_id not in allowed_objects:
                continue
            wav_dir = os.path.join(object_dir, "wav")
            if not os.path.isdir(wav_dir):
                continue
            by_material: dict[str, list[tuple[int, str]]] = {name: [] for name in MATERIAL_NAMES}
            for filename in os.listdir(wav_dir):
                match = RAND_RE.match(filename)
                if not match:
                    continue
                material = match.group("material")
                if material not in by_material:
                    continue
                by_material[material].append((int(match.group("hit")), os.path.join(wav_dir, filename)))
            for material in MATERIAL_NAMES:
                paths = tuple(path for _, path in sorted(by_material[material]))
                if paths:
                    material_index = MATERIAL_TO_INDEX[material]
                    groups.append(
                        DatasetBImpactGroup(
                            object_id=object_id,
                            source="rand",
                            material=material,
                            material_index=material_index,
                            wav_paths=paths,
                            theta_raw=tuple(float(v) for v in pure_targets[material_index]),
                            class_probs=tuple(1.0 if idx == material_index else 0.0 for idx in range(len(MATERIAL_NAMES))),
                        )
                    )

    if include_blend:
        for object_id, object_dir in _iter_training_object_dirs(data_root):
            if allowed_objects is not None and object_id not in allowed_objects:
                continue
            wav_dir = os.path.join(object_dir, "wav")
            blend_dir = os.path.join(object_dir, "blend")
            params_path = os.path.join(wav_dir, "blend_params.npz")
            if not os.path.isdir(wav_dir) or not os.path.isdir(blend_dir) or not os.path.exists(params_path):
                continue

            params = np.load(params_path, allow_pickle=True)
            target_by_combo: dict[tuple[str, str, int], tuple[float, ...]] = {}
            for idx, (m1, m2, w1) in enumerate(zip(params["blend_m1"], params["blend_m2"], params["blend_w1"])):
                combo = (str(m1), str(m2), int(w1))
                target_by_combo[combo] = _target_values_from_params(
                    rho=float(params["blend_rho"][idx]),
                    youngs=float(params["blend_E"][idx]),
                    alpha=float(params["blend_alpha"][idx]),
                    beta=float(params["blend_beta"][idx]),
                    nu=float(params["blend_nu"][idx]) if "blend_nu" in params else None,
                    eta_reference_hz=eta_reference_hz,
                    target_names=target_names,
                )

            wavs_by_combo: dict[tuple[str, str, int], list[tuple[int, str]]] = {}
            if allowed_objects is not None:
                for combo in target_by_combo:
                    m1, m2, w1 = combo
                    wavs_by_combo[combo] = [
                        (hit, os.path.join(wav_dir, f"blend_{m1}_{m2}_w{w1}_{hit}.wav"))
                        for hit in range(8)
                    ]
            else:
                for filename in os.listdir(wav_dir):
                    match = BLEND_WAV_RE.match(filename)
                    if not match:
                        continue
                    combo = (match.group("m1"), match.group("m2"), int(match.group("w")))
                    wavs_by_combo.setdefault(combo, []).append((int(match.group("hit")), os.path.join(wav_dir, filename)))

            for combo, hit_paths in sorted(wavs_by_combo.items()):
                if combo not in target_by_combo:
                    continue
                m1, m2, w1 = combo
                feat_path = os.path.join(blend_dir, f"feat_{m1}_{m2}_w{w1}.npz")
                if not os.path.exists(feat_path):
                    continue
                paths = tuple(path for _, path in sorted(hit_paths))
                if not paths:
                    continue
                groups.append(
                    DatasetBImpactGroup(
                        object_id=object_id,
                        source="blend",
                        material=f"{m1}_{m2}_w{w1}",
                        material_index=-1,
                        wav_paths=paths,
                        theta_raw=target_by_combo[combo],
                        class_probs=tuple(
                            (float(w1) / 100.0 if name == m1 else (1.0 - float(w1) / 100.0 if name == m2 else 0.0))
                            for name in MATERIAL_NAMES
                        ),
                        blend_m1=m1,
                        blend_m2=m2,
                        blend_w1=w1,
                        feat_path=feat_path,
                    )
                )
    return groups


def split_groups_by_object(
    groups: Sequence[ImpactGroup],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[ImpactGroup], list[ImpactGroup], list[ImpactGroup]]:
    object_ids = sorted({group.object_id for group in groups})
    rng = random.Random(seed)
    rng.shuffle(object_ids)
    n = len(object_ids)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    train_ids = set(object_ids[:n_train])
    val_ids = set(object_ids[n_train : n_train + n_val])
    test_ids = set(object_ids[n_train + n_val :])
    return (
        [group for group in groups if group.object_id in train_ids],
        [group for group in groups if group.object_id in val_ids],
        [group for group in groups if group.object_id in test_ids],
    )


def read_wav_mono(path: str) -> tuple[int, np.ndarray]:
    try:
        import soundfile as sf

        data, sample_rate = sf.read(path, always_2d=False)
        waveform = data.astype(np.float32)
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)
        return int(sample_rate), waveform
    except Exception:
        pass

    try:
        from scipy.io import wavfile

        sample_rate, data = wavfile.read(path)
        waveform = data.astype(np.float32)
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)
        if np.issubdtype(data.dtype, np.integer):
            waveform /= max(float(np.iinfo(data.dtype).max), 1.0)
        return int(sample_rate), waveform
    except Exception:
        pass

    with wave.open(path, "rb") as handle:
        sample_rate = handle.getframerate()
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        frames = handle.readframes(handle.getnframes())
    if width != 2:
        raise ValueError(f"Fallback wav reader supports 16-bit PCM only: {path}")
    data = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return int(sample_rate), data


def _resample_linear(waveform: torch.Tensor, src_sr: int, dst_sr: int) -> torch.Tensor:
    if src_sr == dst_sr:
        return waveform
    out_len = max(1, int(round(waveform.numel() * float(dst_sr) / float(src_sr))))
    return F.interpolate(waveform.view(1, 1, -1), size=out_len, mode="linear", align_corners=False).view(-1)


def _db_to_amplitude(db: float) -> float:
    return float(10.0 ** (float(db) / 20.0))


def _match_length(waveform: torch.Tensor, num_samples: int) -> torch.Tensor:
    if waveform.numel() >= num_samples:
        return waveform[:num_samples]
    return F.pad(waveform, (0, num_samples - waveform.numel()))


def _zero_shift(waveform: torch.Tensor, shift_samples: int) -> torch.Tensor:
    if shift_samples == 0 or waveform.numel() == 0:
        return waveform
    if abs(shift_samples) >= waveform.numel():
        return torch.zeros_like(waveform)
    if shift_samples > 0:
        return F.pad(waveform[:-shift_samples], (shift_samples, 0))
    shift = abs(shift_samples)
    return F.pad(waveform[shift:], (0, shift))


class WaveformAugmenter:
    def __init__(self, config: dict | None, sample_rate: int, clip_samples: int) -> None:
        self.config = dict(config or {})
        self.enabled = bool(self.config.get("enabled", False))
        self.sample_rate = int(sample_rate)
        self.clip_samples = int(clip_samples)
        self.noise_paths = self._discover_noise_paths(self.config.get("noise_paths", []))

    @staticmethod
    def _discover_noise_paths(patterns: Sequence[str]) -> list[str]:
        paths: list[str] = []
        seen: set[str] = set()
        for pattern in patterns:
            for path in sorted(glob(str(pattern), recursive=True)):
                if not os.path.isfile(path):
                    continue
                real_path = os.path.realpath(path)
                if real_path in seen:
                    continue
                seen.add(real_path)
                paths.append(path)
        return paths

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return waveform
        probability = float(self.config.get("probability", 1.0))
        if random.random() >= probability:
            return waveform
        out = waveform.clone()
        out = self._time_shift(out)
        out = self._apply_reverb(out)
        out = self._apply_spectral_tilt(out)
        out = self._add_real_noise(out)
        out = self._add_colored_noise(out)
        out = self._apply_saturation(out)
        return out.clamp(-1.5, 1.5)

    def _range_sample(self, key: str, default: Sequence[float]) -> float:
        values = self.config.get(key, default)
        if len(values) != 2:
            raise ValueError(f"audio_augmentation.{key} must have two values.")
        lo, hi = float(values[0]), float(values[1])
        if hi < lo:
            lo, hi = hi, lo
        return random.uniform(lo, hi)

    def _time_shift(self, waveform: torch.Tensor) -> torch.Tensor:
        shift_range = self.config.get("time_shift_seconds_range")
        if not shift_range:
            return waveform
        shift_seconds = self._range_sample("time_shift_seconds_range", [0.0, 0.0])
        return _zero_shift(waveform, int(round(shift_seconds * self.sample_rate)))

    def _load_quiet_noise_segment(self) -> torch.Tensor | None:
        if not self.noise_paths:
            return None
        path = random.choice(self.noise_paths)
        src_sr, waveform_np = read_wav_mono(path)
        noise = torch.from_numpy(waveform_np).float()
        noise = _resample_linear(noise, src_sr, self.sample_rate)
        if noise.numel() == 0:
            return None
        noise = noise - noise.mean()
        if noise.numel() <= self.clip_samples:
            return _match_length(noise, self.clip_samples)

        hop = max(1, self.clip_samples // 8)
        frames = noise.unfold(0, self.clip_samples, hop)
        if frames.numel() == 0:
            start = random.randint(0, max(noise.numel() - self.clip_samples, 0))
            return noise[start : start + self.clip_samples]
        rms = torch.sqrt((frames * frames).mean(dim=1).clamp_min(0.0))
        quiet_fraction = float(self.config.get("quiet_noise_fraction", 0.25))
        quiet_count = max(1, int(round(rms.numel() * max(0.0, min(quiet_fraction, 1.0)))))
        quiet_indices = torch.argsort(rms)[:quiet_count]
        chosen = int(quiet_indices[random.randrange(int(quiet_indices.numel()))].item())
        start = chosen * hop
        return noise[start : start + self.clip_samples]

    def _add_real_noise(self, waveform: torch.Tensor) -> torch.Tensor:
        if random.random() >= float(self.config.get("real_noise_probability", 0.0)):
            return waveform
        segment = self._load_quiet_noise_segment()
        if segment is None:
            return waveform
        segment = segment - segment.mean()
        segment = segment / segment.pow(2).mean().sqrt().clamp_min(1e-8)
        noise_db = self._range_sample("real_noise_db_range", [-48.0, -28.0])
        return waveform + segment * _db_to_amplitude(noise_db)

    def _colored_noise(self, kind: str) -> torch.Tensor:
        noise = torch.randn(self.clip_samples)
        spectrum = torch.fft.rfft(noise)
        freqs = torch.fft.rfftfreq(self.clip_samples, d=1.0 / float(self.sample_rate)).clamp_min(1.0)
        kind = kind.lower()
        if kind == "pink":
            spectrum = spectrum / torch.sqrt(freqs)
        elif kind == "brown":
            spectrum = spectrum / freqs
        elif kind == "rumble":
            spectrum = spectrum / freqs
            spectrum[freqs > float(self.config.get("rumble_cutoff_hz", 500.0))] = 0
        shaped = torch.fft.irfft(spectrum, n=self.clip_samples)
        shaped = shaped - shaped.mean()
        return shaped / shaped.pow(2).mean().sqrt().clamp_min(1e-8)

    def _add_colored_noise(self, waveform: torch.Tensor) -> torch.Tensor:
        if random.random() >= float(self.config.get("colored_noise_probability", 0.0)):
            return waveform
        kinds = list(self.config.get("colored_noise_kinds", ["pink", "brown", "rumble"]))
        kind = str(random.choice(kinds)) if kinds else "pink"
        noise = self._colored_noise(kind)
        noise_db = self._range_sample("colored_noise_db_range", [-55.0, -35.0])
        return waveform + noise * _db_to_amplitude(noise_db)

    def _apply_spectral_tilt(self, waveform: torch.Tensor) -> torch.Tensor:
        if random.random() >= float(self.config.get("spectral_tilt_probability", 0.0)):
            return waveform
        tilt = self._range_sample("spectral_tilt_db_per_octave_range", [-3.0, 3.0])
        spectrum = torch.fft.rfft(waveform)
        freqs = torch.fft.rfftfreq(waveform.numel(), d=1.0 / float(self.sample_rate)).clamp_min(80.0)
        octaves = torch.log2(freqs / 1000.0)
        gain = torch.pow(torch.tensor(10.0), (tilt * octaves / 20.0)).clamp(0.25, 4.0)
        filtered = torch.fft.irfft(spectrum * gain, n=waveform.numel())
        peak = filtered.abs().max().clamp_min(1e-8)
        return filtered / peak * waveform.abs().max().clamp_min(1e-8)

    def _apply_reverb(self, waveform: torch.Tensor) -> torch.Tensor:
        if random.random() >= float(self.config.get("reverb_probability", 0.0)):
            return waveform
        taps = int(self.config.get("reverb_taps", 4))
        max_delay = int(round(float(self.config.get("reverb_max_delay_seconds", 0.12)) * self.sample_rate))
        if taps <= 0 or max_delay <= 1:
            return waveform
        out = waveform.clone()
        decay = self._range_sample("reverb_decay_range", [0.15, 0.45])
        for tap in range(taps):
            delay = random.randint(1, max_delay)
            gain = (decay ** (tap + 1)) * random.choice([-1.0, 1.0])
            out = out + _zero_shift(waveform, delay) * float(gain)
        peak = out.abs().max().clamp_min(1e-8)
        return out / peak * waveform.abs().max().clamp_min(1e-8)

    def _apply_saturation(self, waveform: torch.Tensor) -> torch.Tensor:
        if random.random() >= float(self.config.get("saturation_probability", 0.0)):
            return waveform
        drive = self._range_sample("saturation_drive_range", [1.1, 2.0])
        saturated = torch.tanh(waveform * drive) / torch.tanh(torch.tensor(drive))
        return saturated.to(dtype=waveform.dtype)


class DatasetAImpactGroups(Dataset):
    def __init__(
        self,
        groups: Sequence[ImpactGroup],
        k_choices: Sequence[int] = (1, 2, 4, 8),
        sample_rate: int = 32000,
        clip_seconds: float = 2.0,
        random_hits: bool = True,
        embedding_cache_root: str | None = None,
        audio_augmentation: dict | None = None,
    ) -> None:
        self.groups = list(groups)
        self.k_choices = tuple(int(k) for k in k_choices)
        self.sample_rate = int(sample_rate)
        self.clip_samples = int(round(float(clip_seconds) * sample_rate))
        self.random_hits = bool(random_hits)
        self.embedding_cache_root = embedding_cache_root
        self.augmenter = WaveformAugmenter(audio_augmentation, self.sample_rate, self.clip_samples)

    def __len__(self) -> int:
        return len(self.groups)

    def _load_clip(self, path: str) -> torch.Tensor:
        src_sr, waveform_np = read_wav_mono(path)
        waveform = torch.from_numpy(waveform_np).float()
        max_abs = waveform.abs().max().clamp_min(1e-8)
        waveform = waveform / max_abs
        waveform = _resample_linear(waveform, src_sr, self.sample_rate)
        if waveform.numel() >= self.clip_samples:
            clip = waveform[: self.clip_samples]
        else:
            clip = F.pad(waveform, (0, self.clip_samples - waveform.numel()))
        return self.augmenter(clip)

    def _embedding_path(self, group: ImpactGroup, wav_path: str) -> str:
        filename = os.path.splitext(os.path.basename(wav_path))[0] + ".pt"
        return os.path.join(str(self.embedding_cache_root), group.object_id, filename)

    def _load_embedding(self, group: ImpactGroup, wav_path: str) -> torch.Tensor:
        path = self._embedding_path(group, wav_path)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing cached embedding: {path}. "
                "Run scripts/precompute_audio_embeddings.py or disable data.embedding_cache.enabled."
            )
        value = torch.load(path, map_location="cpu")
        if isinstance(value, dict):
            value = value.get("embedding", value)
        return value.float().view(-1)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        group = self.groups[index]
        max_k = min(max(self.k_choices), len(group.wav_paths))
        valid_k = [k for k in self.k_choices if k <= max_k]
        k = random.choice(valid_k) if self.random_hits else max_k
        paths = list(group.wav_paths)
        if self.random_hits:
            paths = random.sample(paths, k)
        else:
            paths = paths[:k]
        out = {
            "material_index": torch.tensor(group.material_index, dtype=torch.long),
            "object_id": group.object_id,
            "material": group.material,
            "num_hits": torch.tensor(k, dtype=torch.long),
        }
        if self.embedding_cache_root:
            out["clip_embeddings"] = torch.stack([self._load_embedding(group, path) for path in paths], dim=0)
        else:
            out["waveforms"] = torch.stack([self._load_clip(path) for path in paths], dim=0)
        return out


class DatasetBImpactGroups(Dataset):
    def __init__(
        self,
        groups: Sequence[DatasetBImpactGroup],
        k_choices: Sequence[int] = (1, 2, 4, 8),
        sample_rate: int = 32000,
        clip_seconds: float = 2.0,
        random_hits: bool = True,
        embedding_cache_root: str | None = None,
        audio_augmentation: dict | None = None,
    ) -> None:
        self.groups = list(groups)
        self.k_choices = tuple(int(k) for k in k_choices)
        self.sample_rate = int(sample_rate)
        self.clip_samples = int(round(float(clip_seconds) * sample_rate))
        self.random_hits = bool(random_hits)
        self.embedding_cache_root = embedding_cache_root
        self.augmenter = WaveformAugmenter(audio_augmentation, self.sample_rate, self.clip_samples)

    def __len__(self) -> int:
        return len(self.groups)

    def _load_clip(self, path: str) -> torch.Tensor:
        src_sr, waveform_np = read_wav_mono(path)
        waveform = torch.from_numpy(waveform_np).float()
        max_abs = waveform.abs().max().clamp_min(1e-8)
        waveform = waveform / max_abs
        waveform = _resample_linear(waveform, src_sr, self.sample_rate)
        if waveform.numel() >= self.clip_samples:
            clip = waveform[: self.clip_samples]
        else:
            clip = F.pad(waveform, (0, self.clip_samples - waveform.numel()))
        return self.augmenter(clip)

    def _embedding_path(self, group: DatasetBImpactGroup, wav_path: str) -> str:
        filename = os.path.splitext(os.path.basename(wav_path))[0] + ".pt"
        return os.path.join(str(self.embedding_cache_root), group.object_id, filename)

    def _load_embedding(self, group: DatasetBImpactGroup, wav_path: str) -> torch.Tensor:
        path = self._embedding_path(group, wav_path)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing cached embedding: {path}. "
                "Run scripts/precompute_audio_embeddings.py or disable data.embedding_cache.enabled."
            )
        value = torch.load(path, map_location="cpu")
        if isinstance(value, dict):
            value = value.get("embedding", value)
        return value.float().view(-1)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        group = self.groups[index]
        max_k = min(max(self.k_choices), len(group.wav_paths))
        valid_k = [k for k in self.k_choices if k <= max_k]
        k = random.choice(valid_k) if self.random_hits else max_k
        paths = list(group.wav_paths)
        if self.random_hits:
            paths = random.sample(paths, k)
        else:
            paths = paths[:k]
        out = {
            "material_index": torch.tensor(group.material_index, dtype=torch.long),
            "theta_raw": torch.tensor(group.theta_raw, dtype=torch.float32),
            "class_probs": torch.tensor(group.class_probs, dtype=torch.float32),
            "object_id": group.object_id,
            "material": group.material,
            "source": group.source,
            "blend_m1": group.blend_m1 or "",
            "blend_m2": group.blend_m2 or "",
            "blend_w1": torch.tensor(-1 if group.blend_w1 is None else group.blend_w1, dtype=torch.long),
            "num_hits": torch.tensor(k, dtype=torch.long),
        }
        if self.embedding_cache_root:
            out["clip_embeddings"] = torch.stack([self._load_embedding(group, path) for path in paths], dim=0)
        else:
            out["waveforms"] = torch.stack([self._load_clip(path) for path in paths], dim=0)
        return out


def collate_impact_groups(batch: Sequence[dict[str, torch.Tensor | str]]) -> dict[str, torch.Tensor | list[str]]:
    uses_embeddings = "clip_embeddings" in batch[0]
    tensor_key = "clip_embeddings" if uses_embeddings else "waveforms"
    max_k = max(int(item[tensor_key].shape[0]) for item in batch)  # type: ignore[index]
    hit_mask = torch.zeros(len(batch), max_k, dtype=torch.bool)
    output: dict[str, torch.Tensor | list[str]] = {}
    if uses_embeddings:
        emb_dim = int(batch[0]["clip_embeddings"].shape[-1])  # type: ignore[index]
        clip_embeddings = torch.zeros(len(batch), max_k, emb_dim, dtype=torch.float32)
    else:
        clip_samples = int(batch[0]["waveforms"].shape[-1])  # type: ignore[index]
        waveforms = torch.zeros(len(batch), max_k, clip_samples, dtype=torch.float32)
    for i, item in enumerate(batch):
        clips = item[tensor_key]  # type: ignore[assignment]
        k = int(clips.shape[0])
        if uses_embeddings:
            clip_embeddings[i, :k] = clips
        else:
            waveforms[i, :k] = clips
        hit_mask[i, :k] = True
    if uses_embeddings:
        output["clip_embeddings"] = clip_embeddings
    else:
        output["waveforms"] = waveforms
    output.update({
        "hit_mask": hit_mask,
        "material_index": torch.stack([item["material_index"] for item in batch]),  # type: ignore[list-item]
        "object_id": [str(item["object_id"]) for item in batch],
        "material": [str(item["material"]) for item in batch],
        "source": [str(item.get("source", "dataset_a")) for item in batch],
        "num_hits": torch.stack([item["num_hits"] for item in batch]),  # type: ignore[list-item]
    })
    if "blend_m1" in batch[0]:
        output["blend_m1"] = [str(item["blend_m1"]) for item in batch]
    if "blend_m2" in batch[0]:
        output["blend_m2"] = [str(item["blend_m2"]) for item in batch]
    if "blend_w1" in batch[0]:
        output["blend_w1"] = torch.stack([item["blend_w1"] for item in batch])  # type: ignore[list-item]
    if "theta_raw" in batch[0]:
        output["theta_raw"] = torch.stack([item["theta_raw"] for item in batch])  # type: ignore[list-item]
    if "class_probs" in batch[0]:
        output["class_probs"] = torch.stack([item["class_probs"] for item in batch])  # type: ignore[list-item]
    return output
