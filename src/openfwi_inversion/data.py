"""Shared OpenFWI NumPy dataset handling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def normalize_velocity(array: np.ndarray, scale: float, offset: float) -> np.ndarray:
    return array.astype(np.float32, copy=False) / scale + offset


def split_velocity_batch(
    path: str | Path,
    output_dir: str | Path,
    samples_per_file: int = 1,
    overwrite: bool = False,
) -> list[Path]:
    """Split an OpenFWI ``(N, 1, H, W)`` array along its sample axis.

    With the default ``samples_per_file=1``, each output file has shape
    ``(1, H, W)`` and can be loaded as one item by :class:`NpyVelocityDataset`.
    Larger chunks retain the batched ``(n, 1, H, W)`` shape. The input is
    memory-mapped, so splitting does not require loading the whole batch into
    memory at once.
    """
    source_path = Path(path)
    destination = Path(output_dir)
    if not source_path.is_file():
        raise FileNotFoundError(f"Velocity batch not found: {source_path}")
    if samples_per_file < 1:
        raise ValueError("samples_per_file must be at least 1.")

    source = np.load(source_path, mmap_mode="r")
    if source.ndim != 4 or source.shape[1] != 1:
        raise ValueError(
            f"Expected a batched velocity array shaped (N, 1, H, W), "
            f"got {source.shape} in {source_path}"
        )
    if source.shape[0] == 0:
        raise ValueError(f"Velocity batch is empty: {source_path}")

    ranges = [
        (start, min(start + samples_per_file, source.shape[0]))
        for start in range(0, source.shape[0], samples_per_file)
    ]
    width = max(5, len(str(source.shape[0] - 1)))
    output_paths = [
        destination
        / (
            f"{source_path.stem}_{start:0{width}d}.npy"
            if end - start == 1
            else f"{source_path.stem}_{start:0{width}d}-{end - 1:0{width}d}.npy"
        )
        for start, end in ranges
    ]
    existing = [output_path for output_path in output_paths if output_path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite {len(existing)} existing output file(s); "
            f"first one is {existing[0]}"
        )

    destination.mkdir(parents=True, exist_ok=True)
    for (start, end), output_path in zip(ranges, output_paths, strict=True):
        chunk = source[start] if samples_per_file == 1 else source[start:end]
        np.save(output_path, chunk, allow_pickle=False)
    return output_paths


def load_velocity(
    path: Path,
    scale: float,
    offset: float,
    device: torch.device,
    dtype: torch.dtype,
    sample_index: int | None = None,
) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(f"Velocity sample not found: {path}")
    array = np.load(path)
    if sample_index is not None:
        array = array[sample_index]
    array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2-D velocity model, got {array.shape} in {path}")
    return torch.as_tensor(normalize_velocity(array, scale, offset), device=device, dtype=dtype)


class NpyVelocityDataset(Dataset):
    def __init__(self, files: list[Path], scale: float, offset: float, image_size: int) -> None:
        if not files:
            raise ValueError("At least one .npy file is required.")
        self.samples: list[tuple[Path, int | None]] = []
        for path in files:
            array = np.load(path, mmap_mode="r")
            if array.ndim in {2, 3}:
                self.samples.append((path, None))
            elif array.ndim == 4:
                self.samples.extend((path, index) for index in range(array.shape[0]))
            else:
                raise ValueError(f"Unsupported array shape {array.shape} in {path}")
        self.scale = scale
        self.offset = offset
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        path, item_index = self.samples[index]
        source = np.load(path, mmap_mode="r")
        array = np.squeeze(source if item_index is None else source[item_index])
        if array.ndim != 2:
            raise ValueError(f"Expected a 2-D model, got {array.shape} in {path}")
        pad_h = self.image_size - array.shape[0]
        pad_w = self.image_size - array.shape[1]
        if pad_h < 0 or pad_w < 0 or pad_h % 2 or pad_w % 2:
            raise ValueError(f"Cannot symmetrically pad {array.shape} to {self.image_size} x {self.image_size}")
        array = np.pad(array, ((pad_h // 2, pad_h // 2), (pad_w // 2, pad_w // 2)), mode="edge")
        tensor = torch.from_numpy(normalize_velocity(array, self.scale, self.offset))
        return {"images": tensor.unsqueeze(0)}
