"""Run one diffusion-guided inversion selected by YAML."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from diffusers import DDPMScheduler, UNet2DModel

from . import diffusion_core as core
from .config import require, resolve_path, snapshot_config
from .data import load_velocity


def _device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {name!r} was requested, but CUDA is unavailable.")
    return device


def _sample_specs(dataset: dict) -> list[dict]:
    if "samples" in dataset:
        samples = dataset["samples"]
        if not isinstance(samples, list) or not samples:
            raise ValueError("datasets.<selected>.samples must be a non-empty list.")
        return samples
    return [
        {
            "name": Path(require(dataset, "sample_file", "datasets.<selected>")).stem,
            "file": require(dataset, "sample_file", "datasets.<selected>"),
            "index": dataset.get("sample_index"),
        }
    ]


def run(config: dict) -> None:
    dataset = config["_dataset"]
    section = config["diffusion"]
    runtime = config["runtime"]
    acquisition = config["acquisition"]
    if int(section["image_size"]) != int(acquisition["model_size"]) + 2:
        raise ValueError("diffusion.image_size must equal acquisition.model_size + 2.")
    device = _device(runtime.get("device", "cuda:0" if torch.cuda.is_available() else "cpu"))
    seed = int(runtime.get("seed", 0))
    model_path = resolve_path(config, require(dataset, "model_path", "datasets.<selected>"))
    sample_specs = _sample_specs(dataset)
    sample_seeds = (
        [int(item.get("seed", seed)) for item in sample_specs]
        if any("seed" in item for item in sample_specs)
        else None
    )
    output_dir = resolve_path(config, runtime.get("output_dir", "outputs")) / config["dataset"] / "diffusion"
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_config(config, output_dir)

    core.device = device
    core.configure_physics(acquisition, config["data"])
    torch.manual_seed(seed)
    np.random.seed(seed)

    unet = UNet2DModel.from_pretrained(model_path, subfolder="unet").to(device)
    scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")
    scheduler.set_timesteps(int(section["timesteps"]))
    sample_paths = [resolve_path(config, require(item, "file", "samples[]")) for item in sample_specs]
    x_true = torch.stack(
        [
            load_velocity(
                path,
                float(config["data"]["scale"]),
                float(config["data"]["offset"]),
                device,
                unet.dtype,
                item.get("index"),
            )
            for path, item in zip(sample_paths, sample_specs)
        ]
    )
    expected_shape = (int(acquisition["model_size"]),) * 2
    if tuple(x_true.shape[1:]) != expected_shape:
        raise ValueError(f"Expected sample shape {expected_shape}, got {tuple(x_true.shape[1:])}")

    latents, errors, x0_errors, losses = core.sample(
        scheduler,
        unet,
        int(section["image_size"]),
        batch_size=x_true.shape[0],
        rho=float(section["rho"]),
        rate_down=float(section["rate_down"]),
        rate_power=float(section["rate_power"]),
        x_true=x_true,
        loss_type=section["loss"],
        grad_decay=float(section["grad_decay"]),
        shot_num=int(acquisition["shots"]),
        source_locations=acquisition["source_side"],
        receiver_locations=acquisition["receiver_side"],
        k=float(section["weight_k"]),
        seed=seed,
        noise_std=float(section["observation_noise_std"]),
        sample_seeds=sample_seeds,
    )
    prediction = latents[:, 0, 1:-1, 1:-1]
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        prediction,
        data_dir / f"results_batch_{section['rho']}_{section['grad_decay']}.pt",
    )
    mean_errors = np.mean(np.asarray(errors), axis=1).tolist()
    mean_x0_errors = np.mean(np.asarray(x0_errors), axis=1).tolist()
    mean_losses = np.mean(np.asarray(losses), axis=1).tolist()
    core.plot_errors(mean_errors, mean_x0_errors, output_dir, section["rho"], section["grad_decay"])
    core.plot_loss(mean_losses, output_dir, section["rho"], section["grad_decay"])

    sample_metrics = []
    for index, (item, path) in enumerate(zip(sample_specs, sample_paths)):
        name = str(item.get("name") or f"sample_{index}")
        sample_dir = output_dir / "samples" / f"{index:03d}_{name}"
        sample_errors = [values[index] for values in errors]
        sample_x0_errors = [values[index] for values in x0_errors]
        sample_losses = [values[index] for values in losses]
        core.plot_real_and_pred(
            x_true[index], prediction[index], sample_dir, section["rho"], section["grad_decay"]
        )
        core.plot_errors(
            sample_errors, sample_x0_errors, sample_dir, section["rho"], section["grad_decay"]
        )
        core.plot_loss(sample_losses, sample_dir, section["rho"], section["grad_decay"])
        core.save_latents_pt(
            prediction[index].clone(),
            sample_dir / "data",
            section["rho"],
            section["grad_decay"],
        )
        sample_metrics.append(
            {
                "name": name,
                "sample_file": str(path),
                "sample_index": item.get("index"),
                "seed": sample_seeds[index] if sample_seeds is not None else None,
                "final_error": sample_errors[-1],
                "final_x0_error": sample_x0_errors[-1],
                "final_loss": sample_losses[-1],
            }
        )
    metrics = {
        "dataset": config["dataset"],
        "batch_size": len(sample_specs),
        "seed": seed,
        "sample_seeds": sample_seeds,
        "mean_final_error": mean_errors[-1],
        "mean_final_x0_error": mean_x0_errors[-1],
        "mean_final_loss": mean_losses[-1],
        "samples": sample_metrics,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Results written to {output_dir}")
