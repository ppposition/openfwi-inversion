"""Run one conventional gradient-based inversion selected by YAML."""

from __future__ import annotations

import json

import numpy as np
import torch

from . import optimization_core as core
from .config import require, resolve_path, snapshot_config
from .data import load_velocity


def run(config: dict) -> None:
    dataset = config["_dataset"]
    section = config["optimization"]
    runtime = config["runtime"]
    acquisition = config["acquisition"]
    if section.get("loss", "w2") != "w2":
        raise ValueError("The optimization workflow currently supports only loss: w2.")
    if int(section["image_size"]) != int(acquisition["model_size"]):
        raise ValueError("optimization.image_size must equal acquisition.model_size.")
    device = torch.device(runtime.get("device", "cuda:0" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {device} was requested, but CUDA is unavailable.")
    seed = int(runtime.get("seed", 0))
    sample_path = resolve_path(config, require(dataset, "sample_file", "datasets.<selected>"))
    output_dir = resolve_path(config, runtime.get("output_dir", "outputs")) / config["dataset"] / "optimization"
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_config(config, output_dir)

    core.device = device
    core.configure_physics(acquisition, config["data"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    x_true = load_velocity(
        sample_path,
        float(config["data"]["scale"]),
        float(config["data"]["offset"]),
        device,
        torch.float32,
        dataset.get("sample_index"),
    )
    expected_shape = (int(acquisition["model_size"]),) * 2
    if tuple(x_true.shape) != expected_shape:
        raise ValueError(f"Expected sample shape {expected_shape}, got {tuple(x_true.shape)}")
    latents, errors, losses = core.optimize_with_gd(
        image_size=int(section["image_size"]),
        x_true=x_true,
        steps=int(section["steps"]),
        lr=float(section["learning_rate"]),
        lam_tv=float(section["lambda_tv"]),
        sigma=float(section["observation_noise_std"]),
        shot_num=int(acquisition["shots"]),
        source_locations=acquisition["source_side"],
        receiver_locations=acquisition["receiver_side"],
        k=float(section["weight_k"]),
        use_weighted=bool(section["use_weighting"]),
        rate_down=float(section["rate_down"]),
        rate_power=float(section["rate_power"]),
    )
    prediction = latents.squeeze()
    core.plot_real_and_pred(x_true, prediction, output_dir)
    core.plot_errors(errors, output_dir)
    core.plot_loss(losses, output_dir)
    core.save_latents_pt(prediction, output_dir / "data")
    metrics = {
        "dataset": config["dataset"],
        "sample_file": str(sample_path),
        "seed": seed,
        "final_error": errors[-1],
        "final_loss": losses[-1],
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Results written to {output_dir}")
