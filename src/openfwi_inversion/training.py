"""YAML adapter for the diffusion training implementation."""

from __future__ import annotations

import glob
import os
from pathlib import Path

import torch
from accelerate import notebook_launcher
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader

from .config import require, resolve_path, snapshot_config
from .data import NpyVelocityDataset
from . import training_core as core


def run(config: dict) -> None:
    dataset_config = config["_dataset"]
    section = config["train"]
    runtime = config["runtime"]
    if int(section["batch_size"]) < 1 or int(section["eval_batch_size"]) < 1:
        raise ValueError("train batch sizes must be positive.")
    if int(section["epochs"]) < 1:
        raise ValueError("train.epochs must be positive.")
    if int(section["save_image_epochs"]) < 1 or int(section["save_model_epochs"]) < 1:
        raise ValueError("Training save intervals must be positive.")
    if section["prediction_type"] not in {"epsilon", "v_prediction", "sample"}:
        raise ValueError("train.prediction_type must be epsilon, v_prediction, or sample.")
    if section["mixed_precision"] not in {"no", "fp16", "bf16"}:
        raise ValueError("train.mixed_precision must be no, fp16, or bf16.")
    visible_devices = runtime.get("cuda_visible_devices")
    if visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_devices)
    data_dir = resolve_path(config, require(dataset_config, "train_dir", "datasets.<selected>"))
    output_dir = resolve_path(config, runtime.get("output_dir", "outputs")) / config["dataset"] / "train"
    snapshot_config(config, output_dir)
    file_glob = section.get("file_glob", "**/model*.npy")
    files = [Path(path) for path in sorted(glob.glob(os.path.join(data_dir, file_glob), recursive=True))]
    if not files:
        raise FileNotFoundError(f"No .npy files found under {data_dir}")

    train_config = core.TrainingConfig(
        image_size=int(section["image_size"]),
        train_batch_size=int(section["batch_size"]),
        eval_batch_size=int(section["eval_batch_size"]),
        eval_num_inference_steps=int(section["eval_steps"]),
        eval_eta=float(section["eval_eta"]),
        num_epochs=int(section["epochs"]),
        gradient_accumulation_steps=int(section["gradient_accumulation_steps"]),
        learning_rate=float(section["learning_rate"]),
        lr_warmup_steps=int(section["lr_warmup_steps"]),
        save_image_epochs=int(section["save_image_epochs"]),
        save_model_epochs=int(section["save_model_epochs"]),
        mixed_precision=section["mixed_precision"],
        output_dir=str(output_dir),
        seed=int(runtime.get("seed", 42)),
        timesteps=int(section["timesteps"]),
        num_workers=int(section["num_workers"]),
        num_processes=int(section["num_processes"]),
        prediction_type=section["prediction_type"],
    )
    dataset = NpyVelocityDataset(
        files,
        float(config["data"]["scale"]),
        float(config["data"]["offset"]),
        train_config.image_size,
    )
    if len(dataset) < train_config.train_batch_size:
        raise ValueError(
            f"Dataset has {len(dataset)} samples, fewer than batch_size={train_config.train_batch_size}."
        )
    loader = DataLoader(
        dataset,
        batch_size=train_config.train_batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    resume_dir = section.get("resume_dir")
    if resume_dir:
        checkpoint = core._resolve_resume_dir(str(resolve_path(config, resume_dir)))
        pipeline = DDPMPipeline.from_pretrained(checkpoint)
        model, noise_scheduler = pipeline.unet, pipeline.scheduler
        state_path = os.path.join(checkpoint, "trainer_state.pt")
        state = torch.load(state_path, map_location="cpu") if os.path.exists(state_path) else None
    else:
        base_value = section.get("base_model") or dataset_config.get("model_path")
        base_path = resolve_path(config, base_value) if base_value else None
        if base_path and base_path.exists():
            model = UNet2DModel.from_config(UNet2DModel.load_config(base_path, subfolder="unet"))
            noise_scheduler = DDPMScheduler.from_config(DDPMScheduler.load_config(base_path, subfolder="scheduler"))
        else:
            model = core.build_unet(train_config.image_size)
            noise_scheduler = DDPMScheduler(num_train_timesteps=train_config.timesteps)
        state = None
    noise_scheduler.register_to_config(prediction_type=train_config.prediction_type)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=train_config.lr_warmup_steps,
        num_training_steps=len(loader) * train_config.num_epochs,
    )
    if state:
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["lr_scheduler"])
        train_config.start_epoch = int(state.get("epoch", 0))
        train_config.start_step = int(state.get("global_step", 0))
    notebook_launcher(
        core.train_loop,
        (train_config, model, noise_scheduler, optimizer, loader, scheduler),
        num_processes=train_config.num_processes,
    )
