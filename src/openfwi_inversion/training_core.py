import glob
import math
import os
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import DDPMPipeline, DDPMScheduler, DDIMPipeline, DDIMScheduler, UNet2DModel
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


@dataclass
class TrainingConfig:
    image_size: int = 72
    train_batch_size: int = 64
    eval_batch_size: int = 16
    eval_num_inference_steps: int = 1000
    eval_eta: float = 0.0
    num_epochs: int = 500
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    lr_warmup_steps: int = 500
    save_image_epochs: int = 20
    save_model_epochs: int = 20
    mixed_precision: str = "no"
    output_dir: str = "outputs"
    seed: int = 42
    timesteps: int = 1000
    num_workers: int = 4
    num_processes: int = 1
    start_epoch: int = 0
    start_step: int = 0
    prediction_type: str = "epsilon"


def build_unet(image_size: int) -> UNet2DModel:
    return UNet2DModel(
        sample_size=image_size,
        in_channels=1,
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(64, 128, 256, 256),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )


def evaluate(config: TrainingConfig, epoch: int, pipeline: DDIMPipeline) -> None:
    pipeline.set_progress_bar_config(disable=True)
    device = next(pipeline.unet.parameters()).device
    pipeline = pipeline.to(device)
    generator = torch.Generator(device=device).manual_seed(config.seed)
    with torch.no_grad():
        images = pipeline(
            batch_size=config.eval_batch_size,
            generator=generator,
            num_inference_steps=config.eval_num_inference_steps,
            eta=config.eval_eta,
            output_type="numpy",
        ).images
    sample_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(sample_dir, exist_ok=True)
    np.save(f"{sample_dir}/{epoch + 1:04d}.npy", images)

    columns = math.ceil(math.sqrt(config.eval_batch_size))
    rows = math.ceil(config.eval_batch_size / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(1.5 * columns, 1.5 * rows), squeeze=False)
    axes = axes.reshape(-1)
    for idx in range(config.eval_batch_size):
        axes[idx].imshow(images[idx].squeeze(), interpolation="nearest")
        axes[idx].axis("off")
    for axis in axes[config.eval_batch_size:]:
        axis.axis("off")
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig(f"{sample_dir}/{epoch + 1:04d}.png", dpi=150)
    plt.close(fig)


def train_loop(
    config: TrainingConfig,
    model: torch.nn.Module,
    noise_scheduler: DDPMScheduler,
    optimizer: torch.optim.Optimizer,
    train_dataloader: DataLoader,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> None:
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    if accelerator.is_main_process:
        os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers("train")

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    if accelerator.is_main_process:
        print(f"Training prediction_type: {noise_scheduler.config.prediction_type}")

    global_step = config.start_step
    for epoch in range(config.start_epoch, config.num_epochs):
        model.train()
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for batch in train_dataloader:
            clean_images = batch["images"]
            noise = torch.randn(clean_images.shape, device=clean_images.device)
            batch_size = clean_images.shape[0]

            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (batch_size,),
                device=clean_images.device,
                dtype=torch.int64,
            )

            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(clean_images, noise, timesteps)
                elif noise_scheduler.config.prediction_type == "sample":
                    target = clean_images
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                loss = F.mse_loss(noise_pred, target)
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "step": global_step,
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        if accelerator.is_main_process:
            unet = accelerator.unwrap_model(model)
            was_training = unet.training
            unet.eval()

            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                eval_scheduler = DDIMScheduler.from_config(noise_scheduler.config, clip_sample=False)
                eval_pipeline = DDIMPipeline(unet=unet, scheduler=eval_scheduler)
                evaluate(config, epoch, eval_pipeline)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                epoch_dir = os.path.join(config.output_dir, f"epoch_{epoch + 1:04d}")
                save_pipeline = DDPMPipeline(unet=unet, scheduler=noise_scheduler)
                save_pipeline.save_pretrained(epoch_dir)
                trainer_state = {
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                }
                torch.save(trainer_state, os.path.join(epoch_dir, "trainer_state.pt"))

            if was_training:
                unet.train()

    accelerator.end_training()


def _resolve_resume_dir(resume_dir: str) -> str:
    if os.path.isdir(os.path.join(resume_dir, "unet")):
        return resume_dir

    pattern = os.path.join(resume_dir, "epoch_*", "trainer_state.pt")
    candidates = glob.glob(pattern)
    if not candidates:
        raise FileNotFoundError(f"No trainer_state.pt found under {resume_dir}")

    candidates.sort()
    print(f"Found {len(candidates)} checkpoint candidates, resuming from {candidates[-1]}")
    return os.path.dirname(candidates[-1])
