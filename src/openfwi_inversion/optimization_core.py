import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import deepwave
from deepwave import scalar


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PHYSICS = {
    "dx": 10.0,
    "dt": 0.001,
    "nt": 1000,
    "frequency": 15.0,
    "model_size": 70,
    "accuracy": 8,
    "pml_width": [6, 6, 6, 6],
    "velocity_scale": 1500.0,
    "velocity_offset": -2.0,
}


def configure_physics(acquisition, data):
    """Apply shared acquisition and normalization values from the YAML file."""
    PHYSICS.update(
        dx=float(acquisition["dx"]),
        dt=float(acquisition["dt"]),
        nt=int(acquisition["nt"]),
        frequency=float(acquisition["frequency"]),
        model_size=int(acquisition["model_size"]),
        accuracy=int(acquisition["accuracy"]),
        pml_width=list(acquisition["pml_width"]),
        velocity_scale=float(data["scale"]),
        velocity_offset=float(data["offset"]),
    )


def clear_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def tv_regularization(x):
    if x.dim() == 2:
        tv_h = torch.mean(torch.abs(x[1:, :] - x[:-1, :]))
        tv_w = torch.mean(torch.abs(x[:, 1:] - x[:, :-1]))
    elif x.dim() == 3:
        tv_h = torch.mean(torch.abs(x[:, 1:, :] - x[:, :-1, :]))
        tv_w = torch.mean(torch.abs(x[:, :, 1:] - x[:, :, :-1]))
    else:
        raise ValueError(f"Unsupported input dimension: {x.dim()}")

    return tv_h + tv_w

def w2_distance_from_discretized_pdf(pdf_values1: torch.Tensor,
                                     pdf_values2: torch.Tensor,
                                     x_coords: torch.Tensor) -> torch.Tensor:
    original_shape = pdf_values1.shape

    if original_shape != pdf_values2.shape:
        raise ValueError(
            f"Input tensor shapes must be identical. pdf1: {original_shape}, pdf2: {pdf_values2.shape}"
        )

    if original_shape[-1] != x_coords.shape[0]:
        raise ValueError(
            f"The last dimension of input tensors ({original_shape[-1]}) must match the size of x_coords ({x_coords.shape[0]})."
        )

    num_points = original_shape[-1]
    reshaped_pdf1 = pdf_values1.reshape(-1, num_points)
    reshaped_pdf2 = pdf_values2.reshape(-1, num_points)

    device_local = reshaped_pdf1.device
    effective_batch_size = reshaped_pdf1.shape[0]

    min_pdf2 = reshaped_pdf2.min(dim=-1, keepdim=True)[0]

    non_negative_pdf1 = reshaped_pdf1 - min_pdf2 * 1.1
    non_negative_pdf2 = reshaped_pdf2 - min_pdf2 * 1.1

    def _integrate(pdf, x):
        dx = torch.diff(x)
        areas = (pdf[..., :-1] + pdf[..., 1:]) / 2.0 * dx
        return torch.sum(areas, dim=-1)

    total_area1 = _integrate(non_negative_pdf1, x_coords)
    total_area2 = _integrate(non_negative_pdf2, x_coords)

    reshaped_pdf1 = non_negative_pdf1 / (total_area1.unsqueeze(-1) + 1e-9)
    reshaped_pdf2 = non_negative_pdf2 / (total_area2.unsqueeze(-1) + 1e-9)

    def _pdf_to_cdf(pdf_values, x):
        dx = torch.diff(x)
        areas = (pdf_values[..., :-1] + pdf_values[..., 1:]) / 2.0 * dx
        zeros = torch.zeros((effective_batch_size, 1), device=device_local, dtype=pdf_values.dtype)
        cdf_values = torch.cat([zeros, torch.cumsum(areas, dim=-1)], dim=-1)
        return cdf_values

    def _cdf_to_quantile(cdf_values, x, t):
        t_expanded = t.expand(cdf_values.shape[0], -1).contiguous()
        right_indices = torch.searchsorted(cdf_values, t_expanded)

        right_indices = torch.clamp(right_indices, 1, num_points - 1)
        left_indices = right_indices - 1

        cdf_left = torch.gather(cdf_values, 1, left_indices)
        cdf_right = torch.gather(cdf_values, 1, right_indices)

        x_expanded = x.expand(effective_batch_size, -1)
        x_left = torch.gather(x_expanded, 1, left_indices)
        x_right = torch.gather(x_expanded, 1, right_indices)

        dcdf = cdf_right - cdf_left
        dcdf = torch.where(dcdf < 1e-8, torch.tensor(1.0, device=device_local, dtype=dcdf.dtype), dcdf)
        slope = (x_right - x_left) / dcdf
        quantile_values = x_left + (t_expanded - cdf_left) * slope

        return quantile_values

    cdf1 = _pdf_to_cdf(reshaped_pdf1, x_coords)
    cdf2 = _pdf_to_cdf(reshaped_pdf2, x_coords)

    t = torch.linspace(0, 1, num_points, device=device_local, dtype=reshaped_pdf1.dtype)

    quantile1 = _cdf_to_quantile(cdf1, x_coords, t)
    quantile2 = _cdf_to_quantile(cdf2, x_coords, t)

    integrand = (quantile1 - quantile2) ** 2
    w2_squared = torch.trapezoid(integrand, t, dim=-1)
    result_flat = torch.sqrt(w2_squared)

    if len(original_shape) == 1:
        return result_flat.squeeze(0)
    output_shape = original_shape[:-1]
    return result_flat.view(output_shape)


def receiver(v, device_local, shot_num, source_locations="up", receiver_locations="up"):
    v = (torch.squeeze(v) - PHYSICS["velocity_offset"]) * PHYSICS["velocity_scale"]

    dx = PHYSICS["dx"]
    n_shots = shot_num

    n_sources_per_shot = 1
    model_size = PHYSICS["model_size"]
    d_source = int(model_size / n_shots)
    first_source = 0
    if source_locations in ("up", "left"):
        source_depth = 0
    elif source_locations in ("down", "right"):
        source_depth = model_size - 1
    else:
        raise ValueError("source_locations must be 'up', 'down', 'left', or 'right'")

    if receiver_locations in ("up", "left"):
        receiver_depth = 0
    elif receiver_locations in ("down", "right"):
        receiver_depth = model_size - 1
    else:
        raise ValueError("receiver_locations must be 'up', 'down', 'left', or 'right'")

    n_receivers_per_shot = model_size
    d_receiver = 1
    first_receiver = 0

    freq = PHYSICS["frequency"]
    nt = PHYSICS["nt"]
    dt = PHYSICS["dt"]
    peak_time = 1.0933 / freq
    pml_width = PHYSICS["pml_width"]

    if source_locations in ("up", "down"):
        source_locations = torch.zeros(n_shots, n_sources_per_shot, 2, dtype=torch.long, device=device_local)
        source_locations[..., 0] = source_depth
        source_locations[:, 0, 1] = (torch.arange(n_shots, device=device_local) * d_source + first_source)
    else:
        source_locations = torch.zeros(n_shots, n_sources_per_shot, 2, dtype=torch.long, device=device_local)
        source_locations[..., 1] = source_depth
        source_locations[:, 0, 0] = (torch.arange(n_shots, device=device_local) * d_source + first_source)

    if receiver_locations in ("up", "down"):
        receiver_locations = torch.zeros(n_shots, n_receivers_per_shot, 2, dtype=torch.long, device=device_local)
        receiver_locations[..., 0] = receiver_depth
        receiver_locations[:, :, 1] = (
            (torch.arange(n_receivers_per_shot, device=device_local) * d_receiver + first_receiver).repeat(n_shots, 1)
        )
    else:
        receiver_locations = torch.zeros(n_shots, n_receivers_per_shot, 2, dtype=torch.long, device=device_local)
        receiver_locations[..., 1] = receiver_depth
        receiver_locations[:, :, 0] = (
            (torch.arange(n_receivers_per_shot, device=device_local) * d_receiver + first_receiver).repeat(n_shots, 1)
        )

    source_amplitudes = (
        deepwave.wavelets.ricker(freq, nt, dt, peak_time)
        .repeat(n_shots, n_sources_per_shot, 1)
        .to(device_local)
    )

    out = scalar(
        v,
        dx,
        dt,
        source_amplitudes=source_amplitudes,
        source_locations=source_locations,
        receiver_locations=receiver_locations,
        accuracy=PHYSICS["accuracy"],
        pml_width=pml_width,
        pml_freq=freq,
    )

    receiver_amplitudes = out[-1]
    return receiver_amplitudes, source_locations, receiver_locations


def plot_real_and_pred(x_true, x_pred, path):
    os.makedirs(path, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(10, 5))

    true_data = x_true.detach().cpu().numpy()
    pred_data = x_pred.detach().cpu().numpy()
    diff_data = true_data - pred_data

    vmin = min(np.min(true_data), np.min(pred_data))
    vmax = max(np.max(true_data), np.max(pred_data))

    diff_vmax = max(abs(np.min(diff_data)), abs(np.max(diff_data)))
    diff_vmin = -diff_vmax

    im1 = axes[0].imshow(true_data, vmin=vmin, vmax=vmax)
    axes[0].set_title("True")
    axes[0].axis("off")
    fig.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(pred_data, vmin=vmin, vmax=vmax)
    axes[1].set_title("Predicted")
    axes[1].axis("off")
    fig.colorbar(im2, ax=axes[1])

    im3 = axes[2].imshow(diff_data, vmin=diff_vmin, vmax=diff_vmax)
    axes[2].set_title("Difference")
    axes[2].axis("off")
    fig.colorbar(im3, ax=axes[2])

    plt.tight_layout()
    plt.savefig(os.path.join(path, "real_and_pred.png"), dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close()


def plot_errors(error_history, path):
    os.makedirs(path, exist_ok=True)
    plt.figure(figsize=(10, 6))
    steps = range(len(error_history))
    plt.plot(steps, error_history, label="Error", color="blue")
    plt.xlabel("Step")
    plt.ylabel("Error Value")
    plt.title("Error over Steps")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(path, "errors.png"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_loss(loss_history, path):
    os.makedirs(path, exist_ok=True)
    plt.figure(figsize=(10, 6))
    steps = range(len(loss_history))
    plt.plot(steps, loss_history, label="Loss", color="green")
    plt.xlabel("Step")
    plt.ylabel("Loss Value")
    plt.title("Loss over Steps")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(path, "loss.png"), dpi=300, bbox_inches="tight")
    plt.close()


def save_latents_pt(x, path):
    os.makedirs(path, exist_ok=True)
    torch.save(x, os.path.join(path, "results.pt"))


def optimize_with_gd(
    image_size,
    x_true,
    steps=1000,
    lr=1e-2,
    lam_tv=0.0,
    sigma=0.0,
    shot_num=10,
    source_locations="up",
    receiver_locations="up",
    k=100,
    use_weighted=True,
    rate_down=1.0,
    rate_power=1.0,
):
    x_true = x_true.to(device)

    latents = torch.zeros((1, 1, image_size, image_size), device=device, dtype=x_true.dtype)
    latents.requires_grad_(True)

    with torch.no_grad():
        wave_true_init, _, _ = receiver(
            x_true,
            device,
            shot_num=shot_num,
            source_locations=source_locations,
            receiver_locations=receiver_locations,
        )
        wave_true = wave_true_init + sigma * torch.randn(
            wave_true_init.shape,
            device=device,
            dtype=wave_true_init.dtype,
        )

    if use_weighted:
        amplitude_max = torch.max(abs(wave_true))
        weights = 1 / (k * abs(wave_true) / amplitude_max + 1)
        wave_true = wave_true * weights
    else:
        weights = None

    _, _, n_timesteps = wave_true.shape
    wave_true_reshaped = wave_true.reshape(-1, n_timesteps)
    x_coords = torch.linspace(0, PHYSICS["dt"] * (n_timesteps - 1), n_timesteps, device=device)
    with torch.no_grad():
        base_loss_normalizer = torch.mean(
            w2_distance_from_discretized_pdf(torch.zeros_like(wave_true_reshaped), wave_true_reshaped, x_coords)
        )

    error_history = []
    loss_history = []

    def clamp_latents():
        with torch.no_grad():
            latents.clamp_(-1, 1)

    def apply_change_rate():
        if latents.grad is None:
            return
        grad = latents.grad
        change_rate = ((torch.max(abs(grad)) + rate_down) / (abs(grad) + rate_down)) ** rate_power
        latents.grad = grad * change_rate

    def compute_loss(x0_pred_local):
        wave_pred_local, _, _ = receiver(
            x0_pred_local,
            device,
            shot_num=shot_num,
            source_locations=source_locations,
            receiver_locations=receiver_locations,
        )
        if weights is not None:
            wave_pred_local = wave_pred_local * weights

        _, _, n_timesteps_local = wave_pred_local.shape
        wave_pred_reshaped_local = wave_pred_local.reshape(-1, n_timesteps_local)
        wave_true_reshaped_local = wave_true.reshape(-1, n_timesteps_local)
        x_coords_local = torch.linspace(
            0, PHYSICS["dt"] * (n_timesteps_local - 1), n_timesteps_local, device=device
        )
        base_loss_local = torch.mean(
            w2_distance_from_discretized_pdf(wave_pred_reshaped_local, wave_true_reshaped_local, x_coords_local)
        )
        total_loss_local = base_loss_local / (base_loss_normalizer + 1e-9)

        tv_loss_local = tv_regularization(x0_pred_local)

        return total_loss_local + lam_tv * tv_loss_local

    for step in range(steps):
        latents.grad = None
        x0_pred = latents.squeeze()
        loss = compute_loss(x0_pred)
        loss.backward()
        apply_change_rate()
        with torch.no_grad():
            latents.add_(latents.grad, alpha=-lr)
        clamp_latents()

        with torch.no_grad():
            error = torch.linalg.norm(x0_pred - x_true) / torch.linalg.norm(x_true - PHYSICS["velocity_offset"])

        error_history.append(error.item())
        loss_history.append(loss.item())

        if step == 0:
            print(f"Initial relative error: {error.item():.4f}")
        if (step + 1) % 50 == 0:
            print(f"Step {step+1}/{steps} | loss={loss.item():.6f} | error={error.item():.6f}")
            clear_cuda_cache()

    return latents.detach(), error_history, loss_history
