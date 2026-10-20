import torch
from tqdm.auto import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
import deepwave
from deepwave import scalar
from diffusers.utils.torch_utils import randn_tensor

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


def w2_distance_from_discretized_pdf(pdf_values1: torch.Tensor,
                                     pdf_values2: torch.Tensor,
                                     x_coords: torch.Tensor) -> torch.Tensor:
    """
    计算一个或多个离散 PDF 对之间的 W2 距离。
    这个函数是完全可微分的，并且支持对多维输入进行批处理。

    Args:
        pdf_values1 (torch.Tensor): 第一个分布的 PDF 值。
                                    支持的形状: (N,), (B, N), (B, R, N)
        pdf_values2 (torch.Tensor): 第二个分布的 PDF 值。形状必须与 pdf_values1 相同。
        x_coords (torch.Tensor): PDF 值对应的 x 坐标。形状: (N,)

    Returns:
        torch.Tensor: W2 距离。
                      如果输入是 (N,), 返回标量。
                      如果输入是 (B, N), 返回 (B,)。
                      如果输入是 (B, R, N), 返回 (B, R)。
    """
    # 记录原始形状以便恢复
    original_shape = pdf_values1.shape
    
    if original_shape != pdf_values2.shape:
        raise ValueError(f"Input tensor shapes must be identical. pdf1: {original_shape}, pdf2: {pdf_values2.shape}")
    
    if original_shape[-1] != x_coords.shape[0]:
        raise ValueError(f"The last dimension of input tensors ({original_shape[-1]}) must match the size of x_coords ({x_coords.shape[0]}).")

    # 将输入重塑为 (B_eff, N) 以进行批处理
    # B_eff 是所有批次维度的乘积
    num_points = original_shape[-1]
    reshaped_pdf1 = pdf_values1.reshape(-1, num_points)
    reshaped_pdf2 = pdf_values2.reshape(-1, num_points)
    
    device = reshaped_pdf1.device
    effective_batch_size = reshaped_pdf1.shape[0]
    
    # --- 1. 将PDF值置为正并进行归一化 ---
    # 通过减去每个道中的全局最小值来确保非负，而不是取绝对值
    min_pdf2 = reshaped_pdf2.min(dim=-1, keepdim=True)[0]
    
    non_negative_pdf1 = reshaped_pdf1 - min_pdf2*1.1
    non_negative_pdf2 = reshaped_pdf2 - min_pdf2*1.1

    # 辅助函数，使用梯形法则计算积分（总面积）
    def _integrate(pdf, x):
        dx = torch.diff(x)
        # pdf shape: (B_eff, N), dx shape: (N-1)
        areas = (pdf[..., :-1] + pdf[..., 1:]) / 2.0 * dx
        return torch.sum(areas, dim=-1)

    total_area1 = _integrate(non_negative_pdf1, x_coords)
    total_area2 = _integrate(non_negative_pdf2, x_coords)

    # 归一化，并添加 epsilon 以避免除以零
    # total_area shapes are (B_eff,), need to be (B_eff, 1) for broadcasting
    reshaped_pdf1 = non_negative_pdf1 / (total_area1.unsqueeze(-1) + 1e-9)
    reshaped_pdf2 = non_negative_pdf2 / (total_area2.unsqueeze(-1) + 1e-9)
    
    # 辅助函数: 将离散 PDF 转换为离散 CDF
    def _pdf_to_cdf(pdf_values, x):
        dx = torch.diff(x)
        areas = (pdf_values[..., :-1] + pdf_values[..., 1:]) / 2.0 * dx
        zeros = torch.zeros((effective_batch_size, 1), device=device, dtype=pdf_values.dtype)
        cdf_values = torch.cat([zeros, torch.cumsum(areas, dim=-1)], dim=-1)
        return cdf_values

    # 辅助函数: 将离散 CDF 转换为离散分位数函数
    def _cdf_to_quantile(cdf_values, x, t):
        # cdf_values: (B_eff, N), t: (N,)
        # 为了让 searchsorted 正确处理批次，需要将 t 扩展为 (B_eff, N)
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
        # Add epsilon to avoid division by zero
        dcdf = torch.where(dcdf < 1e-8, torch.tensor(1.0, device=device, dtype=dcdf.dtype), dcdf)
        slope = (x_right - x_left) / dcdf
        # 使用扩展后的 t_expanded 进行插值
        quantile_values = x_left + (t_expanded - cdf_left) * slope
        
        return quantile_values

    # --- 主要计算流程 ---
    
    # 1. 计算两个分布的 CDF
    cdf1 = _pdf_to_cdf(reshaped_pdf1, x_coords)
    cdf2 = _pdf_to_cdf(reshaped_pdf2, x_coords)

    # 2. 计算两个分布的分位数函数
    t = torch.linspace(0, 1, num_points, device=device, dtype=reshaped_pdf1.dtype)
    
    quantile1 = _cdf_to_quantile(cdf1, x_coords, t)
    quantile2 = _cdf_to_quantile(cdf2, x_coords, t)

    # 3. 计算 W2 距离
    integrand = (quantile1 - quantile2) ** 2
    
    # Integrate along the last dimension (the N points)
    w2_squared = torch.trapezoid(integrand, t, dim=-1)
    
    result_flat = torch.sqrt(w2_squared) # Shape: (B_eff,)
    
    # 恢复到原始批次形状
    if len(original_shape) == 1: # Input was (N,)
        return result_flat.squeeze(0) # Return scalar
    else:
        output_shape = original_shape[:-1] # (B,) or (B, R)
        return result_flat.view(output_shape)


def _as_model_batch(v):
    """Return velocity models as [batch, height, width]."""
    if v.ndim == 2:
        return v.unsqueeze(0)
    if v.ndim == 3:
        return v
    if v.ndim == 4 and v.shape[1] == 1:
        return v[:, 0]
    raise ValueError(f"Expected [H,W], [B,H,W], or [B,1,H,W], got {tuple(v.shape)}")


def _crop_model_batch(v):
    """Remove the one-cell diffusion padding without touching the batch axis."""
    models = _as_model_batch(v)
    return models[:, 1:-1, 1:-1]


def _tv_per_sample(x):
    tv_h = torch.mean(torch.abs(x[:, 1:, :] - x[:, :-1, :]), dim=(1, 2))
    tv_w = torch.mean(torch.abs(x[:, :, 1:] - x[:, :, :-1]), dim=(1, 2))
    return tv_h + tv_w


def receiver(v, device, shot_num, source_locations='up', receiver_locations='up', noise_std=0.05, noise_generators=None):
    v = _as_model_batch(v)
    batch_size = v.shape[0]
    v = (v - PHYSICS["velocity_offset"]) * PHYSICS["velocity_scale"]

    dx = PHYSICS["dx"]

    n_shots = shot_num
    total_shots = batch_size * n_shots

    n_sources_per_shot = 1
    model_size = PHYSICS["model_size"]
    d_source = int(model_size/n_shots)
    first_source = 0  
    if source_locations == 'up' or source_locations == 'left':
        source_depth = 0
    elif source_locations == 'down'or source_locations == 'right':
        source_depth = model_size - 1
    else:
        raise ValueError("source_locations must be 'up', 'down', 'left', or 'right'")
    
    if receiver_locations == 'up' or receiver_locations == 'left':
        receiver_depth = 0
    elif receiver_locations == 'down' or receiver_locations == 'right':
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

    # source_locations
    if source_locations == 'up' or source_locations == 'down':
        source_locations = torch.zeros(n_shots, n_sources_per_shot, 2,
                                    dtype=torch.long, device=device)
        source_locations[..., 0] = source_depth
        source_locations[:, 0, 1] = (torch.arange(n_shots, device=device) * d_source +
                                    first_source)
    elif source_locations == 'left' or source_locations == 'right':
        source_locations = torch.zeros(n_shots, n_sources_per_shot, 2,
                                    dtype=torch.long, device=device)
        source_locations[..., 1] = source_depth
        source_locations[:, 0, 0] = (torch.arange(n_shots, device=device) * d_source +
                                    first_source)

    if receiver_locations == 'up' or receiver_locations == 'down':
        # receiver_locations
        receiver_locations = torch.zeros(n_shots, n_receivers_per_shot, 2,
                                        dtype=torch.long, device=device)
        receiver_locations[..., 0] = receiver_depth
        receiver_locations[:, :, 1] = (
            (torch.arange(n_receivers_per_shot, device=device) * d_receiver +
            first_receiver)
            .repeat(n_shots, 1)
        )
    elif receiver_locations == 'left' or receiver_locations == 'right':
        # receiver_locations
        receiver_locations = torch.zeros(n_shots, n_receivers_per_shot, 2,
                                        dtype=torch.long, device=device)
        receiver_locations[..., 1] = receiver_depth
        receiver_locations[:, :, 0] = (
            (torch.arange(n_receivers_per_shot, device=device) * d_receiver +
            first_receiver)
            .repeat(n_shots, 1)
        )

    # source_amplitudes
    source_locations = source_locations.repeat(batch_size, 1, 1)
    receiver_locations = receiver_locations.repeat(batch_size, 1, 1)
    source_amplitudes = (
        deepwave.wavelets.ricker(freq, nt, dt, peak_time)
        .repeat(total_shots, n_sources_per_shot, 1)
        .to(device)
    )

    # Deepwave associates one batched model with each shot. Repeating every
    # model for its shots performs all (batch, shot) simulations in one call.
    # Keep the original shared-model path for batch size one.
    deepwave_v = v[0] if batch_size == 1 else v.repeat_interleave(n_shots, dim=0)

    out = scalar(deepwave_v, dx, dt, source_amplitudes=source_amplitudes,
                source_locations=source_locations,
                receiver_locations=receiver_locations,
                accuracy=PHYSICS["accuracy"],
                pml_width=pml_width,
                pml_freq=freq)

    receiver_amplitudes = out[-1].reshape(
        batch_size, n_shots, n_receivers_per_shot, nt
    )
    if noise_std > 0:
        noise = (
            torch.randn_like(receiver_amplitudes)
            if noise_generators is None
            else randn_tensor(
                receiver_amplitudes.shape, generator=noise_generators,
                device=receiver_amplitudes.device, dtype=receiver_amplitudes.dtype,
            )
        )
        receiver_amplitudes = receiver_amplitudes + noise * noise_std
    return (
        receiver_amplitudes,
        source_locations.reshape(batch_size, n_shots, n_sources_per_shot, 2),
        receiver_locations.reshape(batch_size, n_shots, n_receivers_per_shot, 2),
    )

def sample(scheduler, unet, npy_size, batch_size=1, rho=0, rate_down=1, rate_power=1, x_true=None, loss_type='mse', grad_decay=1.0, shot_num=5, source_locations='up', receiver_locations='up', k=100, seed=1, noise_std=0.05, sample_seeds=None):
    
    unet.eval()
    for param in unet.parameters():
        param.requires_grad = False
    if sample_seeds is not None:
        if len(sample_seeds) != batch_size:
            raise ValueError("sample_seeds must contain one seed per batch member.")
        generator = [torch.Generator(device).manual_seed(int(s)) for s in sample_seeds]
        # Observation noise uses a separate stream from diffusion sampling.
        noise_generators = [torch.Generator(device).manual_seed(int(s)) for s in sample_seeds]
    else:
        generator = torch.Generator(device).manual_seed(seed)
        noise_generators = None
    latents = randn_tensor(
        (batch_size, 1, npy_size, npy_size),
        generator=generator,
        device=device,
        dtype=unet.dtype
    )
    
    # 初始化用于保存中间过程的列表
    error_history = []
    error_0_history = []
    loss_history = []
    
    if x_true is not None:
        x_true = _as_model_batch(x_true.to(device))
        if x_true.shape[0] != batch_size:
            raise ValueError(
                f"x_true batch size {x_true.shape[0]} does not match latent batch size {batch_size}."
            )
        wave_true, _, _ = receiver(x_true, device, shot_num=shot_num, source_locations=source_locations, receiver_locations=receiver_locations, noise_std=noise_std, noise_generators=noise_generators)
        amplitude_max = torch.amax(abs(wave_true), dim=(1, 2, 3), keepdim=True)
        weights = 1 / (k * abs(wave_true) / amplitude_max + 1)
        wave_true = wave_true * weights


    progress_bar = tqdm(scheduler.timesteps)
    
    for i, t_step in enumerate(progress_bar):
        latent_model_input = scheduler.scale_model_input(latents, t_step)
        if rho > 0:
            latent_model_input.requires_grad_(True)
        noise_pred = unet(latent_model_input, t_step).sample
            
        # 执行scheduler step
        scheduler_output = scheduler.step(noise_pred, t_step, latents, generator=generator)
        latents = scheduler_output.prev_sample
        
        x0_pred = scheduler_output.pred_original_sample
        
        if rho > 0:
            # 根据选择的损失类型计算损失
            x0_models = _crop_model_batch(x0_pred)
            wave_pred, _, _ = receiver(x0_models, device, shot_num=shot_num, source_locations=source_locations, receiver_locations=receiver_locations, noise_std=0)
            # 将观测波场的权重应用到合成波场
            wave_pred = wave_pred * weights

            # 计算基础损失
            if loss_type == 'mse':
                base_loss_per_sample = torch.mean(
                    (wave_pred - wave_true) ** 2, dim=(1, 2, 3)
                )
            elif loss_type == 'w2':
                _, n_shots, n_receivers, n_timesteps = wave_pred.shape
                wave_pred_reshaped = wave_pred.reshape(-1, n_timesteps)
                wave_true_reshaped = wave_true.reshape(-1, n_timesteps)
                x_coords = torch.linspace(0, PHYSICS["dt"] * (n_timesteps - 1), n_timesteps, device=device)
                trace_loss = w2_distance_from_discretized_pdf(
                    wave_pred_reshaped, wave_true_reshaped, x_coords
                ).reshape(batch_size, n_shots, n_receivers)
                trace_normalizer = w2_distance_from_discretized_pdf(
                    torch.zeros_like(wave_true_reshaped), wave_true_reshaped, x_coords
                ).reshape(batch_size, n_shots, n_receivers)
                base_loss_per_sample = trace_loss.mean(dim=(1, 2)) / (
                    trace_normalizer.mean(dim=(1, 2)) + 1e-9
                )
            else:
                raise ValueError("Unsupported loss type. Use 'mse' or 'w2'.")
            
            loss_per_sample = base_loss_per_sample
            # Sum rather than mean so each batch member receives the same
            # gradient magnitude as an equivalent single-sample run.
            loss = loss_per_sample.sum()

            latents_grad_x0 = torch.autograd.grad(loss, x0_pred, retain_graph=True)[0]
            latents_grad = torch.autograd.grad(loss, latent_model_input)[0]
            current_rho = rho * grad_decay
            c = 0.1
            gradient_max = torch.amax(
                abs(latents_grad_x0), dim=(1, 2, 3), keepdim=True
            )
            change_rate = (
                (gradient_max + rate_down) / (abs(latents_grad_x0) + rate_down)
            ) ** rate_power


            # TV controls the update magnitude independently of the data loss.
            tv_damping = torch.exp(-_tv_per_sample(x0_models) / c).view(-1, 1, 1, 1)
            latents = latents - current_rho * change_rate * latents_grad * tv_damping
            latents = latents.detach()
            
        if x_true is not None:
            with torch.no_grad():
                latent_models = _crop_model_batch(latents)
                x0_models = _crop_model_batch(x0_pred)
                denominator = torch.linalg.vector_norm(
                    (x_true - PHYSICS["velocity_offset"]).flatten(1), dim=1
                )
                error = torch.linalg.vector_norm(
                    (latent_models - x_true).flatten(1), dim=1
                ) / denominator
                error_0 = torch.linalg.vector_norm(
                    (x0_models - x_true).flatten(1), dim=1
                ) / denominator
            if i == 0:
                print(
                    f"Initial mean relative error: {error.mean().item():.4f}, "
                    f"Initial mean x0 relative error: {error_0.mean().item():.4f}"
                )
            error_history.append(error.detach().cpu().tolist())
            error_0_history.append(error_0.detach().cpu().tolist())
            if rho > 0:
                loss_history.append(loss_per_sample.detach().cpu().tolist())
            else:
                loss_history.append([0.0] * batch_size)
                
            progress_bar.set_postfix(
                error=error.mean().item(),
                error_0=error_0.mean().item(),
                loss=loss_per_sample.mean().item() if rho > 0 else 0,
                current_rho=current_rho if rho > 0 else 0,
            )
            
    return latents, error_history, error_0_history, loss_history


def plot_real_and_pred(x_true, x_pred, path, rho=0, grad_decay=1.0):
    os.makedirs(path, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(10, 5))
    
    # 获取数据并计算统一的颜色范围
    true_data = x_true.detach().cpu().numpy()
    pred_data = x_pred.detach().cpu().numpy()
    diff_data = true_data - pred_data
    
    # 计算true和pred数据的全局最小值和最大值
    vmin = min(np.min(true_data), np.min(pred_data))
    vmax = max(np.max(true_data), np.max(pred_data))
    
    # 计算差值图的颜色范围
    diff_vmax = max(abs(np.min(diff_data)), abs(np.max(diff_data)))
    diff_vmin = -diff_vmax
    
    # 使用相同的颜色范围绘制true和pred图
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
    plt.savefig(os.path.join(path, f"real_and_pred_rho_{rho}_{grad_decay}.png"), dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close()
    
def plot_errors(error_history, error_0_history, path, rho=0, grad_decay=1.0):
    os.makedirs(path, exist_ok=True)
    plt.figure(figsize=(10, 6))
    steps = range(len(error_history))
    plt.plot(steps, error_history, label='Error', color='blue')
    plt.plot(steps, error_0_history, label='Error_0', color='red')
    plt.xlabel('Step')
    plt.ylabel('Error Value')
    plt.title(f'Error and Error_0 over Steps (rho={rho})')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(path, f"errors_rho_{rho}_{grad_decay}.png"), dpi=300, bbox_inches="tight")
    plt.close()

def plot_loss(loss_history, path, rho=0, grad_decay=1.0):
    """
    绘制loss的曲线图
    """
    os.makedirs(path, exist_ok=True)
    plt.figure(figsize=(10, 6))
    steps = range(len(loss_history))
    plt.plot(steps, loss_history, label='Loss', color='green')
    plt.xlabel('Step')
    plt.ylabel('Loss Value')
    plt.title(f'Loss over Steps (rho={rho})')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(path, f"loss_rho_{rho}_{grad_decay}.png"), dpi=300, bbox_inches="tight")
    plt.close()


def save_latents_pt(x, path, rho, grad_decay):
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(x, os.path.join(path, f"results_{rho}_{grad_decay}.pt"))
