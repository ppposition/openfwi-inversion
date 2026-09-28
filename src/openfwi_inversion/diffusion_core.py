import torch
from tqdm.auto import tqdm
from diffusers import DDPMScheduler, UNet2DModel
import os
import numpy as np
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

def cumsum_error(x, y):
    """计算两个信号积分后的均方误差，仿照W2损失函数添加归一化"""
    # 获取输入形状
    original_shape = x.shape
    
    # 将输入重塑为二维以便批处理
    if x.dim() > 1:
        x_reshaped = x.reshape(-1, x.shape[-1])
        y_reshaped = y.reshape(-1, y.shape[-1])
    else:
        x_reshaped = x.unsqueeze(0)
        y_reshaped = y.unsqueeze(0)
    
    batch_size = x_reshaped.shape[0]
    n_points = x_reshaped.shape[1]
    
    # 计算每个批次的最小值
    min_vals = torch.min(y_reshaped, dim=-1, keepdim=True)[0]
    
    # 确保非负，并减去最小值的1.1倍
    non_negative_x = x_reshaped - 1.1 * min_vals
    non_negative_y = y_reshaped - 1.1 * min_vals
    
    # 计算积分（使用梯形法则）
    def _integrate(signal):
        # signal shape: (batch_size, n_points)
        # 使用简单梯形法则计算积分
        integral = torch.sum(signal, dim=-1)
        return integral
    
    '''# 计算每个信号的总积分
    total_integral_x = _integrate(non_negative_x)
    total_integral_y = _integrate(non_negative_y)
    
    # 归一化，添加小量避免除以零
    normalized_x = non_negative_x / (total_integral_x.unsqueeze(-1) + 1e-9)
    normalized_y = non_negative_y / (total_integral_y.unsqueeze(-1) + 1e-9)
    '''    
    # 计算累积和
    int_x = torch.cumsum(non_negative_x, dim=-1)
    int_y = torch.cumsum(non_negative_y, dim=-1)
    
    # 计算均方误差
    mse = torch.linalg.norm(int_x - int_y)
    
    return mse

def range_constraint_loss(x):
    """
    计算范围约束损失: L_range = (1/N) * sum((x_i - clamp(x_i, -1, 1))^2)
    这是一个软约束，鼓励x的值保持在[-1, 1]范围内
    
    Args:
        x (torch.Tensor): 输入张量
        
    Returns:
        torch.Tensor: 范围约束损失
    """
    clamped_x = torch.clamp(x, -1, 1)
    loss = torch.mean((x - clamped_x) ** 2)
    return loss

def tv_regularization(x):
    """
    计算全变分(Total Variation)正则化项
    TV正则化鼓励图像平滑，减少噪声
    
    Args:
        x (torch.Tensor): 输入张量，形状为 (H, W) 或 (C, H, W)
        
    Returns:
        torch.Tensor: TV正则化损失
    """
    if x.dim() == 2:
        # 2D输入: (H, W)
        tv_h = torch.mean(torch.abs(x[1:, :] - x[:-1, :]))
        tv_w = torch.mean(torch.abs(x[:, 1:] - x[:, :-1]))
    elif x.dim() == 3:
        # 3D输入: (C, H, W)
        tv_h = torch.mean(torch.abs(x[:, 1:, :] - x[:, :-1, :]))
        tv_w = torch.mean(torch.abs(x[:, :, 1:] - x[:, :, :-1]))
    else:
        raise ValueError(f"Unsupported input dimension: {x.dim()}")
    
    return tv_h + tv_w

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

def agc_gain_control(wavelet, window_size=50, return_rms=False):
    """
    自动增益控制(AGC)算法 - 向量化实现
    对每个道应用滑动窗口的增益控制，使能量均衡
    
    Args:
        wavelet (torch.Tensor): 输入波场数据，形状为 (n_shots, n_receivers, n_timesteps)
        window_size (int): 滑动窗口大小
        return_rms (bool): 是否返回RMS值
        
    Returns:
        torch.Tensor: 应用AGC后的波场数据
        torch.Tensor (可选): RMS值，形状为 (n_shots, n_receivers, n_timesteps)
    """
    # 获取输入形状
    n_shots, n_receivers, n_timesteps = wavelet.shape
    
    # 创建输出张量
    agc_wavelet = torch.zeros_like(wavelet)
    
    # 将输入重塑为二维以便批处理 (n_shots*n_receivers, n_timesteps)
    wavelet_2d = wavelet.reshape(-1, n_timesteps)
    
    # 创建一个张量来存储RMS值
    rms_values = torch.zeros_like(wavelet_2d)
    # 计算每个时间点的RMS值
    for t in range(n_timesteps):
        # 确定窗口范围
        start = max(0, t - window_size // 2)
        end = min(n_timesteps, t + window_size // 2)
        
        # 计算所有道在当前时间点的窗口内的RMS值
        window = wavelet_2d[:, start:end]
        rms = torch.sqrt(torch.mean(window ** 2, dim=1)) + 1e-9  # 添加小量避免除以零
        rms_values[:, t] = rms
    
    # 应用增益控制
    agc_wavelet_2d = wavelet_2d / rms_values
    
    # 恢复原始形状
    agc_wavelet = agc_wavelet_2d.reshape(n_shots, n_receivers, n_timesteps)
    rms_values = rms_values.reshape(n_shots, n_receivers, n_timesteps)
    
    if return_rms:
        return agc_wavelet, rms_values
    else:
        return agc_wavelet

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


def receiver(v, device, shot_num, source_locations='up', receiver_locations='up', noise_std=0.05):
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
        receiver_amplitudes = receiver_amplitudes + torch.randn_like(receiver_amplitudes) * noise_std
    return (
        receiver_amplitudes,
        source_locations.reshape(batch_size, n_shots, n_sources_per_shot, 2),
        receiver_locations.reshape(batch_size, n_shots, n_receivers_per_shot, 2),
    )

def sample(scheduler, unet, npy_size, batch_size=1, rho=0, rate_down=1, rate_power=1, save_x0_steps=False, x_true=None, loss_type='mse', grad_decay=1.0, lam_range=0.1, lam_tv=0.01, shot_num=5, source_locations='up', receiver_locations='up', warmup_steps=0, k=100, rms_threshold=0.01, agc_window_size=50, seed=1, noise_std=0.05):
    
    unet.eval()
    for param in unet.parameters():
        param.requires_grad = False
    generator = torch.Generator(device).manual_seed(seed)
    latents = torch.randn(
        (batch_size, 1, npy_size, npy_size),
        generator=generator,
        device=device,
        dtype=unet.dtype
    )
    
    if save_x0_steps:
        x0_predictions = []
        
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
        wave_true, source_locs_true, receiver_locs_true = receiver(x_true, device, shot_num=shot_num, source_locations=source_locations, receiver_locations=receiver_locations, noise_std=noise_std)
        # 使用AGC算法处理观测数据，并获取RMS值（但不应用AGC）
        
        # _, rms_values = agc_gain_control(wave_true, window_size=agc_window_size, return_rms=True)
        
        rms_values = wave_true
        rms_max = torch.amax(abs(wave_true), dim=(1, 2, 3), keepdim=True)

        # 计算权重：(1/rms_values)^weight_power，当rms_values小于阈值时设为1
        '''weights = torch.ones_like(rms_values)
        mask = rms_values > rms_threshold
        weights[mask] = (1.0 / rms_values[mask]) ** weight_power'''

        weights = 1/(k*abs(rms_values)/rms_max+1)
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
        
        # 计算当前步长：warm-up阶段线性增长，之后保持定值
        if warmup_steps > 0 and i < warmup_steps:
            # warm-up阶段：步长从0线性增长到rho
            current_rho = rho * (i + 1) / warmup_steps
        else:
            # warm-up结束后：保持定值rho
            current_rho = rho
        # 尝试使用scheduler返回的pred_original_sample，如果不支持则手动计算
        if hasattr(scheduler_output, 'pred_original_sample') and scheduler_output.pred_original_sample is not None:
            # 使用scheduler返回的pred_original_sample作为x_0预测
            x0_pred = scheduler_output.pred_original_sample
        else:
            # 手动计算x_0预测
            # 根据DDPM的公式: x_0 = (x_t - sqrt(1-alpha_t) * epsilon) / sqrt(alpha_t)
            # 其中epsilon是噪声预测，alpha_t是当前时间步的alpha值
            alpha_t = scheduler.alphas_cumprod[t_step]
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_t)

            x0_pred = (latents - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t
        
        if rho > 0:
            # 根据选择的损失类型计算损失
            x0_models = _crop_model_batch(x0_pred)
            wave_pred, source_locs_pred, receiver_locs_pred = receiver(x0_models, device, shot_num=shot_num, source_locations=source_locations, receiver_locations=receiver_locations, noise_std=0)
            # 不对合成波场做AGC，直接将观测波场的权重应用到合成波场
            wave_pred = wave_pred * weights

            # 计算基础损失（不再使用 mute 后的波场）
            if loss_type == 'mse':
                base_loss_per_sample = torch.mean(
                    (wave_pred - wave_true) ** 2, dim=(1, 2, 3)
                )
            elif loss_type == 'cumsum':
                min_values = torch.amin(wave_true, dim=-1, keepdim=True)
                pred_cumsum = torch.cumsum(wave_pred - 1.1 * min_values, dim=-1)
                true_cumsum = torch.cumsum(wave_true - 1.1 * min_values, dim=-1)
                base_loss_per_sample = torch.linalg.vector_norm(
                    (pred_cumsum - true_cumsum).flatten(1), dim=1
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
                raise ValueError("Unsupported loss type. Use 'mse', 'cumsum', or 'w2'.")
            
            # 添加范围约束损失
            clamped_models = torch.clamp(x0_models, -1, 1)
            range_loss_per_sample = torch.mean(
                (x0_models - clamped_models) ** 2, dim=(1, 2)
            )
            
            # 添加TV正则化损失
            tv_loss_per_sample = _tv_per_sample(x0_models)
            
            # 总损失 = 基础损失 + 范围约束损失 + TV正则化损失
            loss_per_sample = (
                base_loss_per_sample
                + lam_range * range_loss_per_sample
                + lam_tv * tv_loss_per_sample
            )
            # Sum rather than mean so each batch member receives the same
            # gradient magnitude as an equivalent single-sample run.
            loss = loss_per_sample.sum()

            latents_grad_x0 = torch.autograd.grad(loss, x0_pred, retain_graph=True)[0]
            latents_grad = torch.autograd.grad(loss, latent_model_input)[0]
            # 使用current_rho而不是rho，以便实现warmup
            current_rho = current_rho * grad_decay
            c = 0.1
            gradient_max = torch.amax(
                abs(latents_grad_x0), dim=(1, 2, 3), keepdim=True
            )
            change_rate = (
                (gradient_max + rate_down) / (abs(latents_grad_x0) + rate_down)
            ) ** rate_power


            tv_damping = torch.exp(-tv_loss_per_sample / c).view(-1, 1, 1, 1)
            latents = latents - current_rho * change_rate * latents_grad * tv_damping
            latents = latents.detach()
            
        if save_x0_steps:
            x0_predictions.append(x0_pred.detach().cpu().numpy())
        
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
            
    if save_x0_steps:
        return latents, x0_predictions, error_history, error_0_history, loss_history
    return latents, None, error_history, error_0_history, loss_history

def plot_npy(latents, path, rows=4, cols=4, rho=0, grad_decay=1.0):
    os.makedirs(path, exist_ok=True)
    fig, axes = plt.subplots(rows, cols, figsize=(16, 16))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        im = ax.imshow(latents[i, 0].detach().cpu().numpy())
        ax.axis("off")
        fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(path, f"sample_rho_{rho}_{grad_decay}.png"), dpi=300, bbox_inches="tight", pad_inches=0.1)

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
    
def plot_npy_steps(x0_predictions, path, rows=4, cols=4, rho=0, grad_decay=1.0):
    os.makedirs(path, exist_ok=True)
    num_steps = len(x0_predictions)
    if num_steps > rows * cols:
        jump = (num_steps // (rows * cols)) + 1
        x0_predictions = [x0_predictions[i] for i in range(0, num_steps, jump)]
    num_steps = len(x0_predictions)
    if len(x0_predictions[0].shape)==2:
        fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
        for step in range(num_steps):
            ax = axes[step // cols, step % cols]
            im = ax.imshow(x0_predictions[step])
            ax.axis("off")
            fig.colorbar(im, ax=ax)
        for step in range(num_steps, rows * cols):
            ax = axes[step // cols, step % cols]
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(path, f"x0_step_{step}_rho_{rho}_{grad_decay}.png"), dpi=300, bbox_inches="tight", pad_inches=0.1)
        plt.close()
    else:
        for batch_idx in range(len(x0_predictions[0])):
            fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
            for step in range(num_steps):
                ax = axes[step // cols, step % cols]
                im = ax.imshow(np.squeeze(x0_predictions[step][batch_idx]))
                ax.axis("off")
                fig.colorbar(im, ax=ax)
            for step in range(num_steps, rows * cols):
                ax = axes[step // cols, step % cols]
                ax.axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(path, f"batch_{batch_idx}_x0_step_{step}_rho_{rho}_{grad_decay}.png"), dpi=300, bbox_inches="tight", pad_inches=0.1)
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

def save_results_to_file(error, error_0, loss, loss_type, rho, grad_decay, lam_range, lam_tv, warmup_steps, weight_power, rms_threshold, agc_window_size, filename="result.txt"):
    """
    将error、error_0和loss的最终值保存到文件中（追加模式）
    包含优化算法、误差计算方法和相应参数的信息
    """
    with open(filename, "a") as f:
        f.write(f"误差计算方法: {loss_type}\n")
        f.write(f"梯度下降步长(rho): {rho}\n")
        f.write(f"梯度衰减参数(grad_decay): {grad_decay}\n")
        f.write(f"Warm-up步数: {warmup_steps}\n")
        f.write(f"权重参数: lam_range={lam_range}, lam_tv={lam_tv}\n")
        f.write(f"权重幂次(weight_power): {weight_power}\n")
        f.write(f"RMS阈值(rms_threshold): {rms_threshold}\n")
        f.write(f"AGC窗口大小(agc_window_size): {agc_window_size}\n")
        f.write(f"最终Error: {error}\n")
        f.write(f"最终Error_0: {error_0}\n")
        f.write(f"最终Loss: {loss}\n")
        f.write("-" * 50 + "\n")


def save_sorted_summary(results, filename):
    """将所有已完成实验按最终 Error 从小到大写入一个汇总文件。"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    sorted_results = sorted(results, key=lambda item: item["error"])

    with open(filename, "w") as f:
        f.write(f"实验结果汇总（按最终 Error 从小到大排序，共 {len(sorted_results)} 组）\n")
        f.write("=" * 100 + "\n")
        for rank, result in enumerate(sorted_results, start=1):
            f.write(f"排名: {rank}\n")
            f.write(f"测试样本(ex_num): {result['ex_num']}\n")
            f.write(f"随机种子(seed): {result['seed']}\n")
            f.write(f"误差计算方法: {result['loss_type']}\n")
            f.write(f"炮数(shot_num): {result['shot_num']}\n")
            f.write(f"炮点/接收点位置: {result['source_locations']}/{result['receiver_locations']}\n")
            f.write(f"梯度下降步长(rho): {result['rho']}\n")
            f.write(f"梯度衰减参数(grad_decay): {result['grad_decay']}\n")
            f.write(f"Warm-up步数: {result['warmup_steps']}\n")
            f.write(f"rate_down: {result['rate_down']}\n")
            f.write(f"rate_power: {result['rate_power']}\n")
            f.write(f"lam_range: {result['lam_range']}\n")
            f.write(f"lam_tv: {result['lam_tv']}\n")
            f.write(f"权重参数(k): {result['k']}\n")
            f.write(f"RMS阈值: {result['rms_threshold']}\n")
            f.write(f"AGC窗口大小: {result['agc_window_size']}\n")
            f.write(f"最终Error: {result['error']}\n")
            f.write(f"最终Error_0: {result['error_0']}\n")
            f.write(f"最终Loss: {result['loss']}\n")
            f.write(f"结果目录: {result['target_path']}\n")
            f.write("-" * 100 + "\n")



def plot_wavefield_comparison(wave_true, wave_pred, path, filename_prefix, rho=0, grad_decay=1.0):
    os.makedirs(path, exist_ok=True)
    n_shots = wave_true.shape[0]
    
    fig, axes = plt.subplots(3, n_shots, figsize=(15, 9))
    if n_shots == 1:
        axes = axes.reshape(3, 1)

    # 统一颜色条范围
    vmax = max(torch.max(torch.abs(wave_true)).item(), torch.max(torch.abs(wave_pred)).item())
    vmin = -vmax
    diff_vmax = torch.max(torch.abs(wave_true - wave_pred)).item()
    diff_vmin = -diff_vmax

    # 标题
    row_titles = ["Predicted", "True", "Difference"]

    for i in range(n_shots):
        # Predicted
        im1 = axes[0, i].imshow(wave_pred[i].cpu().detach().numpy().T, cmap='seismic', aspect='auto', vmin=vmin, vmax=vmax)
        fig.colorbar(im1, ax=axes[0, i])
        axes[0, i].set_title(f"Shot {i+1}")
        if i == 0:
            axes[0, i].set_ylabel(row_titles[0])

        # True
        im2 = axes[1, i].imshow(wave_true[i].cpu().detach().numpy().T, cmap='seismic', aspect='auto', vmin=vmin, vmax=vmax)
        fig.colorbar(im2, ax=axes[1, i])
        if i == 0:
            axes[1, i].set_ylabel(row_titles[1])

        # Difference
        im3 = axes[2, i].imshow((wave_true - wave_pred)[i].cpu().detach().numpy().T, cmap='seismic', aspect='auto', vmin=diff_vmin, vmax=diff_vmax)
        fig.colorbar(im3, ax=axes[2, i])
        if i == 0:
            axes[2, i].set_ylabel(row_titles[2])

    plt.tight_layout()
    plt.savefig(os.path.join(path, f"{filename_prefix}_rho_{rho}_{grad_decay}.png"), dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close()

def plot_receiver_waveforms(wave_true, wave_pred, path, filename_prefix, receiver_idx=0, rho=0, grad_decay=1.0):
    """
    绘制指定receiver接收到的所有shot的一维波形图
    将真实值和预测值画到同一个图中，不画差值
    
    Args:
        wave_true: 真实波场数据，形状为 (n_shots, n_receivers, n_timesteps)
        wave_pred: 预测波场数据，形状为 (n_shots, n_receivers, n_timesteps)
        path: 保存路径
        filename_prefix: 文件名前缀
        receiver_idx: 要绘制的receiver索引，默认为0（第一个receiver）
        rho: 优化参数
        grad_decay: 梯度衰减参数
    """
    os.makedirs(path, exist_ok=True)
    n_shots = wave_true.shape[0]
    n_timesteps = wave_true.shape[2]
    
    # 创建子图，每个shot一个子图
    fig, axes = plt.subplots(n_shots, 1, figsize=(12, 3 * n_shots))
    if n_shots == 1:
        axes = [axes]  # 如果只有一个shot，确保axes是列表形式
    
    # 时间轴
    time_axis = np.linspace(0, n_timesteps * PHYSICS["dt"], n_timesteps)
    
    # 统一y轴范围
    y_max = max(torch.max(torch.abs(wave_true[:, receiver_idx, :])).item(),
                torch.max(torch.abs(wave_pred[:, receiver_idx, :])).item())
    
    for i in range(n_shots):
        ax = axes[i]
        
        # 提取当前shot和指定receiver的数据
        true_data = wave_true[i, receiver_idx, :].cpu().detach().numpy()
        pred_data = wave_pred[i, receiver_idx, :].cpu().detach().numpy()
        
        # 绘制波形
        ax.plot(time_axis, true_data, 'b-', label='True', linewidth=1.5)
        ax.plot(time_axis, pred_data, 'r--', label='Predicted', linewidth=1.5)
        
        # 设置图标题和标签
        ax.set_title(f"Shot {i+1}, Receiver {receiver_idx+1}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 设置统一的y轴范围
        ax.set_ylim(-y_max * 1.1, y_max * 1.1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(path, f"{filename_prefix}_receiver_{receiver_idx+1}_rho_{rho}_{grad_decay}.png"),
                dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close()

def plot_receiver_weights(weights, path, filename_prefix, receiver_idx=0, rho=0, grad_decay=1.0):
    """
    绘制指定receiver接收到的所有shot的weights图
    将weights以一维折线图形式展示
    
    Args:
        weights: 权重数据，形状为 (n_shots, n_receivers, n_timesteps)
        path: 保存路径
        filename_prefix: 文件名前缀
        receiver_idx: 要绘制的receiver索引，默认为0（第一个receiver）
        rho: 优化参数
        grad_decay: 梯度衰减参数
    """
    os.makedirs(path, exist_ok=True)
    n_shots = weights.shape[0]
    n_timesteps = weights.shape[2]
    
    # 创建子图，每个shot一个子图
    fig, axes = plt.subplots(n_shots, 1, figsize=(12, 3 * n_shots))
    if n_shots == 1:
        axes = [axes]  # 如果只有一个shot，确保axes是列表形式
    
    # 时间轴
    time_axis = np.linspace(0, n_timesteps * PHYSICS["dt"], n_timesteps)
    
    # 统一y轴范围
    y_min = torch.min(weights).item()
    y_max = torch.max(weights).item()
    
    for i in range(n_shots):
        ax = axes[i]
        
        # 提取当前shot和指定receiver的weights数据
        weights_data = weights[i, receiver_idx, :].cpu().detach().numpy()
        
        # 绘制weights为一维折线图
        ax.plot(time_axis, weights_data, 'g-', linewidth=1.5)
        
        # 设置图标题和标签
        ax.set_title(f"Shot {i+1}, Receiver {receiver_idx+1} Weights")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Weight Value")
        ax.grid(True, alpha=0.3)
        
        # 设置统一的y轴范围
        ax.set_ylim(y_min * 0.9, y_max * 1.1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(path, f"{filename_prefix}_receiver_{receiver_idx+1}_weights_rho_{rho}_{grad_decay}.png"),
                dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close()

def save_latents_pt(x, path, rho, grad_decay):
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(x, os.path.join(path, f"results_{rho}_{grad_decay}.pt"))
