import torch

def extract(x, t, shape):
    return x[t].reshape(-1, *((1,) * (len(shape) - 1)))

def make_beta_schedule(schedule, timesteps, dtype=torch.float32, **kwargs):
    if schedule == "linear":
        beta_start, beta_end = kwargs.get('start', 1e-4), kwargs.get('end', 2e-2)
        betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, timesteps, dtype=dtype) ** 2 
    elif schedule == "cosine":
        steps = timesteps + 1
        s = kwargs.get('s', 8e-3)
        t = torch.linspace(0, timesteps, steps, dtype=dtype) / timesteps
        alphas_cumprod = torch.cos((t + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = torch.clip(1 - (alphas_cumprod[1:] / alphas_cumprod[:-1]), 0, 0.999)
    elif schedule == "sqrt":
        beta_start, beta_end = kwargs.get('start', 1e-4), kwargs.get('end', 2e-2)
        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=dtype)
    elif schedule == "sqrt":
        beta_start, beta_end = kwargs.get('start', 1e-4), kwargs.get('end', 2e-2)
        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=dtype) ** 0.5
    else:
        raise ValueError(f"Unsupported schedule: {schedule}!")
    return betas

def modulate(x, scale, shift):
    return (scale + 1) * x + shift
