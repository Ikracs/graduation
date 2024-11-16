from tqdm import tqdm

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from driving.audio2video.utils import extract, make_beta_schedule


class PMDiffusion(nn.Module):
    '''
    Pose-Motion Diffusion.
    '''
    def __init__(
        self,
        model,
        timesteps=1000,
        beta_schedule='linear',
        objective='pred_x',
        loss_type='l2',
        use_p2=False,
        use_vel=True,
        use_acc=True,
        guidance=2,
        cond_drop_prob=0.1,
    ):

        super(PMDiffusion, self).__init__()

        self.model = model

        self.timesteps = timesteps
        self.objective = objective
        self.use_vel = use_vel
        self.use_acc = use_acc
        self.guidance = guidance
        self.cond_drop_prob = cond_drop_prob

        betas = make_beta_schedule(schedule=beta_schedule, timesteps=timesteps)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)

        self.register_buffer("posterior_log_variance_clipped", torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer("posterior_mean_coef1", betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer("posterior_mean_coef2", (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))

        p2_loss_weight_gamma = 0.5 if use_p2 else 0
        self.register_buffer("p2_loss_weight", (1 + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma)

        self.criterion = F.mse_loss if loss_type == "l2" else F.l1_loss

    def predict_start_from_noise(self, x_t, t, noise):
        return extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise

    def predict_noise_from_start(self, x_t, t, x0):
        return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = extract(self.posterior_mean_coef1, t, x_t.shape) * x_start + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        return posterior_mean, extract(self.posterior_variance, t, x_t.shape), extract(self.posterior_log_variance_clipped, t, x_t.shape)

    def model_predictions(self, x, cond, t, padding_mask, clip_x_start=False, guidance=None):
        guidance = guidance if guidance is not None else self.guidance
        output = self.model.guided_forward(x, cond, t, padding_mask, guidance)

        if self.objective == 'pred_noise':
            pred_noise, x_start = output, self.predict_start_from_noise(x, t, output)
        elif self.objective == 'pred_x':
            pred_noise, x_start = self.predict_noise_from_start(x, t, output), output
        else:
            raise NotImplementedError(f'Unknown objective: {self.objective}!')

        x_start = torch.clamp(x_start, min=-1.0, max=1.0) if clip_x_start else x_start

        return pred_noise, x_start

    def q_sample(self, x_start, t, noise):
        return extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise

    def p_loss_help(self, x, y, t):
        loss = self.criterion(x, y, reduction='none').flatten(1).mean(dim=-1)
        return loss * extract(self.p2_loss_weight, t, loss.shape)

    def p_loss(self, x_start, cond, t, padding_mask):
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)

        pred_x = self.model(x_noisy, cond, t, padding_mask, cond_drop_prob=self.cond_drop_prob)

        loss_values = {'recon': self.p_loss_help(pred_x, x_start, t).mean()}

        if self.use_vel:
            x_vel, y_vel = pred_x[:, 1:] - pred_x[:, :-1], x_start[:, 1:] - x_start[:, :-1]
            loss_values.update({'vel': self.p_loss_help(x_vel, y_vel, t).mean()})

            if self.use_acc:
                x_acc, y_acc = x_vel[:, 1:] - x_vel[:, :-1], y_vel[:, 1:] - y_vel[:, :-1]
                loss_values.update({'acc': self.p_loss_help(x_acc, y_acc, t).mean()})

        return loss_values

    def forward(self, x, cond, padding_mask):
        return self.p_loss(x, cond, torch.randint(0, self.timesteps, (x.shape[0],), device=x.device), padding_mask)

    def p_mean_variance(self, x, cond, t, padding_mask):
        _, x_start = self.model_predictions(x, cond, t, padding_mask)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start, x, t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.inference_mode()
    def p_sample(self, shape, cond, padding_mask):
        times = range(self.timesteps - 1, -1, -1)
        x = torch.randn(shape, device=cond.device)

        for time in tqdm(times, desc='sampling loop', disable=False):
            t = torch.full((shape[0],), time, device=cond.device, dtype=torch.int64)
            model_mean, _, model_log_variance, _ = self.p_mean_variance(x, cond, t, padding_mask)
            noise = torch.randn_like(x) if time > 0 else 0.0
            x = model_mean + (0.5 * model_log_variance).exp() * noise

        return x

    @torch.inference_mode()
    def ddim_sample(self, shape, cond, padding_mask, sampling_timesteps=50, eta=1):
        times = np.linspace(self.timesteps - 1, -1, num=sampling_timesteps - 1).astype(np.int64).tolist()
        time_pairs = np.stack((times[:-1], times[1:])).transpose().tolist()

        x = torch.randn(shape, device=cond.device)

        for time, time_next in tqdm(time_pairs, desc='sampling loop', disable=False):
            t = torch.full((shape[0],), time, device=cond.device, dtype=torch.int64)
            pred_noise, x_start = self.model_predictions(x, cond, t, padding_mask)

            if time_next < 0:
                x = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            x = x_start * alpha_next.sqrt() + c * pred_noise + sigma * torch.randn_like(x)

        return x
