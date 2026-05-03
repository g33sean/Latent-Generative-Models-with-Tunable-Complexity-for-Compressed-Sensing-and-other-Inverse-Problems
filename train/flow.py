#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Latent flow-matching trainer for unconditional models (e.g. FFHQ).

Not intended for class-conditional training.
"""

import argparse
from pathlib import Path
import torch
import torch.nn.functional as F

from train.base_ldm_trainer import BaseLDMTrainer


def sample_timesteps_logit_normal(
    batch_size: int,
    device: torch.device,
    mean: float = 0.0,
    std: float = 1.0,
) -> torch.Tensor:
    """Sample timesteps in (0, 1) from a logit-normal distribution."""
    u = torch.randn(batch_size, device=device) * std + mean
    return torch.sigmoid(u).clamp(1e-5, 1.0 - 1e-5)


def sample_timesteps_uniform(batch_size: int, device: torch.device) -> torch.Tensor:
    """Sample timesteps uniformly from (0, 1)."""
    return torch.rand(batch_size, device=device).clamp(1e-5, 1.0 - 1e-5)


def flow_forward(
    x0: torch.Tensor, t: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the linear flow forward process and return velocity targets."""
    epsilon = torch.randn_like(x0)
    t_bc = t[:, None, None, None]
    x_t = (1.0 - t_bc) * x0 + t_bc * epsilon
    velocity = epsilon - x0
    return x_t, velocity, epsilon


def flow_matching_loss_weight(t: torch.Tensor, weighting: str = "uniform") -> torch.Tensor:
    """Compute per-sample flow-matching loss weights."""
    if weighting == "snr_like":
        gamma = 5.0
        snr = ((1.0 - t) / t) ** 2
        return torch.minimum(snr, gamma * torch.ones_like(snr)) / snr
    if weighting == "sigma":
        return t
    return torch.ones_like(t)


class FlowMatchingLDMTrainer(BaseLDMTrainer):
    """Latent flow-matching trainer: Euler ODE with optional nested dropout."""

    _script_path = Path(__file__)

    def _load_config(self):
        super()._load_config()
        self.num_sampling_steps = self.tr_cfg.get("num_sampling_steps", 50)
        self.timestep_sampling  = self.tr_cfg.get("timestep_sampling", "logit_normal")
        self.logit_normal_mean  = self.tr_cfg.get("logit_normal_mean", 0.0)
        self.logit_normal_std   = self.tr_cfg.get("logit_normal_std", 1.0)
        self.loss_weighting     = self.tr_cfg.get("loss_weighting", "uniform")
        self.shift_factor       = self.tr_cfg.get("shift_factor", 1.0)

    def _post_prepare_setup(self):
        pass

    def apply_time_shift(self, t: torch.Tensor) -> torch.Tensor:
        """Apply SD3-style time shifting; identity when shift_factor=1."""
        if self.shift_factor == 1.0:
            return t
        return self.shift_factor * t / (1.0 + (self.shift_factor - 1.0) * t)

    def _compute_loss(self, latents, labels):
        bs = latents.size(0)
        if self.timestep_sampling == "logit_normal":
            t = sample_timesteps_logit_normal(
                bs, self.device, mean=self.logit_normal_mean, std=self.logit_normal_std
            )
        else:
            t = sample_timesteps_uniform(bs, self.device)
        t = self.apply_time_shift(t)

        x_t, velocity_target, _ = flow_forward(latents, t)
        weights = flow_matching_loss_weight(t, self.loss_weighting)

        if self.tr_cfg.get("nd"):
            x_t_nd  = self.nd(x_t)
            pred    = self.model(sample=x_t, timestep=t).sample
            pred_nd = self.model(sample=x_t_nd, timestep=t).sample
            loss_n  = F.mse_loss(pred, velocity_target, reduction="none").mean(dim=list(range(1, pred.ndim)))
            loss_d  = F.mse_loss(pred_nd, velocity_target, reduction="none").mean(dim=list(range(1, pred_nd.ndim)))
            return (1 - self.alpha) * (weights * loss_n).mean() + self.alpha * (weights * loss_d).mean()
        else:
            pred = self.model(sample=x_t, timestep=t).sample
            loss = F.mse_loss(pred, velocity_target, reduction="none").mean(dim=list(range(1, pred.ndim)))
            return (weights * loss).mean()

    def _val_loss_for_batch(self, vz, labels):
        bs = vz.size(0)
        vt = self.apply_time_shift(sample_timesteps_uniform(bs, self.device))
        x_t, velocity_target, _ = flow_forward(vz, vt)
        vp = self.model(sample=x_t, timestep=vt).sample
        return F.mse_loss(vp, velocity_target).item()

    def generate_images(self, latent_dummy, vae, unet, batch_size, device, num_steps=None):
        """Generate samples by Euler integration of the learned flow field."""
        if num_steps is None:
            num_steps = self.num_sampling_steps
        vae.eval()
        unet.eval()
        latent_shape = latent_dummy.shape[1:]
        x_t = torch.randn((batch_size, *latent_shape), device=device)
        timesteps = self.apply_time_shift(
            torch.linspace(1.0, 0.0, num_steps + 1, device=device)
        )
        for i in range(num_steps):
            t_curr, t_next = timesteps[i], timesteps[i + 1]
            dt = t_next - t_curr
            t_batch = torch.full((batch_size,), t_curr.item(), device=device)
            with torch.no_grad():
                v_pred = unet(sample=x_t, timestep=t_batch).sample
            x_t = x_t + v_pred * dt
        images = self.decode_images(x_t)
        return ((images + 1) / 2).clamp(0, 1)

    def _generate_for_fid(self, bs, labels=None):
        return self.generate_images(self.z_dummy, self.vae, self.model, bs, self.device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train Latent Flow Matching Model (UNet stage)")
    parser.add_argument("--config", "-c", dest="config_path", required=True,
                        help="Path to YAML config")
    parser.add_argument("--vae_ckpt_path", "-v", required=True,
                        help="Pre-trained VAE checkpoint")
    parser.add_argument("--gpus", "-g", default=None,
                        help="Comma-separated GPU ids (optional)")
    parser.add_argument("--resume_path", "-r", default=None,
                        help="Path to specific checkpoint to resume from (optional)")
    cli_args = parser.parse_args()
    trainer = FlowMatchingLDMTrainer(cli_args)
    trainer.train()
