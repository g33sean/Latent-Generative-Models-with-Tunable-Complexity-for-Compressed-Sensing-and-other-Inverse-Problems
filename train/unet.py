#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DDPM latent diffusion trainer (unconditional, SNR-weighted epsilon prediction)."""

import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler, DPMSolverMultistepScheduler
from diffusers.training_utils import compute_snr

from train.base_ldm_trainer import BaseLDMTrainer


class LDMTrainer(BaseLDMTrainer):
    """Latent-Diffusion UNet trainer: SNR-weighted epsilon prediction with optional nested dropout."""

    _script_path = Path(__file__)

    def _load_config(self):
        super()._load_config()
        self.denoise_steps = self.tr_cfg["denoising_timesteps"]
        self.gamma = self.tr_cfg.get("snr_gamma", 5.0)

    def _post_prepare_setup(self):
        self.noise_sched = DDPMScheduler(
            num_train_timesteps=self.denoise_steps,
            beta_schedule="scaled_linear",
            prediction_type="epsilon",
            clip_sample=False,
        )
        self.noise_sched_sample = DPMSolverMultistepScheduler.from_config(
            self.noise_sched.config,
            solver_order=2,
            use_karras_sigmas=True,
            algorithm_type="dpmsolver++",
        )

    def _compute_loss(self, latents, labels):
        noise = torch.randn_like(latents)
        t = torch.randint(0, self.denoise_steps, (latents.size(0),), device=self.device)
        noisy = self.noise_sched.add_noise(latents, noise, t)
        snr = compute_snr(self.noise_sched, t)
        weights = torch.stack([snr, self.gamma * torch.ones_like(t)], dim=1).min(dim=1)[0]
        weights[snr == 0] = 1.0
        w = weights / snr

        if self.tr_cfg.get("nd"):
            noisy_nd = self.nd(noisy)
            pred    = self.model(sample=noisy, timestep=t).sample
            pred_nd = self.model(sample=noisy_nd, timestep=t).sample
            loss_n = F.mse_loss(pred, noise, reduction="none").mean(dim=list(range(1, pred.ndim)))
            loss_d = F.mse_loss(pred_nd, noise, reduction="none").mean(dim=list(range(1, pred_nd.ndim)))
            return (1 - self.alpha) * (w * loss_n).mean() + self.alpha * (w * loss_d).mean()
        else:
            pred = self.model(sample=noisy, timestep=t).sample
            loss = F.mse_loss(pred, noise, reduction="none").mean(dim=list(range(1, pred.ndim)))
            return (w * loss).mean()

    def _val_loss_for_batch(self, vz, labels):
        vnoise = torch.randn_like(vz)
        vt = torch.randint(0, self.denoise_steps, (vz.size(0),), device=self.device)
        noisy_vz = self.noise_sched.add_noise(vz, vnoise, vt)
        vp = self.model(sample=noisy_vz, timestep=vt).sample
        return F.mse_loss(vp, vnoise).item()

    def generate_images(self, latent_dummy, vae, unet, noise_scheduler, batch_size, device):
        """Sample batch_size images via the reverse diffusion process."""
        vae.eval()
        unet.eval()
        latent_shape = latent_dummy.shape[1:]
        latents = torch.randn((batch_size, *latent_shape), device=device)
        noise_scheduler.set_timesteps(100, device=device)
        for t in noise_scheduler.timesteps:
            with torch.no_grad():
                noise_pred = unet(latents, t).sample
                latents = noise_scheduler.step(noise_pred, t, latents).prev_sample
        images = self.decode_images(latents)
        return ((images + 1) / 2).clamp(0, 1)

    def _generate_for_fid(self, bs, labels=None):
        return self.generate_images(
            self.z_dummy, self.vae, self.model, self.noise_sched_sample, bs, self.device
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train Latent Diffusion Model (UNet stage)")
    parser.add_argument("--config", "-c", dest="config_path", required=True,
                        help="Path to YAML config")
    parser.add_argument("--vae_ckpt_path", "-v", required=True,
                        help="Pre-trained VAE checkpoint")
    parser.add_argument("--gpus", "-g", default=None,
                        help="Comma-separated GPU ids (optional)")
    parser.add_argument("--resume_path", "-r", default=None,
                        help="Path to specific checkpoint to resume from (optional)")
    cli_args = parser.parse_args()
    trainer = LDMTrainer(cli_args)
    trainer.train()
