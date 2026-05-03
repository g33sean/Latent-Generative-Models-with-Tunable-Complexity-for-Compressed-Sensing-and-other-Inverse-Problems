#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Class-conditional DDPM latent diffusion trainer with classifier-free guidance (ImageNet)."""

import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from diffusers import DDPMScheduler, DPMSolverMultistepScheduler

from train.base_ldm_trainer import BaseLDMTrainer


class ImageNetLDMTrainer(BaseLDMTrainer):
    """Class-conditional latent diffusion trainer with classifier-free guidance."""

    _script_path = Path(__file__)

    def _load_config(self):
        super()._load_config()
        self.denoise_steps    = self.tr_cfg["denoising_timesteps"]
        self.gamma            = self.tr_cfg.get("snr_gamma", 5.0)
        self.num_classes      = self.ds_cfg.get("num_classes", 1000)
        self.null_class_id    = self.num_classes
        self.cond_drop_prob   = self.tr_cfg.get("cond_drop_prob", 0.1)
        self.guidance_scale   = self.tr_cfg.get("guidance_scale", 3.0)
        self.fid_sample_steps = self.tr_cfg.get("fid_sample_steps", 100)

    def _make_optimizer(self):
        return AdamW(self.model.parameters(), lr=self.lr, betas=(0.9, 0.99), weight_decay=0.0)

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

    def _unpack_batch(self, batch):
        images, labels = batch
        return images, labels

    def _prepare_class_labels(self, labels, *, drop_labels=False):
        class_labels = labels.to(self.device, dtype=torch.long)
        if drop_labels:
            drop_mask = torch.rand(class_labels.shape[0], device=self.device) < self.cond_drop_prob
            class_labels = class_labels.clone()
            class_labels[drop_mask] = self.null_class_id
        return class_labels

    def _model_predict(self, noisy, timesteps, class_labels=None):
        model_kwargs = {"sample": noisy, "timestep": timesteps}
        if class_labels is not None:
            model_kwargs["class_labels"] = class_labels
        return self.model(**model_kwargs).sample

    def _compute_loss(self, latents, labels):
        class_labels = self._prepare_class_labels(labels, drop_labels=True)
        noise = torch.randn_like(latents)
        t = torch.randint(0, self.denoise_steps, (latents.size(0),), device=self.device)
        noisy = self.noise_sched.add_noise(latents, noise, t)

        if self.tr_cfg.get("nd"):
            noisy_nd = self.nd(noisy)
            pred    = self._model_predict(noisy, t, class_labels)
            pred_nd = self._model_predict(noisy_nd, t, class_labels)
            loss_n  = F.mse_loss(pred, noise, reduction="mean")
            loss_d  = F.mse_loss(pred_nd, noise, reduction="mean")
            return (1 - self.alpha) * loss_n + self.alpha * loss_d
        else:
            pred = self._model_predict(noisy, t, class_labels)
            return F.mse_loss(pred, noise, reduction="mean")

    def _val_loss_for_batch(self, vz, labels):
        val_class_labels = self._prepare_class_labels(labels, drop_labels=False)
        vnoise = torch.randn_like(vz)
        vt = torch.randint(0, self.denoise_steps, (vz.size(0),), device=self.device)
        noisy_vz = self.noise_sched.add_noise(vz, vnoise, vt)
        vp = self._model_predict(noisy_vz, vt, val_class_labels)
        return F.mse_loss(vp, vnoise).item()

    def generate_images(self, latent_dummy, vae, unet, noise_scheduler,
                        batch_size, device, class_labels=None):
        """Sample batch_size images with classifier-free guidance."""
        vae.eval()
        unet.eval()
        latent_shape = latent_dummy.shape[1:]
        latents = torch.randn((batch_size, *latent_shape), device=device)
        if class_labels is None:
            class_labels = torch.arange(batch_size, device=device, dtype=torch.long) % self.num_classes
        noise_scheduler.set_timesteps(self.fid_sample_steps, device=device)
        for t in noise_scheduler.timesteps:
            with torch.no_grad():
                timestep_batch = torch.full((batch_size,), int(t), device=device, dtype=torch.long)
                null_labels       = torch.full_like(class_labels, self.null_class_id)
                noise_pred_uncond = unet(sample=latents, timestep=timestep_batch, class_labels=null_labels).sample
                noise_pred_cond   = unet(sample=latents, timestep=timestep_batch, class_labels=class_labels).sample
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)
                latents = noise_scheduler.step(noise_pred, t, latents).prev_sample
        images = self.decode_images(latents)
        return ((images + 1) / 2).clamp(0, 1)

    def _generate_for_fid(self, bs, labels=None):
        if labels is not None:
            labels = labels.to(self.device)
        return self.generate_images(
            self.z_dummy, self.vae, self.model, self.noise_sched_sample,
            bs, self.device, class_labels=labels,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train Class-Conditional Latent Diffusion Model (ImageNet)")
    parser.add_argument("--config", "-c", dest="config_path", required=True,
                        help="Path to YAML config")
    parser.add_argument("--vae_ckpt_path", "-v", required=True,
                        help="Pre-trained VAE checkpoint")
    parser.add_argument("--gpus", "-g", default=None,
                        help="Comma-separated GPU ids (optional)")
    parser.add_argument("--resume_path", "-r", default=None,
                        help="Path to specific checkpoint to resume from (optional)")
    cli_args = parser.parse_args()
    trainer = ImageNetLDMTrainer(cli_args)
    trainer.train()
