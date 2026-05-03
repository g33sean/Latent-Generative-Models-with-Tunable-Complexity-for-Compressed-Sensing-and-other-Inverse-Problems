#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Latent flow-matching trainer for unconditional runs such as FFHQ.

This trainer does not use class conditioning and is not intended for
ImageNet-style conditional training.
"""

import os
import argparse
import random
import yaml
import shutil
from pathlib import Path
from math import inf
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from PIL import Image
from accelerate import Accelerator
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.training_utils import EMAModel
from cleanfid import fid
from models.unet import UNET_MODELS
from models.vae import VAE_MODELS
from data.dataset_loader import get_train_val_dataloaders
from utils.nd import NestedDropout
from accelerate.utils import DistributedDataParallelKwargs
import glob

def sample_timesteps_logit_normal(
    batch_size: int,
    device: torch.device,
    mean: float = 0.0,
    std: float = 1.0,
) -> torch.Tensor:
    """Sample timesteps in (0, 1) from a logit-normal distribution."""
    u = torch.randn(batch_size, device=device) * std + mean
    t = torch.sigmoid(u)
    return t.clamp(1e-5, 1.0 - 1e-5)

def sample_timesteps_uniform(batch_size: int, device: torch.device) -> torch.Tensor:
    """Sample timesteps uniformly from (0, 1)."""
    return torch.rand(batch_size, device=device).clamp(1e-5, 1.0 - 1e-5)

def flow_forward(x0: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the linear flow forward process and return velocity targets."""
    epsilon = torch.randn_like(x0)
    t_bc = t[:, None, None, None]
    x_t = (1.0 - t_bc) * x0 + t_bc * epsilon
    velocity = epsilon - x0
    return x_t, velocity, epsilon

def flow_matching_loss_weight(t: torch.Tensor, weighting: str = "uniform") -> torch.Tensor:
    """Compute per-sample flow-matching loss weights."""
    if weighting == "uniform":
        return torch.ones_like(t)
    if weighting == "snr_like":
        gamma = 5.0
        snr = ((1.0 - t) / t) ** 2
        return torch.minimum(snr, gamma * torch.ones_like(snr)) / snr
    if weighting == "sigma":
        return t
    return torch.ones_like(t)

def estimate_global_std(vae, dataloader, device):
    vae.eval()
    n, moment_2 = 0, 0.0
    with torch.no_grad():
        for imgs, _ in tqdm(dataloader, desc="Computing VAE scale factor", leave=False):
            z = vae.encode(imgs.to(device)).latent_dist.sample()
            moment_2 += (z ** 2).mean().item() * imgs.size(0)
            n += imgs.size(0)
    return (moment_2 / n) ** 0.5

def count_parameters(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def print_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

def save_images_to_folder(images, folder_path, prefix="img"):
    os.makedirs(folder_path, exist_ok=True)
    images = images.cpu()
    if images.min() < 0:
        images = (images + 1.0) / 2.0
    images_uint8 = (images * 255.0).clamp(0, 255).to(torch.uint8)
    images_np = images_uint8.permute(0, 2, 3, 1).numpy()
    for i, img_arr in enumerate(images_np):
        if img_arr.shape[2] == 1:
            img_arr = img_arr.squeeze(2)
            img_pil = Image.fromarray(img_arr, mode="L")
        else:
            img_pil = Image.fromarray(img_arr, mode="RGB")
        img_path = os.path.join(folder_path, f"{prefix}_{i:06d}.png")
        img_pil.save(img_path)

class FlowMatchingLDMTrainer:
    """Latent flow-matching UNet trainer for unconditional latent models."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self._setup_device()
        self._load_config()
        self._seed_everything()
        self._build_dataloaders()
        self._build_models_and_optim()
        self._setup_monitoring()
        self.global_step = 0

        if self.args.resume_path:
            resume_path = Path(self.args.resume_path)
            if not resume_path.exists():
                raise FileNotFoundError(f"Specified resume_path does not exist: {resume_path}")
        else:
            resume_path = self.task_dir / "resume_checkpoint.pth"

        if resume_path.exists():
            try:
                ckpt = torch.load(resume_path, map_location=self.device)
                self.accelerator.unwrap_model(self.model).load_state_dict(ckpt["model_state"])
                self.ema_model.load_state_dict(ckpt["ema_state"])
                self.optimizer.load_state_dict(ckpt["optimizer_state"])
                self.lr_scheduler.load_state_dict(ckpt["scheduler_state"])
                self.best_mse = ckpt.get("best_mse", self.best_mse)
                self.best_fid = ckpt.get("best_fid", self.best_fid)
                self.global_step = ckpt.get("global_step", 0)
                print(f"Resuming from step {self.global_step}")
                print(f"Resumed best-FID ({self.best_fid:.2f})")
            except Exception as e:
                print(f"Warning: Failed to load {resume_path}: {e}")
                print("  Starting training from scratch...")
                self.global_step = 0

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        self._load_or_compute_global_std()

    def _setup_device(self):
        if self.args.gpus:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.args.gpus
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"Using GPU(s): {self.args.gpus or 'all available'}")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using Apple M-series (MPS)")
        else:
            self.device = torch.device("cpu")
            print("Using CPU – expect slow training!")

    def _load_config(self):
        with Path(self.args.config_path).open() as f:
            cfg = yaml.safe_load(f)
        self.ds_cfg = cfg["dataset_params"]
        self.tr_cfg = cfg["train_params"]

        self.num_sampling_steps = self.tr_cfg.get("num_sampling_steps", 50)
        self.timestep_sampling = self.tr_cfg.get("timestep_sampling", "logit_normal")
        self.logit_normal_mean = self.tr_cfg.get("logit_normal_mean", 0.0)
        self.logit_normal_std = self.tr_cfg.get("logit_normal_std", 1.0)
        self.loss_weighting = self.tr_cfg.get("loss_weighting", "uniform")
        self.shift_factor = self.tr_cfg.get("shift_factor", 1.0)

        self.lr = self.tr_cfg["lr"]
        self.warmup_steps = self.tr_cfg.get("num_warmup_steps", 0)
        self.max_steps = self.tr_cfg["max_steps"]
        self.log_steps = self.tr_cfg["log_steps"]
        self.batch_size = self.tr_cfg.get("batch_size", 64)
        self.val_batch = self.tr_cfg.get("val_batch_size", 64)
        self.acc_steps = self.tr_cfg.get("acc_steps", 2)
        self.val_samples = self.ds_cfg.get("val_num_samples", 0)
        self.fid_steps = self.tr_cfg["fid_steps"]
        self.ckpt_name = self.tr_cfg.get("ldm_ckpt_name", "unet.pt")
        self.alpha = self.tr_cfg.get("alpha", 0.25)

    def _seed_everything(self):
        seed = self.tr_cfg.get("seed", 42)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

    def _build_dataloaders(self):
        self.train_dl, self.val_loader = get_train_val_dataloaders(
            self.ds_cfg,
            train=True,
            train_batch_size=self.batch_size,
            val_batch_size=self.val_batch,
            val_num_samples=self.val_samples,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

    def load_autoencoder(self, ckpt_path: str, model: torch.nn.Module, device: torch.device):
        ckpt = torch.load(ckpt_path, map_location=device)
        state = ckpt.get("model_state", ckpt)
        model.load_state_dict(state)
        return model

    def _build_models_and_optim(self):
        vae_model = VAE_MODELS()
        self.vae = vae_model.create_autoencoder_from_dataset(self.ds_cfg).to(self.device)
        self.vae = self.load_autoencoder(self.args.vae_ckpt_path, self.vae, self.device)
        for param in self.vae.parameters():
            param.requires_grad = False
        self.vae.eval()

        with torch.no_grad():
            dummy = torch.zeros(
                1,
                self.ds_cfg["im_channels"],
                self.ds_cfg.get("im_size", 32),
                self.ds_cfg.get("im_size", 32),
                device=self.device,
            )
            self.z_dummy = self.vae.encode(dummy).latent_dist.sample()

        k = int(np.prod(self.z_dummy.shape[1:]))
        self.nd = NestedDropout(k, self.tr_cfg.get("drop_p", 1e-3), self.device).to(self.device)

        unet_model = UNET_MODELS()
        self.model = unet_model.create_unet_from_dataset(self.ds_cfg)
        print_model_parameters(self.model)

        if getattr(self.model, "enable_gradient_checkpointing", None):
            self.model.enable_gradient_checkpointing()
        elif getattr(self.model, "gradient_checkpointing_enable", None):
            self.model.gradient_checkpointing_enable()

        self.optimizer = AdamW(self.model.parameters(), lr=self.lr)

        ddp_kwargs = DistributedDataParallelKwargs(
            gradient_as_bucket_view=True,
            find_unused_parameters=False,
        )

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.acc_steps,
            kwargs_handlers=[ddp_kwargs],
        )
        self.device = self.accelerator.device
        print(
            f"Using device: {self.device} | "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'None')}"
        )

        self.lr_scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, self.warmup_steps, self.max_steps
        )

        self.model, self.optimizer, self.train_dl, self.val_loader, self.lr_scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dl, self.val_loader, self.lr_scheduler
        )

        self.ema_model = EMAModel(
            self.model.parameters(),
            decay=0.9999,
            use_ema_warmup=True,
            inv_gamma=1.0,
            power=0.75,
            min_decay=0.0,
        )

    def _setup_monitoring(self):
        self.task_dir = Path(self.tr_cfg["task_name"])
        self.task_dir.mkdir(parents=True, exist_ok=True)
        (self.task_dir / "config_used.yaml").write_text(
            yaml.safe_dump({"dataset_params": self.ds_cfg, "train_params": self.tr_cfg})
        )

        script_src = Path(__file__)
        script_dst = self.task_dir / script_src.name
        try:
            shutil.copy(script_src, script_dst)
            print(f"  Archived training script → {script_dst}")
        except Exception as e:
            print(f"  Warning: failed to archive script: {e}")

        self.tb = SummaryWriter(self.task_dir / "tb_logs")
        self.tb.add_scalar(
            "Params/total",
            count_parameters(self.accelerator.unwrap_model(self.model)),
            0,
        )

        self.top_mse_scores = []
        self.top_fid_scores = []
        self.k_best = 3
        self.best_mse = inf
        self.best_fid = inf

        scores_file = self.task_dir / "top_scores.yaml"
        if scores_file.exists():
            with open(scores_file, "r") as f:
                data = yaml.safe_load(f)
            self.top_mse_scores = [(float(s), int(e)) for s, e in data.get("top_mse_scores", [])]
            self.top_fid_scores = [(float(s), int(e)) for s, e in data.get("top_fid_scores", [])]
            if self.top_mse_scores:
                self.best_mse = self.top_mse_scores[0][0]
            if self.top_fid_scores:
                self.best_fid = self.top_fid_scores[0][0]
            print(f"Loaded top scores: MSE={len(self.top_mse_scores)}, FID={len(self.top_fid_scores)}")

        self._top_fid_ckpts = []
        for rank, (score, step) in enumerate(self.top_fid_scores, start=1):
            path = self.task_dir / f"fid_rank_{rank}_{self.ckpt_name}"
            if not path.exists():
                continue
            loc = self.device if rank == 1 else "cpu"
            try:
                ck = torch.load(path, map_location=loc)
                self._top_fid_ckpts.append((score, step, ck))
            except Exception as e:
                print(f"Warning: Failed to load FID checkpoint {path}: {e}")

        self._top_mse_ckpts = []
        for rank, (score, step) in enumerate(self.top_mse_scores, start=1):
            path = self.task_dir / f"mse_rank_{rank}_{self.ckpt_name}"
            if not path.exists():
                continue
            loc = self.device if rank == 1 else "cpu"
            try:
                ck = torch.load(path, map_location=loc)
                self._top_mse_ckpts.append((score, step, ck))
            except Exception as e:
                print(f"Warning: Failed to load MSE checkpoint {path}: {e}")

    def _load_or_compute_global_std(self):
        stats_path = self.task_dir / "vae_stats.yaml"
        if stats_path.exists():
            try:
                with stats_path.open("r") as f:
                    data = yaml.safe_load(f) or {}
                cached_std = data.get("global_std")
                if cached_std is not None:
                    self.global_std = float(cached_std)
                    self.scale = 1.0 / self.global_std
                    print(f"Loaded VAE global std from {stats_path}")
                    return
            except Exception as e:
                print(f"Warning: failed to read {stats_path}: {e}")

        self.global_std = estimate_global_std(self.vae, self.train_dl, self.device)
        self.scale = 1.0 / self.global_std
        try:
            with stats_path.open("w") as f:
                yaml.safe_dump({"global_std": float(self.global_std)}, f)
            print(f"Saved VAE global std to {stats_path}")
        except Exception as e:
            print(f"Warning: failed to write {stats_path}: {e}")

    def _save_top_scores(self):
        data = {
            "top_mse_scores": [[float(s), int(e)] for s, e in self.top_mse_scores],
            "top_fid_scores": [[float(s), int(e)] for s, e in self.top_fid_scores],
        }
        with open(self.task_dir / "top_scores.yaml", "w") as f:
            yaml.safe_dump(data, f)

    def _update_and_save_top_fid(self, score, step, ckpt):
        if not hasattr(self, "_top_fid_ckpts"):
            self._top_fid_ckpts = []
        self._top_fid_ckpts.append((score, step, ckpt))
        self._top_fid_ckpts.sort(key=lambda x: x[0])
        self._top_fid_ckpts = self._top_fid_ckpts[: self.k_best]
        self.top_fid_scores = [(s, e) for s, e, _ in self._top_fid_ckpts]
        if self.top_fid_scores:
            self.best_fid = self.top_fid_scores[0][0]
        for path in glob.glob(str(self.task_dir / f"fid_rank_*_{self.ckpt_name}")):
            os.remove(path)
        for rank, (s, e, c) in enumerate(self._top_fid_ckpts, start=1):
            c["fid_score"] = s
            c["step"] = e
            out_path = self.task_dir / f"fid_rank_{rank}_{self.ckpt_name}"
            torch.save(c, out_path)
            print(f"  Saved fid rank {rank} (step {e}, FID={s:.4f}) → {out_path}")
        self._save_top_scores()

    def _update_and_save_top_mse(self, score, step, ckpt):
        if not hasattr(self, "_top_mse_ckpts"):
            self._top_mse_ckpts = []
        self._top_mse_ckpts.append((score, step, ckpt))
        self._top_mse_ckpts.sort(key=lambda x: x[0])
        self._top_mse_ckpts = self._top_mse_ckpts[: self.k_best]
        self.top_mse_scores = [(s, e) for s, e, _ in self._top_mse_ckpts]
        if self.top_mse_scores:
            self.best_mse = self.top_mse_scores[0][0]
        for path in glob.glob(str(self.task_dir / f"mse_rank_*_{self.ckpt_name}")):
            os.remove(path)
        for rank, (s, e, c) in enumerate(self._top_mse_ckpts, start=1):
            c["mse_score"] = s
            c["step"] = e
            out_path = self.task_dir / f"mse_rank_{rank}_{self.ckpt_name}"
            torch.save(c, out_path)
            print(f"  Saved mse rank {rank} (step {e}, MSE={s:.4f}) → {out_path}")
        self._save_top_scores()

    def encode_images(self, images):
        with torch.no_grad():
            latents = self.vae.encode(images).latent_dist.sample()
        return latents * self.scale

    def decode_images(self, latents):
        with torch.no_grad():
            latents = latents / self.scale
            images = self.vae.decode(latents).sample
        return images

    def apply_time_shift(self, t: torch.Tensor) -> torch.Tensor:
        """Apply SD3-style time shifting with identity at shift_factor=1."""
        if self.shift_factor == 1.0:
            return t
        return self.shift_factor * t / (1.0 + (self.shift_factor - 1.0) * t)

    def generate_images(self, latent_dummy, vae, unet, batch_size, device, num_steps=None):
        """Generate samples by Euler integration of the learned flow field."""
        if num_steps is None:
            num_steps = self.num_sampling_steps

        vae.eval()
        unet.eval()
        latent_shape = latent_dummy.shape[1:]
        x_t = torch.randn((batch_size, *latent_shape), device=device)
        timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
        timesteps = self.apply_time_shift(timesteps)

        for i in range(num_steps):
            t_curr = timesteps[i]
            t_next = timesteps[i + 1]
            dt = t_next - t_curr
            t_batch = torch.full((batch_size,), t_curr.item(), device=device)
            with torch.no_grad():
                v_pred = unet(sample=x_t, timestep=t_batch).sample
            x_t = x_t + v_pred * dt

        images = self.decode_images(x_t)
        return ((images + 1) / 2).clamp(0, 1)

    def train(self):
        self.model.train()
        data_iter = iter(self.train_dl)
        pbar = tqdm(
            total=self.max_steps,
            initial=self.global_step,
            desc="Steps",
            disable=not self.accelerator.is_local_main_process,
        )

        while self.global_step < self.max_steps:
            try:
                images, _ = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_dl)
                images, _ = next(data_iter)

            images = images.to(self.device)
            with self.accelerator.accumulate(self.model):
                latents = self.encode_images(images)
                bs = latents.size(0)

                if self.timestep_sampling == "logit_normal":
                    t = sample_timesteps_logit_normal(
                        bs,
                        self.device,
                        mean=self.logit_normal_mean,
                        std=self.logit_normal_std,
                    )
                else:
                    t = sample_timesteps_uniform(bs, self.device)
                t = self.apply_time_shift(t)

                x_t, velocity_target, _ = flow_forward(latents, t)

                if self.tr_cfg.get("nd"):
                    x_t_nd = self.nd(x_t)
                    pred = self.model(sample=x_t, timestep=t).sample
                    pred_nd = self.model(sample=x_t_nd, timestep=t).sample

                    loss_normal = F.mse_loss(pred, velocity_target, reduction="none")
                    loss_normal = loss_normal.mean(dim=list(range(1, len(loss_normal.shape))))

                    loss_nd = F.mse_loss(pred_nd, velocity_target, reduction="none")
                    loss_nd = loss_nd.mean(dim=list(range(1, len(loss_nd.shape))))

                    weights = flow_matching_loss_weight(t, self.loss_weighting)
                    weighted_loss_normal = (weights * loss_normal).mean()
                    weighted_loss_nd = (weights * loss_nd).mean()
                    loss = (1 - self.alpha) * weighted_loss_normal + self.alpha * weighted_loss_nd
                else:
                    pred = self.model(sample=x_t, timestep=t).sample
                    loss = F.mse_loss(pred, velocity_target, reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape))))
                    weights = flow_matching_loss_weight(t, self.loss_weighting)
                    loss = (weights * loss).mean()

                self.accelerator.backward(loss)
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    self.ema_model.step(self.model.parameters())
                    self.global_step += 1
                    if self.accelerator.is_local_main_process:
                        pbar.update(1)

            if self.accelerator.sync_gradients and self.accelerator.is_local_main_process:
                self.tb.add_scalar("Loss/train_total", loss.item(), self.global_step)
                self.tb.add_scalar("LR", self.lr_scheduler.get_last_lr()[0], self.global_step)
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

                if self.global_step > 0 and (
                    self.global_step % self.log_steps == 0
                    or self.global_step % self.fid_steps == 0
                ):
                    self._validate_and_log(self.global_step)
                    if self.global_step % self.log_steps == 0:
                        self._log_samples(self.global_step)

        pbar.close()
        if self.accelerator.is_local_main_process:
            self._save_final()

    def convert_to_rgb(self, images):
        if images.shape[1] == 1:
            return images.repeat(1, 3, 1, 1)
        return images

    def _validate_and_log(self, step: int):
        self.ema_model.store(self.model.parameters())
        self.ema_model.copy_to(self.model.parameters())

        self.model.eval()
        val_losses = []
        with torch.no_grad():
            for vimgs, _ in self.val_loader:
                vimgs = vimgs.to(self.device)
                vz = self.vae.encode(vimgs).latent_dist.sample() * self.scale
                bs = vz.size(0)
                vt = sample_timesteps_uniform(bs, self.device)
                vt = self.apply_time_shift(vt)
                x_t, velocity_target, _ = flow_forward(vz, vt)
                vp = self.model(sample=x_t, timestep=vt).sample
                val_losses.append(F.mse_loss(vp, velocity_target).item())

        avg_val = sum(val_losses) / len(val_losses)
        self.ema_model.restore(self.model.parameters())

        ckpt = {
            "model_state": self.accelerator.unwrap_model(self.model).state_dict(),
            "ema_state": self.ema_model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.lr_scheduler.state_dict(),
        }

        self._update_and_save_top_mse(avg_val, step, ckpt.copy())
        self.tb.add_scalar("Val/mse", avg_val, step)
        print(f"Step {step} → Val velocity MSE: {avg_val:.4f}")

        resume_ckpt = {
            "step": step,
            "global_step": self.global_step,
            "model_state": self.accelerator.unwrap_model(self.model).state_dict(),
            "ema_state": self.ema_model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.lr_scheduler.state_dict(),
            "best_mse": self.best_mse,
            "best_fid": self.best_fid,
        }
        torch.save(resume_ckpt, self.task_dir / "resume_checkpoint.pth")
        print(f"  Saved resume checkpoint → {self.task_dir / 'resume_checkpoint.pth'}")

        if step % self.fid_steps == 0:
            real_folder = self.task_dir / "fid_real"
            fake_folder = self.task_dir / f"temp_fake_step_{step}"
            real_folder.mkdir(parents=True, exist_ok=True)
            if fake_folder.exists():
                shutil.rmtree(fake_folder)
            write_real = not any(real_folder.iterdir())

            self.ema_model.store(self.model.parameters())
            self.ema_model.copy_to(self.model.parameters())
            self.model.eval()

            batch_count = 0
            with torch.no_grad():
                for real_imgs, _ in self.val_loader:
                    real_imgs = real_imgs.to(self.device)
                    bs = real_imgs.size(0)

                    if write_real:
                        real_01 = ((real_imgs + 1) / 2).clamp(0, 1)
                        real_rgb = self.convert_to_rgb(real_01)
                        save_images_to_folder(real_rgb, real_folder, f"real_{batch_count}")

                    fake_01 = self.generate_images(self.z_dummy, self.vae, self.model, bs, self.device)
                    fake_rgb = self.convert_to_rgb(fake_01)
                    save_images_to_folder(fake_rgb, fake_folder, f"fake_{batch_count}")
                    batch_count += 1

            fid_score = fid.compute_fid(str(real_folder), str(fake_folder), mode="clean")
            shutil.rmtree(fake_folder)

            self.ema_model.restore(self.model.parameters())

            self.tb.add_scalar("Val/FID", fid_score, step)
            print(f"Step {step} → Val velocity MSE: {avg_val:.4f}, FID: {fid_score:.2f}")

            ckpt = {
                "step": step,
                "global_step": self.global_step,
                "model_state": self.accelerator.unwrap_model(self.model).state_dict(),
                "ema_state": self.ema_model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.lr_scheduler.state_dict(),
                "best_mse": self.best_mse,
                "best_fid": self.best_fid,
            }
            self._update_and_save_top_fid(fid_score, step, ckpt.copy())

    def _log_samples(self, step: int):
        self.ema_model.store(self.model.parameters())
        self.ema_model.copy_to(self.model.parameters())
        with torch.no_grad():
            _ = self.generate_images(self.z_dummy, self.vae, self.model, 32, self.device)
        self.ema_model.restore(self.model.parameters())

    def _save_final(self):
        print("\nTraining complete – saving final checkpoints …")
        torch.save(self.accelerator.unwrap_model(self.model).state_dict(), self.task_dir / "unet_final.pt")
        torch.save(self.ema_model.state_dict(), self.task_dir / "ema_final.pt")
        self.tb.close()
        print(" Saved: unet_final.pt  |  ema_final.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train Latent Flow Matching Model (UNet stage)")
    parser.add_argument("--config", "-c", dest="config_path", required=True, help="Path to YAML config")
    parser.add_argument("--vae_ckpt_path", "-v", required=True, help="Pre-trained VAE checkpoint")
    parser.add_argument("--gpus", "-g", default=None, help="Comma-separated GPU ids (optional)")
    parser.add_argument("--resume_path", "-r", default=None, help="Path to specific checkpoint to resume from (optional)")
    cli_args = parser.parse_args()
    trainer = FlowMatchingLDMTrainer(cli_args)
    trainer.train()
