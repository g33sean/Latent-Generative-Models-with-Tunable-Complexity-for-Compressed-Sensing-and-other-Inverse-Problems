#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import yaml
import random
import numpy as np
import torch
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import torch.nn as nn
from models.discriminator import Discriminator
from utils.nd import NestedDropout
from models.vae import VAE_MODELS
from models.lpips import LPIPS
from data.dataset_loader import get_train_val_dataloaders
from accelerate import Accelerator
import torch.nn.functional as F
from torch.autograd import grad as torch_grad

def print_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

def load_transfer_weights(model, ckpt_path: str, device) -> dict:
    """
    Load encoder + decoder weights from a prior checkpoint into model.
    Discriminator is intentionally excluded — it starts fresh (domain shift).

    Returns a dict with load stats for logging.
    """
    print(f"[Transfer] Loading weights from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    # Support both raw state_dict and our full checkpoint format
    if "model_state" in ckpt:
        src_state = ckpt["model_state"]
        src_step  = ckpt.get("global_step", "unknown")
        print(f"[Transfer] Source checkpoint was at step {src_step}")
    else:
        # Assume the file IS the state dict
        src_state = ckpt
        src_step  = "unknown"

    tgt_state = model.state_dict()

    matched, skipped, shape_mismatch = [], [], []

    new_state = {}
    for k, v in tgt_state.items():
        if k not in src_state:
            skipped.append(k)
            new_state[k] = v                        # keep random init
        elif src_state[k].shape != v.shape:
            shape_mismatch.append(k)
            new_state[k] = v                        # keep random init
        else:
            matched.append(k)
            new_state[k] = src_state[k]             # transfer weight

    model.load_state_dict(new_state)

    print(f"[Transfer] Matched  : {len(matched)}")
    print(f"[Transfer] Skipped  : {len(skipped)}  (not in source)")
    print(f"[Transfer] Mismatch : {len(shape_mismatch)}  (shape differs, kept random)")
    if shape_mismatch:
        print(f"[Transfer] Mismatched keys: {shape_mismatch}")

    return {"matched": matched, "skipped": skipped, "shape_mismatch": shape_mismatch}

class AutoencoderTrainer:
    """VAE training loop with GAN loss, LPIPS perceptual loss, R1 regularization, and nested dropout."""

    def __init__(self, args: argparse.Namespace):
        self.args = args

        # Set GPU visibility BEFORE creating accelerator
        if self.args.gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.args.gpu

        with open(self.args.config, "r") as fh:
            self.config = yaml.safe_load(fh)
        self.ds_cfg   = self.config["dataset_params"]
        self.tr_cfg   = self.config["train_params"]
        self.task_name = self.tr_cfg["task_name"]

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.tr_cfg.get("acc_steps", 1)
        )
        self.device = self.accelerator.device
        print(
            f"Using device: {self.device} | "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'None')}"
        )

        seed = self.tr_cfg["seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        self.train_loader, self.val_loader = get_train_val_dataloaders(
            self.ds_cfg,
            train=True,
            train_batch_size=self.tr_cfg.get("batch_size", 64),
            val_batch_size=self.tr_cfg.get("val_batch_size", 64),
            val_num_samples=self.ds_cfg.get("val_num_samples", 1000),
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

        models     = VAE_MODELS()
        self.model = models.create_autoencoder_from_dataset(self.ds_cfg)
        print_model_parameters(self.model)

        if getattr(self.model, "enable_gradient_checkpointing", None):
            self.model.enable_gradient_checkpointing()
        elif getattr(self.model, "gradient_checkpointing_enable", None):
            self.model.gradient_checkpointing_enable()

        self.discriminator = Discriminator(
            in_channels=self.ds_cfg["im_channels"], base_channels=128
        )

        # Resolve the flag: CLI --transfer_learn overrides; fall back to
        # config key transfer_learn_params.enabled if present.
        tl_cfg       = self.config.get("transfer_learn_params", {})
        cli_tl       = self.args.transfer_learn           # bool from argparse
        self.use_tl  = cli_tl or tl_cfg.get("enabled", False)

        # Checkpoint path: CLI --transfer_ckpt > config > None
        tl_ckpt_path = self.args.transfer_ckpt or tl_cfg.get("ckpt_path", None)

        # LR multiplier for transferred weights (lower = less forgetting)
        tl_lr_scale  = tl_cfg.get("lr_scale", 0.1)       # default 10x lower

        if self.use_tl:
            if not tl_ckpt_path:
                raise ValueError(
                    "Transfer learning enabled but no checkpoint path provided. "
                    "Pass --transfer_ckpt <path> or set transfer_learn_params.ckpt_path in config."
                )
            tl_stats = load_transfer_weights(self.model, tl_ckpt_path, self.device)
            if self.accelerator.is_main_process:
                print(
                    f"[Transfer] LR scale for transferred weights: {tl_lr_scale}x  "
                    f"(discriminator always uses full LR)"
                )

        lr = self.tr_cfg["lr"]

        if self.use_tl:
            # Two param groups: transferred weights at reduced LR, everything
            # else (any randomly-init'd layers) at full LR.
            transferred_ids = set()
            for name, param in self.model.named_parameters():
                if name in tl_stats["matched"]:
                    transferred_ids.add(id(param))

            transferred_params = [
                p for p in self.model.parameters()
                if id(p) in transferred_ids
            ]
            fresh_params = [
                p for p in self.model.parameters()
                if id(p) not in transferred_ids
            ]

            g_param_groups = []
            if transferred_params:
                g_param_groups.append(
                    {"params": transferred_params, "lr": lr * tl_lr_scale}
                )
            if fresh_params:
                g_param_groups.append(
                    {"params": fresh_params, "lr": lr}
                )
            # Fall back gracefully if somehow both are empty
            if not g_param_groups:
                g_param_groups = self.model.parameters()

            self.opt_g = Adam(g_param_groups, betas=(0.5, 0.999))
        else:
            self.opt_g = Adam(self.model.parameters(), lr=lr, betas=(0.5, 0.999))

        # Discriminator always trains from scratch at full LR
        self.opt_d = Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

        (
            self.model,
            self.discriminator,
            self.opt_g,
            self.opt_d,
            self.train_loader,
            self.val_loader,
        ) = self.accelerator.prepare(
            self.model,
            self.discriminator,
            self.opt_g,
            self.opt_d,
            self.train_loader,
            self.val_loader,
        )

        self.unwrapped_model          = self.accelerator.unwrap_model(self.model)
        self.unwrapped_discriminator  = self.accelerator.unwrap_model(self.discriminator)

        self.recon_crit  = torch.nn.L1Loss()
        self.lpips_model = LPIPS().eval().to(self.device)

        with torch.no_grad():
            dummy  = torch.zeros(
                1,
                self.ds_cfg["im_channels"],
                self.ds_cfg.get("im_size", 64),
                self.ds_cfg.get("im_size", 64),
                device=self.device,
            )
            z_dummy = self.unwrapped_model.encode(dummy)["latent_dist"].sample()

        k       = int(np.prod(z_dummy.shape[1:]))
        self.nd = NestedDropout(
            k, self.tr_cfg.get("drop_p", 1e-3), self.device
        ).to(self.device)

        if self.accelerator.is_main_process:
            os.makedirs(self.task_name, exist_ok=True)
            self.writer = SummaryWriter(os.path.join(self.task_name, "tb_logs"))
            with open(os.path.join(self.task_name, "config.yaml"), "w") as cfg:
                yaml.safe_dump(self.config, cfg)
        else:
            self.writer = None

        self.global_step    = 0
        self.acc_steps      = self.tr_cfg["acc_steps"]
        self.best_val_lpips = float("inf")

        self.max_train_steps = self.tr_cfg.get("max_train_steps", 50000)
        self.save_steps      = self.tr_cfg.get("save_steps", 5000)
        self.eval_steps      = self.tr_cfg.get("eval_steps", 1000)
        self.log_steps       = self.tr_cfg.get("log_steps", 100)

        def _zero():
            return torch.tensor(0.0, device=self.device)
        self._recon_loss = self._kl_loss = _zero()
        self._lpips_loss = self._lpips_drop = self._disc_loss = _zero()
        self._adv_loss   = _zero()
        self.adaptive_w  = _zero()
        self._r1_penalty = _zero()

        ckpt_path = os.path.join(self.task_name, "checkpoint.pth")
        if os.path.exists(ckpt_path):
            try:
                if self.accelerator.num_processes > 1:
                    map_location = {
                        "cuda:%d" % 0: "cuda:%d" % self.accelerator.local_process_index
                    }
                else:
                    map_location = self.device

                ckpt = torch.load(ckpt_path, map_location=map_location)

                self.unwrapped_model.load_state_dict(ckpt["model_state"])
                self.opt_g.load_state_dict(ckpt["opt_g_state"])
                self.opt_d.load_state_dict(ckpt["opt_d_state"])
                self.unwrapped_discriminator.load_state_dict(ckpt["disc_state"])
                self.best_val_lpips = ckpt["best_val_lpips"]
                self.global_step    = ckpt["global_step"]

                if self.accelerator.is_main_process:
                    print(
                        f"Resuming from step {self.global_step} "
                        f"(best LPIPS={self.best_val_lpips:.4f})"
                    )
            except Exception as e:
                if self.accelerator.is_main_process:
                    print(f"Failed to load checkpoint: {e}")
                    print("Starting from scratch...")

    def calculate_adaptive_weight(self, nll_loss, adv_loss):
        gen_params = [p for p in self.unwrapped_model.parameters() if p.requires_grad]

        grads_nll = torch.autograd.grad(
            nll_loss, gen_params, retain_graph=True, allow_unused=True
        )
        grads_adv = torch.autograd.grad(
            adv_loss, gen_params, retain_graph=True, allow_unused=True
        )

        def norm_sq(grads):
            return sum((g.detach() ** 2).sum() for g in grads if g is not None)

        nll_norm  = torch.sqrt(norm_sq(grads_nll))
        adv_norm  = torch.sqrt(norm_sq(grads_adv))
        d_weight  = nll_norm / (adv_norm + 1e-4)
        d_weight  = torch.clamp(d_weight, 0.0, 1e4).detach()
        return d_weight

    def _generator_loss(self, imgs, output, posterior, z):
        recon = self.recon_crit(output, imgs)
        kl    = posterior["latent_dist"].kl().mean()
        self._recon_loss, self._kl_loss = recon.detach(), kl.detach()

        loss = recon + self.tr_cfg["kl_weight"] * kl

        lpips_drop_val = torch.tensor(0.0, device=imgs.device)
        if self.tr_cfg.get("lambda_reg", 0) > 0:
            flat    = z.view(z.size(0), -1)
            dropped = self.nd(flat).view_as(z)
            recon_k = self.unwrapped_model.decode(dropped).sample
            lpips_drop_val = torch.mean(self.lpips_model(recon_k, imgs))
            loss += self.tr_cfg["lambda_reg"] * lpips_drop_val
        self._lpips_drop = lpips_drop_val.detach()

        lpips_val        = torch.mean(self.lpips_model(output, imgs))
        self._lpips_loss = lpips_val.detach()
        loss += self.tr_cfg["perceptual_weight"] * lpips_val

        adv_loss             = torch.tensor(0.0, device=imgs.device)
        disc_weight_current  = 0.0

        disc_start      = self.tr_cfg["disc_start"]
        disc_ramp_steps = self.tr_cfg.get("disc_ramp_steps", 1000)

        if self.global_step > disc_start:
            steps_since_start = self.global_step - disc_start
            if steps_since_start < disc_ramp_steps:
                ramp_progress       = steps_since_start / disc_ramp_steps
                disc_weight_current = ramp_progress * self.tr_cfg.get("disc_weight", 1.0)
            else:
                disc_weight_current = self.tr_cfg.get("disc_weight", 1.0)

            if disc_weight_current > 0:
                fake_pred        = self.discriminator(output)
                fake_pred        = fake_pred.clamp(-5, 5)
                gan_loss_type    = self.tr_cfg.get("gan_loss_type", "hinge")

                if gan_loss_type == "hinge":
                    adv_loss = -fake_pred.mean()
                elif gan_loss_type == "lsgan":
                    adv_loss = F.mse_loss(fake_pred, torch.ones_like(fake_pred))
                elif gan_loss_type == "vanilla":
                    adv_loss = F.binary_cross_entropy_with_logits(
                        fake_pred, torch.ones_like(fake_pred)
                    )
                else:
                    adv_loss = -F.logsigmoid(fake_pred).mean()

                if self.tr_cfg.get("use_adaptive_weight", True):
                    nll = (
                        recon
                        + self.tr_cfg["kl_weight"] * kl
                        + self.tr_cfg["perceptual_weight"] * self._lpips_loss
                        + self.tr_cfg.get("lambda_reg", 0) * self._lpips_drop
                    )
                    self.adaptive_w  = self.calculate_adaptive_weight(nll, adv_loss)
                    final_weight     = self.adaptive_w * disc_weight_current
                else:
                    final_weight     = disc_weight_current
                    self.adaptive_w  = torch.tensor(disc_weight_current, device=imgs.device)

                loss += final_weight * adv_loss

        self._adv_loss            = adv_loss.detach()
        self._disc_weight_current = disc_weight_current
        return loss

    def _discriminator_loss(self, real, fake):
        fake = fake.detach()
        real.requires_grad_(True)

        real_pred     = self.discriminator(real)
        fake_pred     = self.discriminator(fake)
        gan_loss_type = self.tr_cfg.get("gan_loss_type", "hinge")

        if gan_loss_type == "hinge":
            loss_adv = F.relu(1.0 - real_pred).mean() + F.relu(1.0 + fake_pred).mean()
        elif gan_loss_type == "lsgan":
            loss_adv = 0.5 * (
                F.mse_loss(real_pred, torch.ones_like(real_pred))
                + F.mse_loss(fake_pred, torch.zeros_like(fake_pred))
            )
        elif gan_loss_type == "vanilla":
            loss_adv = (
                F.binary_cross_entropy_with_logits(real_pred, torch.ones_like(real_pred))
                + F.binary_cross_entropy_with_logits(fake_pred, torch.zeros_like(fake_pred))
            )
        else:
            loss_adv = -F.logsigmoid(real_pred).mean() - F.logsigmoid(-fake_pred).mean()

        r1_gamma = self.tr_cfg.get("r1_gamma", 10.0)
        r1_every = self.tr_cfg.get("r1_every", 16)

        if (self.global_step % r1_every) == 0:
            grad_real  = torch_grad(
                outputs=real_pred.sum(),
                inputs=real,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            r1_penalty = 0.5 * r1_gamma * grad_real.pow(2).view(
                grad_real.size(0), -1
            ).sum(1).mean()
        else:
            r1_penalty = torch.tensor(0.0, device=real.device)

        loss             = loss_adv + r1_penalty
        self._disc_loss  = loss.detach()
        self._r1_penalty = r1_penalty.detach()
        return loss

    def train(self):
        self.model.train()
        self.discriminator.train()

        train_iter = iter(self.train_loader)

        pbar = tqdm(
            range(self.global_step, self.max_train_steps),
            initial=self.global_step,
            total=self.max_train_steps,
            desc="Training",
            disable=not self.accelerator.is_local_main_process,
        )

        for step in pbar:
            try:
                imgs, _ = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                imgs, _    = next(train_iter)

            with self.accelerator.accumulate([self.model, self.discriminator]):
                for p in self.discriminator.parameters():
                    p.requires_grad = False

                posterior = self.unwrapped_model.encode(imgs)
                z         = posterior["latent_dist"].sample()
                output    = self.unwrapped_model.decode(z).sample

                g_loss = self._generator_loss(imgs, output, posterior, z)
                self.accelerator.backward(g_loss)

                for p in self.discriminator.parameters():
                    p.requires_grad = True

                if self.global_step > self.tr_cfg["disc_start"]:
                    d_loss = self._discriminator_loss(imgs, output)
                    self.accelerator.backward(d_loss)

                if self.accelerator.sync_gradients:
                    self.global_step += 1

                    self.accelerator.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.opt_g.step()
                    self.opt_g.zero_grad()

                    if self.global_step > self.tr_cfg["disc_start"]:
                        self.opt_d.step()
                        self.opt_d.zero_grad()

            if (
                self.writer
                and self.accelerator.is_main_process
                and self.global_step % self.log_steps == 0
            ):
                self.writer.add_scalar("Loss/Generator_Total",  g_loss.item(),              self.global_step)
                self.writer.add_scalar("Loss/Reconstruction",   self._recon_loss.item(),     self.global_step)
                self.writer.add_scalar("Loss/KL",               self._kl_loss.item(),        self.global_step)
                self.writer.add_scalar("Loss/KL_Scaled",        self._kl_loss.item() * self.tr_cfg["kl_weight"], self.global_step)
                self.writer.add_scalar("Loss/LPIPS",            self._lpips_loss.item(),     self.global_step)
                self.writer.add_scalar("Loss/LPIPS_Drop",       self._lpips_drop.item(),     self.global_step)
                self.writer.add_scalar("Loss/LPIPS_Drop_Scaled", self._lpips_drop.item() * self.tr_cfg["lambda_reg"], self.global_step)

                if self.global_step > self.tr_cfg["disc_start"]:
                    self.writer.add_scalar("Loss/Adversarial",         self._adv_loss.item(),          self.global_step)
                    self.writer.add_scalar("Loss/Discriminator",        self._disc_loss.item(),         self.global_step)
                    self.writer.add_scalar("Weights/Adaptive_D",        self.adaptive_w.item(),         self.global_step)
                    self.writer.add_scalar("Weights/Disc_Weight_Current", self._disc_weight_current,    self.global_step)
                    self.writer.add_scalar("Loss/R1_penalty",           self._r1_penalty.item(),        self.global_step)

                # Log TL LR groups once so you can verify in tensorboard
                if self.use_tl:
                    for i, pg in enumerate(self.opt_g.param_groups):
                        self.writer.add_scalar(f"LR/opt_g_group{i}", pg["lr"], self.global_step)

            if (
                self.writer
                and self.accelerator.is_main_process
                and self.global_step % self.tr_cfg.get("img_save_steps", 1000) == 0
            ):
                with torch.no_grad():
                    imgs_vis   = torch.clamp(imgs[:4],   -1, 1)
                    output_vis = torch.clamp(output[:4], -1, 1)
                    grid = make_grid(
                        torch.cat([imgs_vis, output_vis], 0).add(1).div_(2), nrow=4
                    )
                    self.writer.add_image("Samples", grid, self.global_step)

            if self.global_step % self.eval_steps == 0:
                self.validate()
                self.model.train()
                self.discriminator.train()

            if self.global_step % self.save_steps == 0 and self.accelerator.is_main_process:
                self.save_checkpoint()

            if self.accelerator.is_local_main_process:
                postfix = {
                    "g_loss": f"{g_loss.item():.3f}",
                    "recon":  f"{self._recon_loss.item():.3f}",
                    "kl":     f"{self._kl_loss.item():.3f}",
                    "lpips":  f"{self._lpips_loss.item():.3f}",
                }
                if self.global_step > self.tr_cfg["disc_start"]:
                    postfix["adv"]    = f"{self._adv_loss.item():.3f}"
                    postfix["d_loss"] = f"{self._disc_loss.item():.3f}"
                pbar.set_postfix(postfix)

            if self.global_step >= self.max_train_steps:
                break

    def validate(self):
        self.model.eval()
        vals = []

        with torch.no_grad():
            for vimgs, _ in self.val_loader:
                z    = self.unwrapped_model.encode(vimgs)["latent_dist"].sample()
                vout = self.unwrapped_model.decode(z).sample
                vals.append(torch.mean(self.lpips_model(vout, vimgs)).item())

        avg_val = float(np.mean(vals))
        if self.accelerator.num_processes > 1:
            avg_val = self.accelerator.gather(
                torch.tensor(avg_val, device=self.device)
            ).mean().item()

        if self.writer and self.accelerator.is_main_process:
            self.writer.add_scalar("Val/LPIPS", avg_val, self.global_step)

        if avg_val < self.best_val_lpips and self.accelerator.is_main_process:
            self.best_val_lpips = avg_val
            best_path = os.path.join(
                self.task_name, "best_" + self.tr_cfg["autoencoder_ckpt_name"]
            )
            self.accelerator.save(self.unwrapped_model.state_dict(), best_path)
            print(f"[Best Model] step={self.global_step}  LPIPS={self.best_val_lpips:.4f}")

    def save_checkpoint(self):
        ckpt = {
            "global_step":    self.global_step,
            "model_state":    self.unwrapped_model.state_dict(),
            "opt_g_state":    self.opt_g.state_dict(),
            "opt_d_state":    self.opt_d.state_dict(),
            "disc_state":     self.unwrapped_discriminator.state_dict(),
            "best_val_lpips": self.best_val_lpips,
        }
        ckpt_path      = os.path.join(self.task_name, "checkpoint.pth")
        step_ckpt_path = os.path.join(self.task_name, f"checkpoint-{self.global_step}.pth")
        torch.save(ckpt, ckpt_path)
        torch.save(ckpt, step_ckpt_path)
        print(f"[Checkpoint] step={self.global_step}  LPIPS={self.best_val_lpips:.4f}")

    def run(self):
        if self.accelerator.is_main_process:
            print(f"Starting training from step {self.global_step} to {self.max_train_steps}")
            if self.use_tl:
                print("[Transfer Learning] Encoder + Decoder transferred. Discriminator is fresh.")

        self.train()

        if self.accelerator.is_main_process:
            self.save_checkpoint()

        self.accelerator.wait_for_everyone()

        if self.writer and self.accelerator.is_main_process:
            self.writer.close()

        if self.accelerator.is_main_process:
            print("Training complete.")

# CLI entry-point
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",        "-c",  required=True,               help="Path to YAML config file")
    parser.add_argument("--transfer_learn","-tl", action="store_true",         help="Enable transfer learning from a prior checkpoint")
    parser.add_argument("--transfer_ckpt", "-tc", default=None,                help="Path to source checkpoint for transfer learning")
    parser.add_argument("--gpu",           "-g",  default=None,                help="Single GPU id or comma-separated ids (e.g. '0' or '0,1')")
    args = parser.parse_args()

    trainer = AutoencoderTrainer(args)
    trainer.run()
