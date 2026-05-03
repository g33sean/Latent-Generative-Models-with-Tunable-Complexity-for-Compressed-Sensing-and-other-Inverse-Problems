#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import yaml
import shutil
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from pathlib import Path
from typing import Optional
import pandas as pd
import piq
from cleanfid import fid
from PIL import Image
from diffusers import DDPMScheduler, DPMSolverMultistepScheduler
from diffusers.training_utils import EMAModel

from models.vae import VAE_MODELS
from models.unet import UNET_MODELS
from data.dataset_loader import get_dataset, get_train_val_dataloaders


# ── shared helpers ────────────────────────────────────────────────────────────

def convert_to_rgb(images: torch.Tensor) -> torch.Tensor:
    if images.shape[1] == 1:
        return images.repeat(1, 3, 1, 1)
    return images


def apply_k_mask(z: torch.Tensor, k: Optional[int]) -> torch.Tensor:
    if k is None or k < 1:
        return z
    B, C, H, W = z.shape
    flat = z.view(B, -1)
    masked = torch.zeros_like(flat)
    masked[:, :k] = flat[:, :k]
    return masked.view(B, C, H, W)


def load_vae(ckpt_path: str, model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    ck = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ck.get("model_state", ck))
    model.eval()
    return model


def load_unet(ckpt_path: str, model: torch.nn.Module, device: torch.device,
              use_ema: bool = True) -> torch.nn.Module:
    ck = torch.load(ckpt_path, map_location=device)
    if "ema_state" in ck and use_ema:
        ema = EMAModel(model.parameters())
        ema.load_state_dict(ck["ema_state"])
        ema.copy_to(model.parameters())
    else:
        model.load_state_dict(ck.get("model_state", ck))
    model.eval()
    return model


def save_images_to_folder(images: torch.Tensor, folder: Path, prefix: str = "img"):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    images = images.detach().cpu()
    if images.min() < 0:
        images = (images + 1.0) / 2.0
    images_uint8 = (images * 255.0).clamp(0, 255).to(torch.uint8)
    for i, arr in enumerate(images_uint8.permute(0, 2, 3, 1).numpy()):
        if arr.ndim == 3 and arr.shape[2] == 1:
            img = Image.fromarray(arr.squeeze(2), mode="L")
        else:
            img = Image.fromarray(arr, mode="RGB")
        img.save(folder / f"{prefix}_{i:06d}.png")


# ── VAE evaluation ────────────────────────────────────────────────────────────

def evaluate_vae(args):
    device = torch.device(args.device)

    cfg = yaml.safe_load(open(args.config))
    ds_cfg = cfg["dataset_params"]
    ds_cfg["im_path"] = args.data_samples

    out_dir = Path(__file__).resolve().parent / "VAE" / args.result_name
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, out_dir / "config.yaml")

    ds_test = get_dataset(ds_cfg, train=False)
    n = min(args.num_samples, len(ds_test))
    if n < len(ds_test):
        ds_test = Subset(ds_test, torch.randperm(len(ds_test))[:n].tolist())
    test_loader = DataLoader(ds_test, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers,
                             pin_memory=True)

    ae = load_vae(args.ckpt, VAE_MODELS().create_autoencoder_from_dataset(ds_cfg).to(device), device)

    with torch.no_grad():
        imgs, _ = next(iter(test_loader))
        z0 = ae.encode(imgs.to(device))["latent_dist"].sample()
    latent_dim = z0.shape[1] * z0.shape[2] * z0.shape[3]
    print(f"Latent dim: {latent_dim}  shape: {tuple(z0.shape[1:])}")

    mask_fracs = [round(f, 2) for f in torch.arange(1.0, 0.0, -0.1).tolist()] + [0.01]
    total_samples = len(test_loader.dataset)
    records = []

    for frac in mask_fracs:
        k = max(1, int(latent_dim * frac))
        print(f"\n‣ keep {100*frac:.0f}% → k={k}")

        real_folder = out_dir / f"temp_real_{frac:.2f}"
        pred_folder = out_dir / f"temp_pred_{frac:.2f}"
        for d in (real_folder, pred_folder):
            if d.exists():
                shutil.rmtree(d)

        mse_sum = ssim_sum = psnr_sum = 0.0
        with torch.no_grad():
            for bi, (imgs, _) in enumerate(tqdm(test_loader, desc=f"frac={frac:.2f}")):
                imgs = imgs.to(device)
                z = ae.encode(imgs)["latent_dist"].sample()
                recon = ae.decode(apply_k_mask(z, k)).sample
                real = convert_to_rgb(((imgs  + 1) / 2).clamp(0, 1))
                pred = convert_to_rgb(((recon + 1) / 2).clamp(0, 1))
                bs = imgs.size(0)
                mse_sum  += F.mse_loss(pred, real, reduction="mean").item() * bs
                ssim_sum += piq.ssim(pred, real, data_range=1.0, reduction="mean").item() * bs
                psnr_sum += piq.psnr(pred, real, data_range=1.0, reduction="mean").item() * bs
                save_images_to_folder(real, real_folder, f"real_{bi:05d}")
                save_images_to_folder(pred, pred_folder, f"pred_{bi:05d}")

        fid_val = fid.compute_fid(str(real_folder), str(pred_folder), mode="clean")
        shutil.rmtree(real_folder)
        shutil.rmtree(pred_folder)

        records.append({
            "ckpt": os.path.abspath(args.ckpt),
            "mask_frac": frac, "k": k,
            "mse":  mse_sum  / total_samples,
            "ssim": ssim_sum / total_samples,
            "psnr": psnr_sum / total_samples,
            "fid":  fid_val,
        })
        r = records[-1]
        print(f"  MSE={r['mse']:.4f}  SSIM={r['ssim']:.4f}  PSNR={r['psnr']:.2f}  FID={fid_val:.2f}")

    df = pd.DataFrame(records)
    out_csv = out_dir / args.csv
    df.to_csv(out_csv, index=False)
    print(f"\nSaved → {out_csv}\n{df}")


# ── UNet evaluation ───────────────────────────────────────────────────────────

def _extract_images(batch) -> torch.Tensor:
    if isinstance(batch, torch.Tensor):
        return batch
    if isinstance(batch, (list, tuple)):
        for item in batch:
            if isinstance(item, torch.Tensor) and item.dim() >= 4:
                return item
        for item in batch:
            if isinstance(item, torch.Tensor):
                return item
        raise ValueError("No image tensor in batch.")
    if isinstance(batch, dict):
        for key in ("image", "images", "img", "pixel_values", "input", "data"):
            if key in batch and isinstance(batch[key], torch.Tensor):
                return batch[key]
        for v in batch.values():
            if isinstance(v, torch.Tensor) and v.dim() >= 4:
                return v
        raise ValueError("No image tensor in batch.")
    raise TypeError(f"Unsupported batch type: {type(batch)}")


def _estimate_scale(vae, dataloader, device) -> float:
    vae.eval()
    n, moment_2 = 0, 0.0
    with torch.no_grad():
        for imgs, _ in tqdm(dataloader, desc="Computing scale factor", leave=False):
            z = vae.encode(imgs.to(device)).latent_dist.sample()
            moment_2 += (z ** 2).mean().item() * imgs.size(0)
            n += imgs.size(0)
    return (moment_2 / n) ** 0.5


def _dump_real(loader, split: str, root: Path, max_images: Optional[int]) -> Path:
    folder = root / f"real_{split}"
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True, exist_ok=True)
    saved = 0
    with torch.no_grad():
        for bi, batch in enumerate(tqdm(loader, desc=f"Saving real ({split})")):
            if max_images and saved >= max_images:
                break
            imgs = _extract_images(batch)
            imgs = ((imgs + 1.0) / 2.0).clamp(0, 1) if imgs.min() < 0 else imgs.clamp(0, 1)
            imgs = convert_to_rgb(imgs)
            if max_images:
                imgs = imgs[: max_images - saved]
            save_images_to_folder(imgs, folder, f"real_{split}_{bi:05d}")
            saved += imgs.size(0)
    return folder


def _generate(*, steps, vae, unet, scheduler, batch_size, device,
              latent_shape, scale, k) -> torch.Tensor:
    vae.eval(); unet.eval()
    latents = torch.randn((batch_size, *latent_shape), device=device)
    scheduler.set_timesteps(steps, device=device)
    for t in scheduler.timesteps:
        with torch.no_grad():
            latents = scheduler.step(unet(latents, t).sample, t, latents).prev_sample
            latents = apply_k_mask(latents, k)
    return ((vae.decode(latents / scale).sample + 1.0) / 2.0).clamp(0, 1)


def evaluate_unet(args):
    device = torch.device(args.device)

    cfg = yaml.safe_load(open(args.config))
    ds_cfg = cfg["dataset_params"]
    tr_cfg = cfg.get("train_params", {})
    num_train_timesteps = tr_cfg.get("denoising_timesteps", 1000)

    out_dir = Path(__file__).resolve().parent / "UNET" / args.result_name
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, out_dir / "config.yaml")

    temp_root = out_dir / "temp_fid_work"
    if temp_root.exists():
        shutil.rmtree(temp_root)
    temp_root.mkdir(parents=True, exist_ok=True)

    base_sched = DDPMScheduler(num_train_timesteps=num_train_timesteps,
                               beta_schedule="scaled_linear",
                               prediction_type="epsilon", clip_sample=False)
    scheduler = DPMSolverMultistepScheduler.from_config(
        base_sched.config, solver_order=2,
        use_karras_sigmas=True, algorithm_type="dpmsolver++")

    train_loader, val_loader = get_train_val_dataloaders(
        ds_cfg, True, args.batch_size, args.batch_size,
        val_num_samples=ds_cfg.get("val_num_samples", 0),
        shuffle=False, num_workers=args.num_workers)
    test_loader = val_loader if (val_loader and len(val_loader)) else DataLoader(
        get_dataset(ds_cfg, train=False), batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers)

    vae  = load_vae(args.vae_ckpt,
                    VAE_MODELS().create_autoencoder_from_dataset(ds_cfg).to(device), device)
    unet = load_unet(args.ckpt,
                     UNET_MODELS().create_unet_from_dataset(ds_cfg).to(device), device, args.use_ema)

    with torch.no_grad():
        z0 = vae.encode(_extract_images(next(iter(test_loader))).to(device)).latent_dist.sample()
    latent_shape = tuple(z0.shape[1:])
    scale = 1.0 / max(_estimate_scale(vae, train_loader, device), 1e-12)
    print(f"Latent shape: {latent_shape}  scale: {scale:.6f}")

    mf_list = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.01] if args.test_masking else [1.0]
    _dump_real(train_loader, "train", temp_root, args.max_train_images)
    _dump_real(test_loader,  "test",  temp_root, args.max_test_images)

    records = []
    total_latents = int(torch.tensor(latent_shape).prod().item())
    try:
        for split, loader in [("train", train_loader), ("test", test_loader)]:
            n_ds = len(loader.dataset)
            n_fakes = (min(args.max_train_images, n_ds) if split == "train"
                       else (min(args.max_test_images, n_ds) if args.max_test_images else n_ds))
            bs = loader.batch_size if hasattr(loader, "batch_size") else args.batch_size
            for mf in mf_list:
                k = None if mf >= 1.0 else max(1, int(total_latents * mf))
                fake_folder = temp_root / f"fake_{split}_k_{k or 'all'}"
                if fake_folder.exists():
                    shutil.rmtree(fake_folder)
                fake_folder.mkdir(parents=True, exist_ok=True)
                for b in tqdm(range((n_fakes + bs - 1) // bs),
                              desc=f"Fakes {split} mf={mf}", leave=False):
                    cur_bs = min(bs, n_fakes - b * bs)
                    if cur_bs <= 0:
                        break
                    fake = convert_to_rgb(_generate(
                        steps=args.num_inference_steps, vae=vae, unet=unet,
                        scheduler=scheduler, batch_size=cur_bs, device=device,
                        latent_shape=latent_shape, scale=scale, k=k))
                    save_images_to_folder(fake, fake_folder, f"fake_{split}_{b:05d}")
                fid_val = float(fid.compute_fid(
                    str(temp_root / f"real_{split}"), str(fake_folder), mode="clean"))
                shutil.rmtree(fake_folder, ignore_errors=True)
                records.append({
                    "dataset": ds_cfg.get("name"), "split": split,
                    "mask_frac": mf, "k": k or total_latents,
                    "num_fakes": n_fakes, "fid": fid_val,
                })
                print(f"[{split}] mf={mf:.2f} k={k} → FID={fid_val:.4f}")
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)

    df = pd.DataFrame(records)
    out_csv = out_dir / args.csv
    df.to_csv(out_csv, index=False)
    print(f"\nSaved → {out_csv}\n{df}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)

    p_vae = sub.add_parser("vae", help="Evaluate VAE reconstructions with masked latents")
    p_vae.add_argument("-c", "--config",       required=True, help="YAML config path")
    p_vae.add_argument("-k", "--ckpt",         required=True, help="VAE checkpoint")
    p_vae.add_argument("-n", "--num_samples",  required=True, type=int)
    p_vae.add_argument("-data", "--data_samples", required=True)
    p_vae.add_argument("-b", "--batch_size",   type=int, default=64)
    p_vae.add_argument("-o", "--csv",          required=True)
    p_vae.add_argument("-r", "--result_name",  required=True)
    p_vae.add_argument("-d", "--device",       default="cuda")
    p_vae.add_argument("--num_workers",        type=int, default=4)

    p_unet = sub.add_parser("unet", help="Evaluate LDM UNet FID")
    p_unet.add_argument("-c", "--config",        required=True)
    p_unet.add_argument("--vae_ckpt",            required=True)
    p_unet.add_argument("--ckpt",                required=True)
    p_unet.add_argument("--csv",                 required=True)
    p_unet.add_argument("--result_name",         required=True)
    p_unet.add_argument("--device",              default="cuda")
    p_unet.add_argument("--batch_size",          type=int, default=200)
    p_unet.add_argument("--num_inference_steps", type=int, default=100)
    p_unet.add_argument("--num_workers",         type=int, default=4)
    p_unet.add_argument("--test_masking",        action="store_true")
    p_unet.add_argument("--use_ema",             action="store_true", default=True)
    p_unet.add_argument("--max_train_images",    type=int, default=50000)
    p_unet.add_argument("--max_test_images",     type=int, default=None)

    args = parser.parse_args()
    (evaluate_vae if args.mode == "vae" else evaluate_unet)(args)
