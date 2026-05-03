import os
import argparse
import yaml
import shutil
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from typing import Optional, Tuple, Dict, Any
from diffusers import DDPMScheduler, DDIMScheduler, EulerAncestralDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor
import piq
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")

from .forward_operators import (
    IdentityOp,
    InpaintOp,
    GaussianBlurOp,
    SuperResDownOp,
    LinearMatOp,
    PhaseRetrievalOp,
    make_gaussian_A,
    add_gaussian_noise,
)

# Dataset and model utilities
from .dataset_loader import get_dataset
from models.vae import VAE_MODELS
from models.unet_old import UNET_MODELS
from diffusers.training_utils import EMAModel
from .img_util import save_images_batch

def apply_k_mask(z: torch.Tensor, k: int) -> torch.Tensor:
    """
    Apply a nested dropout-style mask to the latent tensor.
    Keeps the first k dimensions (flattened order), zeros out the rest.
    """
    B, C, H, W = z.shape
    z_flat = z.view(B, -1)
    mask = torch.zeros_like(z_flat)
    mask[:, :k] = 1.0
    z_masked = z_flat * mask
    return z_masked.view(B, C, H, W)



def make_scheduler(kind, num_train_timesteps, beta_schedule="scaled_linear", clip_sample=False):
    """Create scheduler for diffusion"""
    if kind == "ddpm":
        return DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule=beta_schedule,
            clip_sample=clip_sample,
        )
    elif kind == "ddim":
        return DDIMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule=beta_schedule,
            clip_sample=clip_sample,
        )
    elif kind == "euler_ancestral":
        return EulerAncestralDiscreteScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule=beta_schedule,
        )
    else:
        raise ValueError(f"Unknown scheduler: {kind}")


def load_autoencoder(path: str, model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)
    return model.to(device).eval()


def load_unet(ckpt_path: str, model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "ema_state" in ck:
        print("Loading EMA weights")
        ema = EMAModel(model.parameters())
        ema.load_state_dict(ck["ema_state"])
        ema.copy_to(model.parameters())
    else:
        print("Loading regular weights")
        state = ck.get("model_state", ck)
        model.load_state_dict(state)
    return model.to(device).eval()


# ======================= LDPS / PSLD SOLVER ======================= #


def _get_pred_z0_fallback(scheduler, t, latents, noise_pred):
    """Fallback if pred_original_sample not available"""
    a_bar = scheduler.alphas_cumprod.to(latents.device)[t.long()]
    a_bar = a_bar.view(latents.shape[0], *([1] * (latents.ndim - 1)))
    b_bar = 1.0 - a_bar
    z0_hat = (latents - b_bar.sqrt() * noise_pred) / a_bar.sqrt()
    return z0_hat


def apply_k_mask(z: torch.Tensor, k: int) -> torch.Tensor:
    """
    Apply a nested dropout-style mask to the latent tensor.
    Keeps the first k dimensions (flattened order), zeros out the rest.
    """
    B, C, H, W = z.shape
    z_flat = z.view(B, -1)
    mask = torch.zeros_like(z_flat)
    mask[:, :k] = 1.0
    z_masked = z_flat * mask
    return z_masked.view(B, C, H, W)


class PSLDSolver:
    """
    Unified latent diffusion plug-and-play sampler.

    mode = "ldps":  latent DPS
    mode = "psld":  PSLD with correction
    """

    def __init__(self, vae, unet, scheduler, device, mode: str = "ldps"):
        assert mode in ("ldps", "psld")
        self.vae = vae.eval()
        self.unet = unet.eval()
        self.scheduler = scheduler
        self.device = device
        self.mode = mode

    def _decode(self, z):
        # Hardcode scaling = 1.0 as you requested
        return self.vae.decode(z / 1.0, return_dict=False)[0]

    def solve(
        self,
        A: nn.Module,
        y: torch.Tensor,
        num_inference_steps: int = 250,
        sigma_y: float = 1.0,
        step_size: float = 1.0,
        gamma: float = 0.1,
        eta: float = 0.0,
        scale_L : bool =False,
        latents_start: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        verbose: bool = True,
        mask_frac: float = 1.0,          # NEW: fraction of latent dims to keep
        latent_dim: Optional[int] = None,  # NEW: total latent dimensionality
    ):
        device = self.device
        y = y.to(device)
        B = y.shape[0]

        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # init latents
        if latents_start is None:
            latents = randn_tensor((B, 12, 64, 64), generator=generator, device=device)
            latents = latents * self.scheduler.init_noise_sigma
        else:
            latents = latents_start.to(device)

        # apply mask initially
        if mask_frac < 1.0 and latent_dim is not None:
            k = max(1, int(mask_frac * latent_dim))
            latents = apply_k_mask(latents, k)

        extra = {}
        if isinstance(self.scheduler, DDIMScheduler):
            extra["eta"] = eta

        pbar = tqdm(timesteps, disable=not verbose, desc=self.mode.upper())

        for t in pbar:
            latents.requires_grad_(True)

            # predict noise
            latent_in = self.scheduler.scale_model_input(latents, t)
            noise_pred = self.unet(latent_in, t, return_dict=False)[0]

            # scheduler step
            out = self.scheduler.step(noise_pred, t, latents, return_dict=True, **extra)
            z_next = out.prev_sample
            if hasattr(out, "pred_original_sample") and out.pred_original_sample is not None:
                z0_hat = out.pred_original_sample
            else:
                with torch.no_grad():
                    z0_hat = _get_pred_z0_fallback(self.scheduler, t, latents.detach(), noise_pred.detach())

            # decode and rescale to [0,1] for measurement operators
            x0_hat = self._decode(z0_hat)      # [-1, 1]
            x0_hat_01 = (x0_hat + 1) / 2      # [0, 1]

            # LDPS likelihood term
            res = A(x0_hat_01) - y
            if scale_L == False:
                L_data = (res.pow(2)).mean() / max(sigma_y**2, 1e-12)
            else:
                L_data = (res.pow(2)).mean()
            grad = torch.autograd.grad(L_data, latents, retain_graph=(self.mode == "psld"))[0]

            # PSLD correction
            if self.mode == "psld":
                if not hasattr(A, "transpose"):
                    raise RuntimeError("PSLD mode requires A.transpose")
                ATy = A.transpose(y)
                ATAx = A.transpose(A(x0_hat_01))
                x_corr = ATy + (x0_hat_01 - ATAx)
                with torch.no_grad():
                    q = self.vae.encode(x_corr * 2 - 1)  # [0,1] → [-1,1] for VAE
                    z_corr = q.latent_dist.sample() if hasattr(q, "latent_dist") else q
                    z_corr = z_corr * 1.0  # scaling hardcoded
                L_corr = ((z0_hat - z_corr) ** 2).mean()
                grad_corr = torch.autograd.grad(L_corr, latents)[0]
                grad = grad + gamma * grad_corr

            with torch.no_grad():
                latents = (z_next - step_size * grad).detach()

            # reapply mask after update
            if mask_frac < 1.0 and latent_dim is not None:
                k = max(1, int(mask_frac * latent_dim))
                latents = apply_k_mask(latents, k)

            if verbose:
                pbar.set_postfix({"L_data": float(L_data.detach().cpu())})

        with torch.no_grad():
            x = self._decode(latents)
            x = (x + 1) / 2.0
            x = torch.clamp(x, 0, 1)
        return x

def param_tag_str(d: Dict[str, Any]) -> str:
    parts = []
    for k in sorted(d.keys()):
        v = d[k]
        s = str(v).replace(" ", "").replace("{", "").replace("}", "").replace(":", "").replace(",", "_")
        parts.append(f"{k}_{s}")
    return "_".join(parts) if parts else "none"

def evaluate_inverse(args):
    device = torch.device(args.device)
    print(f"Running {args.mode.upper()} on {device}")

    lpips_metric = piq.LPIPS().to(device)

    cfg = yaml.safe_load(open(args.config))
    ds_cfg = cfg["dataset_params"]
    if args.im_path:
        ds_cfg["im_path"] = args.im_path
    sigma_values = args.sigma_values
    mask_fracs = args.mask_fracs

    out_dir = os.path.join(os.getcwd(), "Inverse_Problems", args.result_name)
    os.makedirs(out_dir, exist_ok=True)
    shutil.copy(args.config, os.path.join(out_dir, "config.yaml"))

    scheduler = make_scheduler(
        kind=args.sampler,
        num_train_timesteps=1000,
        beta_schedule=args.beta_schedule,
        clip_sample=False,
    )

    ds_test = get_dataset(ds_cfg, train=False)
    test_loader = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False)

    vae = load_autoencoder(args.vae_ckpt, VAE_MODELS().create_autoencoder_from_dataset(ds_cfg), device)
    unet = load_unet(args.unet_ckpt, UNET_MODELS().create_unet_from_dataset(ds_cfg), device)

    solver = PSLDSolver(vae, unet, scheduler, device, mode=args.mode)

    results = []

    # compute latent_dim once
    x0, _ = next(iter(test_loader))
    with torch.no_grad():
        lat0 = vae.encode(x0.to(device)).latent_dist.sample()
    latent_dim = int(np.prod(lat0.shape[1:]))

    for sigma in sigma_values:
        print(f"\nProcessing {args.forward_op} with sigma={sigma}")
        for batch_iter, (x_batch, _) in enumerate(test_loader):
            x_batch = x_batch.to(device)
            ref_images = x_batch.clamp(-1, 1)
            print(ref_images.shape)

            # Build operator
            if args.forward_op == "denoise":
                A_op = IdentityOp().to(device)
                y_clean = A_op(ref_images)
            elif args.forward_op == "blur":
                A_op = GaussianBlurOp(kernel_size=args.kernel_sizes[0],
                                      sigma=args.blur_sigmas[0],
                                      device=device, channels=ref_images.shape[1]).to(device)
                y_clean = A_op(ref_images)
            elif args.forward_op == "superres":
                A_op = SuperResDownOp(scale=args.sr_scales[0]).to(device)
                y_clean = A_op(ref_images)
            elif args.forward_op == "inpaint":
                A_op = InpaintOp(missing_frac=args.inpaint_fracs[0], example_x=ref_images).to(device)
                y_clean = A_op(ref_images)
            elif args.forward_op == "compressive":
                n = ref_images.shape[1] * ref_images.shape[2] * ref_images.shape[3]
                m = max(1, int(args.meas_fracs[0] * n))
                A = make_gaussian_A(m, n, device)
                A_op = LinearMatOp(A).to(device)
                y_clean = A_op(ref_images)
            elif args.forward_op == "phase":
                n = ref_images.shape[1] * ref_images.shape[2] * ref_images.shape[3]
                m = max(1, int(args.meas_fracs[0] * n))
                A_op = PhaseRetrievalOp(num_measurements=m, example_x=ref_images).to(device)
                y_clean = A_op(ref_images)
            
            elif args.forward_op == "depth_of_field":
                A_op = DepthOfFieldOp(
                    focal_plane_depth=args.dof_focal_depths[0],
                    aperture=args.dof_apertures[0], 
                    focal_length=args.dof_focal_lengths[0],
                    example_x=ref_images,
                    device=device
                ).to(device)
                y_clean = A_op(ref_images)
            
            else:
                raise ValueError

            y_batch = add_gaussian_noise(y_clean, sigma)

            # fresh random latents with the correct shape from the VAE
            B_cur = x_batch.shape[0]
            latents_init = torch.randn(
                B_cur, *lat0.shape[1:], device=device
            ) * solver.scheduler.init_noise_sigma

            # loop over mask fractions
            for mask_frac in mask_fracs:
                k = max(1, int(mask_frac * latent_dim))

                recon = solver.solve(
                    A=A_op,
                    y=y_batch,
                    num_inference_steps=args.sample_steps,
                    sigma_y=sigma,
                    step_size=args.scale,
                    gamma=args.psld_gamma,
                    eta=args.eta,
                    scale_L=args.L_Scale,
                    verbose=True,
                    mask_frac=mask_frac,
                    latent_dim=latent_dim,
                    latents_start=latents_init,
                )

                rec = recon.clamp(0, 1)
                ref = ref_images.clamp(0, 1)

                psnr_val = piq.psnr(rec, ref, data_range=1.0, reduction="mean").item()
                ssim_val = piq.ssim(rec, ref, data_range=1.0, reduction="mean").item()
                lpips_val = lpips_metric(rec, ref).mean().item()

                results.append({
                    "forward_op": A_op.name,
                    "sampler": args.sampler,
                    "eta": args.eta if args.sampler == "ddim" else 0.0,
                    "batch_iter": batch_iter,
                    "mask_frac": mask_frac,
                    "k": k,
                    "psnr": psnr_val,
                    "ssim": ssim_val,
                    "lpips": lpips_val,
                    "psld_gamma": args.psld_gamma,
                    "scale": args.scale,
                    "sample_steps": args.sample_steps,
                    "beta_schedule": args.beta_schedule,
                    "sigma": sigma,
                    "f_op" : args.forward_op,
                    # NEW: operator-specific metadata
                    **{f"op_{k_}": v_ for k_, v_ in A_op.params.items()},
                })

                save_images_batch(
                    recon=rec,
                    best_recon=None,
                    y=y_batch,
                    ref_images=ref,
                    A_op=A_op,
                    params={"sigma": sigma},
                    mask_frac=mask_frac,
                    batch_iter=batch_iter,
                    out_dir=out_dir,
                )

                # save after each batch
                df = pd.DataFrame(results)
                out_csv = os.path.join(out_dir, args.csv)
                df.to_csv(out_csv, index=False)
                print(f"Saved results to: {out_csv}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PSLD/LDPS for inverse problems")
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--vae_ckpt", required=True)
    parser.add_argument("--unet_ckpt", required=True)
    parser.add_argument("-r", "--result_name", required=True)
    parser.add_argument("-o", "--csv", required=True)
    parser.add_argument("--forward_op", type=str, required=True,
                        choices=["denoise", "blur", "superres", "inpaint", "compressive", "phase"])
    parser.add_argument("--psld_gamma", type=float, default=0.1)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("-b", "--batch_size", type=int, default=1)
    parser.add_argument("--sample_steps", type=int, default=500)
    parser.add_argument("--sampler", type=str, default="ddpm",
                        choices=["ddpm", "ddim", "euler_ancestral"])
    parser.add_argument("--L_Scale", type=bool, default=False)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--beta_schedule", type=str, default="scaled_linear")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--sigma_values", type=float, nargs="+", default=[0.0])
    parser.add_argument("--mask_fracs", type=float, nargs="+", default=[1.0])
    parser.add_argument("--kernel_sizes", type=int, nargs="+", default=[9])
    parser.add_argument("--blur_sigmas", type=float, nargs="+", default=[1.5])
    parser.add_argument("--sr_scales", type=int, nargs="+", default=[4])
    parser.add_argument("--inpaint_fracs", type=float, nargs="+", default=[0.5])
    parser.add_argument("--meas_fracs", type=float, nargs="+", default=[0.1])
    parser.add_argument("--mode", choices=["ldps", "psld"], default="psld")
    parser.add_argument("--im_path", type=str, default=None,
                        help="Override dataset im_path from config (e.g. point at a test set)")
    

    args = parser.parse_args()
    evaluate_inverse(args)
