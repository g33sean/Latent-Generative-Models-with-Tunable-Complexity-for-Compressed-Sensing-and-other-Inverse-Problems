import os
import argparse
import math
import yaml
import shutil
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import piq
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")

from .forward_operators import (
    IdentityOp, InpaintOp, GaussianBlurOp, SuperResDownOp,
    LinearMatOp, PhaseRetrievalOp, make_gaussian_A, add_gaussian_noise,
)
from .dataset_loader import get_dataset
from models.vae import VAE_MODELS
from diffusers.training_utils import EMAModel
from .img_util import save_images_batch


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_autoencoder(path: str, model: nn.Module, device: torch.device) -> nn.Module:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)
    return model.to(device).eval()


def load_unet(path: str, model: nn.Module, device: torch.device) -> nn.Module:
    ck = torch.load(path, map_location=device, weights_only=False)
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


def apply_k_mask(z: torch.Tensor, k: int) -> torch.Tensor:
    B, C, H, W = z.shape
    z_flat = z.view(B, -1)
    mask = torch.zeros_like(z_flat)
    mask[:, :k] = 1.0
    return (z_flat * mask).view(B, C, H, W)


# ─────────────────────────────────────────────────────────────────────────────
# Flow LDM Wrapper
# ─────────────────────────────────────────────────────────────────────────────

class FlowLDMWrapper(nn.Module):
    """
    Wraps a flow-matching VAE + UNet trained in train/flow.py.

    Scaled latent space: z = vae_encode(x) * scale  (scale = 1 / global_std).

    Linear flow convention (matching train/flow.py::flow_forward):
        x_t = (1 - t) * z0 + t * noise,   t ∈ [0, 1]
        velocity v = noise - z0

    Tweedie clean estimate:  z0t = z_t - t * v_θ(z_t, t)
    Noise estimate:          z1t = z_t + (1 - t) * v_θ(z_t, t)
    """

    def __init__(self, vae: nn.Module, unet: nn.Module, scale: float, device: torch.device):
        super().__init__()
        self.vae = vae.eval().requires_grad_(False)
        self.unet = unet.eval().requires_grad_(False)
        self.scale = scale
        self.device = device

    @torch.no_grad()
    def predict_velocity(self, z_t: torch.Tensor, t_scalar: float) -> torch.Tensor:
        B = z_t.shape[0]
        t_batch = torch.full((B,), t_scalar, device=self.device, dtype=torch.float32)
        return self.unet(sample=z_t, timestep=t_batch).sample

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Scaled latent → image in [-1, 1]. No @no_grad so autograd flows for DC gradient."""
        return self.vae.decode(z / self.scale, return_dict=False)[0]


# ─────────────────────────────────────────────────────────────────────────────
# FlowDPS solver
# ─────────────────────────────────────────────────────────────────────────────

def flow_dps_solve(
    model: FlowLDMWrapper,
    A_op: nn.Module,
    y: torch.Tensor,
    latent_shape: tuple,
    NFE: int,
    dc_step_size: float,
    num_dc_iters: int,
    mask_frac: float,
    latent_dim: int,
    device: torch.device,
    shift_factor: float = 1.0,
    verbose: bool = False,
) -> torch.Tensor:
    """
    FlowDPS: Euler ODE with iterative data-consistency (DC) and stochastic renoising.

    Each Euler step:
      1.  v_θ = unet(z_t, t)                                  [no grad]
      2.  z0t = z_t - t * v_θ                                 [clean estimate]
          z1t = z_t + (1 - t) * v_θ                          [noise estimate]
      3.  Iterative DC on z0t (num_dc_iters gradient steps):
            z0y ← z0y - dc_step_size * ∇_{z0y} ‖y - A(decode(z0y))‖₂
      4.  Mix:    z0y = (1 - t) * z0t + t * z0y
                  (high t → trust DC; low t → trust model's clean estimate)
      5.  Renoise to level t_next:
            noise  = √t_next · z1t  +  √(1 - t_next) · ε,  ε ~ N(0,I)
            z_{t-1} = (1 - t_next) · z0y  +  t_next · noise

    DC gradient is masked to the top-k latent dimensions when mask_frac < 1.

    Ref: SD3FlowDPS in Research/FlowDPS/sd3_sampler.py (lines 246-334).
    Ours is unconditional (no text / CFG) and operates in our latent space.
    """
    B = y.shape[0]
    z = torch.randn(B, *latent_shape, device=device)
    k = max(1, int(mask_frac * latent_dim))

    timesteps = torch.linspace(1.0, 0.0, NFE + 1, device=device)
    if shift_factor != 1.0:
        timesteps = shift_factor * timesteps / (1.0 + (shift_factor - 1.0) * timesteps)

    pbar = tqdm(range(NFE), desc="FlowDPS") if verbose else range(NFE)
    for i in pbar:
        t      = float(timesteps[i])
        t_next = float(timesteps[i + 1])

        v = model.predict_velocity(z, t)

        # Tweedie clean / noise estimates for linear flow
        z0t = (z - t * v).detach()
        z1t = (z + (1.0 - t) * v).detach()

        # ── Iterative data-consistency gradient steps on z0t ──────────────────
        z0y = z0t.clone()
        for _ in range(num_dc_iters):
            z0y = z0y.detach().requires_grad_(True)
            x0y = model.decode(z0y)               # [-1, 1]
            y_hat = A_op((x0y + 1.0) / 2.0)      # [0, 1]
            loss = torch.linalg.norm((y_hat - y).flatten())
            grad = torch.autograd.grad(loss, z0y)[0]
            if k < latent_dim:
                grad = apply_k_mask(grad, k)
            z0y = z0y.detach() - dc_step_size * grad.detach()

        # ── Mix: trust DC more at high t, trust model estimate at low t ───────
        z0y = (1.0 - t) * z0t + t * z0y

        # ── Stochastic renoising to level t_next ──────────────────────────────
        # noise = √t_next · z1t + √(1 - t_next) · ε
        eps = torch.randn_like(z1t)
        t_next_safe = max(t_next, 0.0)
        noise = math.sqrt(t_next_safe) * z1t + math.sqrt(max(1.0 - t_next_safe, 0.0)) * eps
        # z_{t-1} = (1 - t_next) · z0y + t_next · noise
        z = (1.0 - t_next_safe) * z0y + t_next_safe * noise

    with torch.no_grad():
        x = model.decode(z)
    return (x.clamp(-1.0, 1.0) + 1.0) / 2.0


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_flow_dps(args):
    device = torch.device(args.device)
    print(f"Running FlowDPS on {device}")

    lpips_metric = piq.LPIPS().to(device)

    cfg = yaml.safe_load(open(args.config))
    ds_cfg = cfg["dataset_params"]
    if args.im_path:
        ds_cfg["im_path"] = args.im_path

    out_dir = os.path.join(os.getcwd(), "Inverse_Problems", args.result_name)
    os.makedirs(out_dir, exist_ok=True)
    shutil.copy(args.config, os.path.join(out_dir, "config.yaml"))

    # Load scale from vae_stats.yaml (written by train/flow.py) if available
    scale = args.vae_scale
    stats_path = Path(cfg.get("train_params", {}).get("task_name", "")) / "vae_stats.yaml"
    if stats_path.exists():
        data = yaml.safe_load(stats_path.read_text()) or {}
        global_std = data.get("global_std")
        if global_std is not None:
            scale = 1.0 / float(global_std)
            print(f"Loaded VAE scale from {stats_path}: scale={scale:.4f}")
    else:
        print(f"vae_stats.yaml not found, using --vae_scale={scale}")

    if args.unet_arch == "old":
        from models.unet_old import UNET_MODELS
    else:
        from models.unet import UNET_MODELS

    vae  = load_autoencoder(args.vae_ckpt,  VAE_MODELS().create_autoencoder_from_dataset(ds_cfg), device)
    unet = load_unet(args.unet_ckpt, UNET_MODELS().create_unet_from_dataset(ds_cfg), device)
    model = FlowLDMWrapper(vae, unet, scale, device)

    ds_test = get_dataset(ds_cfg, train=False)
    test_loader = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False)

    # Compute latent shape
    x0, _ = next(iter(test_loader))
    with torch.no_grad():
        lat0 = vae.encode(x0.to(device)).latent_dist.sample()
    latent_shape = tuple(lat0.shape[1:])
    latent_dim   = int(np.prod(latent_shape))
    print(f"Latent shape: {latent_shape}  dim: {latent_dim}")

    results = []

    for sigma in args.sigma_values:
        print(f"\nProcessing {args.forward_op} with sigma_y={sigma}")
        for batch_iter, (x_batch, _) in enumerate(test_loader):
            x_batch    = x_batch.to(device)
            ref_images = x_batch.clamp(0.0, 1.0)

            # Build forward operator
            npix = ref_images.shape[1] * ref_images.shape[2] * ref_images.shape[3]
            if args.forward_op == "denoise":
                A_op = IdentityOp().to(device)
            elif args.forward_op == "blur":
                A_op = GaussianBlurOp(
                    kernel_size=args.kernel_sizes[0], sigma=args.blur_sigmas[0],
                    device=device, channels=ref_images.shape[1]).to(device)
            elif args.forward_op == "superres":
                A_op = SuperResDownOp(scale=args.sr_scales[0]).to(device)
            elif args.forward_op == "inpaint":
                A_op = InpaintOp(missing_frac=args.inpaint_fracs[0], example_x=ref_images).to(device)
            elif args.forward_op == "compressive":
                m = max(1, int(args.meas_fracs[0] * npix))
                A_op = LinearMatOp(make_gaussian_A(m, npix, device)).to(device)
            elif args.forward_op == "phase":
                m = max(1, int(args.meas_fracs[0] * npix))
                A_op = PhaseRetrievalOp(num_measurements=m, example_x=ref_images).to(device)
            else:
                raise ValueError(f"Unknown forward_op: {args.forward_op}")

            y_clean = A_op(ref_images)
            y_batch = add_gaussian_noise(y_clean, sigma)

            for mask_frac in args.mask_fracs:
                recon = flow_dps_solve(
                    model=model, A_op=A_op, y=y_batch,
                    latent_shape=latent_shape, NFE=args.NFE,
                    dc_step_size=args.dc_step_size, num_dc_iters=args.num_dc_iters,
                    mask_frac=mask_frac, latent_dim=latent_dim,
                    device=device, shift_factor=args.shift_factor, verbose=True,
                )

                rec = recon.clamp(0.0, 1.0)
                ref = ref_images

                psnr_val  = piq.psnr(rec, ref, data_range=1.0, reduction="mean").item()
                ssim_val  = piq.ssim(rec, ref, data_range=1.0, reduction="mean").item()
                lpips_val = lpips_metric(rec, ref).mean().item()

                print(f"  img {batch_iter:03d} | mask {mask_frac:.2f} | "
                      f"PSNR {psnr_val:.2f} dB  LPIPS {lpips_val:.4f}")

                results.append({
                    "forward_op":   A_op.name,
                    "batch_iter":   batch_iter,
                    "mask_frac":    mask_frac,
                    "k":            max(1, int(mask_frac * latent_dim)),
                    "psnr":         psnr_val,
                    "ssim":         ssim_val,
                    "lpips":        lpips_val,
                    "sigma":        sigma,
                    "NFE":          args.NFE,
                    "dc_step_size": args.dc_step_size,
                    "num_dc_iters": args.num_dc_iters,
                    "shift_factor": args.shift_factor,
                    "vae_scale":    scale,
                    **{f"op_{kk}": vv for kk, vv in A_op.params.items()},
                })

                save_images_batch(
                    recon=rec, best_recon=None, y=y_batch,
                    ref_images=ref, A_op=A_op,
                    params={"sigma": sigma}, mask_frac=mask_frac,
                    batch_iter=batch_iter, out_dir=out_dir,
                )

                df = pd.DataFrame(results)
                df.to_csv(os.path.join(out_dir, args.csv), index=False)
                print(f"Saved → {os.path.join(out_dir, args.csv)}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FlowDPS: Euler ODE with iterative DC gradient steps and stochastic renoising")

    # Model / data
    parser.add_argument("-c", "--config",       required=True)
    parser.add_argument("--vae_ckpt",           required=True)
    parser.add_argument("--unet_ckpt",          required=True)
    parser.add_argument("-r", "--result_name",  required=True)
    parser.add_argument("-o", "--csv",          required=True)
    parser.add_argument("--im_path",    type=str,  default=None)
    parser.add_argument("--unet_arch",  choices=["new", "old"], default="new")
    parser.add_argument("-b", "--batch_size", type=int, default=1)
    parser.add_argument("--device",     type=str,  default="cuda")

    # Flow sampler
    parser.add_argument("--NFE",           type=int,   default=100,
                        help="Number of Euler steps (function evaluations)")
    parser.add_argument("--dc_step_size",  type=float, default=30.0,
                        help="Step size for each iterative DC gradient update")
    parser.add_argument("--num_dc_iters",  type=int,   default=3,
                        help="Number of DC gradient steps per Euler step")
    parser.add_argument("--shift_factor",  type=float, default=1.0,
                        help="SD3-style time shift (1.0 = no shift, i.e. uniform spacing)")
    parser.add_argument("--vae_scale",     type=float, default=1.0,
                        help="Latent scale factor (1/global_std). Auto-loaded from "
                             "vae_stats.yaml in the training task dir if present.")

    # Measurement
    parser.add_argument("--forward_op",   required=True,
                        choices=["denoise", "blur", "superres", "inpaint", "compressive", "phase"])
    parser.add_argument("--sigma_values", nargs="+", type=float, default=[0.05])
    parser.add_argument("--meas_fracs",   nargs="+", type=float, default=[0.1])
    parser.add_argument("--mask_fracs",   nargs="+", type=float,
                        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    parser.add_argument("--kernel_sizes",  nargs="+", type=int,   default=[9])
    parser.add_argument("--blur_sigmas",   nargs="+", type=float, default=[1.5])
    parser.add_argument("--sr_scales",     nargs="+", type=int,   default=[4])
    parser.add_argument("--inpaint_fracs", nargs="+", type=float, default=[0.5])

    args = parser.parse_args()
    evaluate_flow_dps(args)
