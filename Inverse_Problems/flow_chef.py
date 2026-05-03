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

    The model operates in *scaled* latent space: z = vae_encode(x) * scale,
    where scale = 1 / global_std (saved in vae_stats.yaml during training).

    Linear flow convention (matching train/flow.py::flow_forward):
        x_t = (1 - t) * z0 + t * noise,   t ∈ [0, 1]
        velocity v = noise - z0   (noise minus clean)

    Tweedie clean estimate at noise level t:
        z0t = z_t - t * v_θ(z_t, t)

    Noise estimate at t:
        z1t = z_t + (1 - t) * v_θ(z_t, t)
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
# FlowChef solver
# ─────────────────────────────────────────────────────────────────────────────

def flow_chef_solve(
    model: FlowLDMWrapper,
    A_op: nn.Module,
    y: torch.Tensor,
    latent_shape: tuple,
    NFE: int,
    step_size: float,
    mask_frac: float,
    latent_dim: int,
    device: torch.device,
    shift_factor: float = 1.0,
    verbose: bool = False,
) -> torch.Tensor:
    """
    FlowChef: Euler ODE with a single data-consistency gradient correction per step.

    Each Euler step:
      1. v_θ = unet(z_t, t)                                [no grad]
      2. z0t  = z_t - t * v_θ                              [clean estimate]
      3. grad = ∇_{z0t} ‖y - A(decode(z0t))‖₂             [DC gradient, norm loss]
      4. z_{t-1} = z_t + dt * v_θ - step_size * grad      [Euler + correction]

    Gradient is masked to the top-k latent dimensions when mask_frac < 1.

    Ref: SD3FlowChef in Research/FlowDPS/sd3_sampler.py (lines 336-419).
    Ours is unconditional (no text / CFG) and operates in our latent space.
    """
    B = y.shape[0]
    z = torch.randn(B, *latent_shape, device=device)
    k = max(1, int(mask_frac * latent_dim))

    timesteps = torch.linspace(1.0, 0.0, NFE + 1, device=device)
    if shift_factor != 1.0:
        timesteps = shift_factor * timesteps / (1.0 + (shift_factor - 1.0) * timesteps)

    pbar = tqdm(range(NFE), desc="FlowChef") if verbose else range(NFE)
    for i in pbar:
        t      = float(timesteps[i])
        t_next = float(timesteps[i + 1])
        dt     = t_next - t  # negative: moving from noise (t=1) to clean (t=0)

        v = model.predict_velocity(z, t)

        # Tweedie clean estimate for linear flow
        z0t = z - t * v

        # Data-consistency gradient (single shot, gradient of L2 norm)
        z0t_tmp = z0t.clone().detach().requires_grad_(True)
        x0t = model.decode(z0t_tmp)               # [-1, 1]
        y_hat = A_op((x0t + 1.0) / 2.0)          # [0, 1]
        loss = torch.linalg.norm((y_hat - y).flatten())
        grad = torch.autograd.grad(loss, z0t_tmp)[0]
        if k < latent_dim:
            grad = apply_k_mask(grad, k)

        with torch.no_grad():
            z = z + dt * v - step_size * grad

    with torch.no_grad():
        x = model.decode(z)
    return (x.clamp(-1.0, 1.0) + 1.0) / 2.0


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_flow_chef(args):
    device = torch.device(args.device)
    print(f"Running FlowChef on {device}")

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
                recon = flow_chef_solve(
                    model=model, A_op=A_op, y=y_batch,
                    latent_shape=latent_shape, NFE=args.NFE,
                    step_size=args.step_size, mask_frac=mask_frac,
                    latent_dim=latent_dim, device=device,
                    shift_factor=args.shift_factor, verbose=True,
                )

                rec = recon.clamp(0.0, 1.0)
                ref = ref_images

                psnr_val  = piq.psnr(rec, ref, data_range=1.0, reduction="mean").item()
                ssim_val  = piq.ssim(rec, ref, data_range=1.0, reduction="mean").item()
                lpips_val = lpips_metric(rec, ref).mean().item()

                print(f"  img {batch_iter:03d} | mask {mask_frac:.2f} | "
                      f"PSNR {psnr_val:.2f} dB  LPIPS {lpips_val:.4f}")

                results.append({
                    "forward_op":  A_op.name,
                    "batch_iter":  batch_iter,
                    "mask_frac":   mask_frac,
                    "k":           max(1, int(mask_frac * latent_dim)),
                    "psnr":        psnr_val,
                    "ssim":        ssim_val,
                    "lpips":       lpips_val,
                    "sigma":       sigma,
                    "NFE":         args.NFE,
                    "step_size":   args.step_size,
                    "shift_factor": args.shift_factor,
                    "vae_scale":   scale,
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
    parser = argparse.ArgumentParser(description="FlowChef: Euler ODE + single DC gradient correction")

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
    parser.add_argument("--NFE",          type=int,   default=100,
                        help="Number of Euler steps (function evaluations)")
    parser.add_argument("--step_size",    type=float, default=50.0,
                        help="DC gradient step size (applied once per Euler step)")
    parser.add_argument("--shift_factor", type=float, default=1.0,
                        help="SD3-style time shift (1.0 = no shift, i.e. uniform spacing)")
    parser.add_argument("--vae_scale",    type=float, default=1.0,
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
    evaluate_flow_chef(args)
