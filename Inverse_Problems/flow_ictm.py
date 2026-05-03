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

    Linear flow convention:
        x_t = (1 - t) * z0 + t * noise,   t ∈ [0, 1]
        velocity v = noise - z0

    Tweedie clean estimate:  z0t = z_t - t * v_θ(z_t, t)
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
        """Scaled latent → image in [-1, 1]. No @no_grad so autograd flows for gradients."""
        return self.vae.decode(z / self.scale, return_dict=False)[0]


# ─────────────────────────────────────────────────────────────────────────────
# Hutchinson trace utilities
# ─────────────────────────────────────────────────────────────────────────────

def draw_epsilon(shape: tuple, device: torch.device) -> torch.Tensor:
    """Rademacher random vector for Hutchinson trace estimation."""
    return 2 * (torch.randint(0, 2, shape, device=device).float() - 0.5)


def hutchinson_trace(unet: nn.Module, z: torch.Tensor, t_batch: torch.Tensor,
                     eps: torch.Tensor) -> torch.Tensor:
    """
    Unbiased estimate of tr(J_v) = div(v_θ(z, t)) via one Rademacher vector.

    create_graph=True so the trace is part of the autograd graph — the outer
    autograd.grad(loss, z_) will correctly differentiate through it.

    Returns shape (B,).
    """
    v = unet(sample=z, timestep=t_batch).sample
    v_dot_eps = torch.sum(v * eps)
    jvp = torch.autograd.grad(v_dot_eps, z, create_graph=True)[0]
    return torch.sum(jvp * eps, dim=tuple(range(1, len(z.shape))))


# ─────────────────────────────────────────────────────────────────────────────
# ICTM solver
# ─────────────────────────────────────────────────────────────────────────────

def flow_ictm_solve(
    model: FlowLDMWrapper,
    A_op: nn.Module,
    y: torch.Tensor,
    latent_shape: tuple,
    NFE: int,
    eta: float,
    lamda: float,
    num_iter: int,
    n_trace: int,
    zeta: float,
    nita: float,
    beta: float,
    mask_frac: float,
    latent_dim: int,
    device: torch.device,
    shift_factor: float = 1.0,
    verbose: bool = False,
) -> torch.Tensor:
    """
    ICTM adapted to our latent flow model.

    Reference: Research/ICTM/sampling.py — euler_sampler inside
    get_rectified_flow_inverse_sampler (NeurIPS 2024).

    Time convention: our t ∈ [1, 0] (noise → clean); ICTM's t_ictm ∈ [0, 1].
    Mapping: t_ictm = 1 - t_ours.

    Each Euler step:
      Inner loop (num_iter Adam steps on z):
        1. v = unet(z, t)
        2. znext = z + v * dt                      [predicted next latent]
        3. lik   = lamda * ||A(decode(znext)) - y_interp||²
        4. prior = tr(J_v) * |dt| * zeta           [Hutchinson trace, optional]
        5. loss  = lik + prior  [+ 0.5||z||² at i=0]
        6. grad  = d(loss)/d(z)
        7. grad_ztraj = -nita/t * (-z + (1-t)*v)   [Tweedie correction, i>0]
        8. Adam step with (grad + grad_ztraj)
        9. z += beta * randn_like(z)               [noise injection, optional]
      ODE step: z = z + v_final * dt
    """
    # Efficient/flash attention doesn't implement backward; switch to math backend
    # so autograd can flow through the UNet (needed for d(loss)/d(z) via znext=z+v*dt).
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    B = y.shape[0]
    z = torch.randn(B, *latent_shape, device=device)
    k_top = max(1, int(mask_frac * latent_dim))

    timesteps = torch.linspace(1.0, 0.0, NFE + 1, device=device)
    if shift_factor != 1.0:
        timesteps = shift_factor * timesteps / (1.0 + (shift_factor - 1.0) * timesteps)

    # Initial measurement for interpolation: u0 = A(decode(z_init))
    with torch.no_grad():
        u0 = A_op((model.decode(z).clamp(-1, 1) + 1.0) / 2.0)

    pbar = tqdm(range(NFE), desc="FlowICTM") if verbose else range(NFE)
    for i in pbar:
        t      = float(timesteps[i])
        t_next = float(timesteps[i + 1])
        dt     = t_next - t  # negative: moving from noise to clean

        # Measurement interpolation: at t=1 (pure noise) use u0; at t=0 (clean) use y.
        # ICTM equiv: y_interp = (t_ictm_next)*y + (1 - t_ictm_next)*u0
        #             t_ictm_next = 1 - t_next
        t_next_safe = max(t_next, 0.0)
        y_interp = t_next_safe * u0 + (1.0 - t_next_safe) * y

        z_ = z.detach().clone()

        for _k in range(num_iter):
            z_ = z_.detach().clone().requires_grad_(True)

            B_sz  = z_.shape[0]
            t_bat = torch.full((B_sz,), t, device=device, dtype=torch.float32)

            # Velocity prediction (grad enabled — needed for Hutchinson trace and Tweedie corr)
            v = model.unet(sample=z_, timestep=t_bat).sample

            # Predicted next latent and decoded pixel state
            znext = z_ + v * dt
            xnext = (model.decode(znext).clamp(-1, 1) + 1.0) / 2.0

            # Measurement consistency loss
            lik = lamda * torch.linalg.norm((A_op(xnext) - y_interp).flatten()) ** 2

            # Hutchinson trace prior term (skip if zeta == 0 to save compute)
            prior_change = torch.zeros(1, device=device)
            if zeta != 0.0:
                for _ in range(n_trace):
                    eps = draw_epsilon(z_.shape, device)
                    prior_change = prior_change + hutchinson_trace(model.unet, z_, t_bat, eps).mean()
                prior_change = (prior_change / n_trace) * abs(dt) * zeta

            loss = lik + prior_change
            if i == 0:
                loss = loss + 0.5 * z_.pow(2).sum()

            grad = torch.autograd.grad(loss, z_, create_graph=False)[0]
            if k_top < latent_dim:
                grad = apply_k_mask(grad, k_top)

            # Tweedie trajectory correction (undefined at t=1 where 1/t blows up)
            if i > 0 and t > 1e-6:
                grad_ztraj = (-nita / t) * (-z_ + (1.0 - t) * v.detach())
                if k_top < latent_dim:
                    grad_ztraj = apply_k_mask(grad_ztraj, k_top)
                total_grad = (grad + grad_ztraj).detach()
            else:
                total_grad = grad.detach()

            # Fresh Adam each inner iteration (matches ICTM reference — acts like sign-GD)
            optimizer = torch.optim.Adam([z_], lr=eta)
            optimizer.zero_grad()
            z_.grad = total_grad
            optimizer.step()

            if beta > 0.0:
                z_ = z_.detach() + beta * torch.randn_like(z_)

        # ODE Euler step from the optimized z_
        with torch.no_grad():
            t_bat  = torch.full((z_.shape[0],), t, device=device, dtype=torch.float32)
            v_fin  = model.unet(sample=z_.detach(), timestep=t_bat).sample
            z      = (z_.detach() + v_fin * dt).detach()

    with torch.no_grad():
        x = model.decode(z)
    return (x.clamp(-1.0, 1.0) + 1.0) / 2.0


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_flow_ictm(args):
    device = torch.device(args.device)
    print(f"Running FlowICTM on {device}")

    lpips_metric = piq.LPIPS().to(device)

    cfg = yaml.safe_load(open(args.config))
    ds_cfg = cfg["dataset_params"]
    if args.im_path:
        ds_cfg["im_path"] = args.im_path

    out_dir = os.path.join(os.getcwd(), "Inverse_Problems", args.result_name)
    os.makedirs(out_dir, exist_ok=True)
    shutil.copy(args.config, os.path.join(out_dir, "config.yaml"))

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

    x0, _ = next(iter(test_loader))
    with torch.no_grad():
        lat0 = vae.encode(x0.to(device)).latent_dist.sample()
    latent_shape = tuple(lat0.shape[1:])
    latent_dim   = int(np.prod(latent_shape))
    print(f"Latent shape: {latent_shape}  dim: {latent_dim}")

    results = []
    existing_keys: set = set()
    key_cols = None
    out_csv = os.path.join(out_dir, args.csv)
    if args.resume and os.path.exists(out_csv):
        try:
            df_ex = pd.read_csv(out_csv)
            results = df_ex.to_dict(orient="records")
            op_cols = sorted(c for c in df_ex.columns if c.startswith("op_"))
            key_cols = ["forward_op", "batch_iter", "mask_frac", "k", "sigma"] + op_cols
            for _, row in df_ex.iterrows():
                existing_keys.add(tuple(row.get(c, None) for c in key_cols))
            print(f"Resume: loaded {len(results)} rows from {out_csv}")
        except Exception as exc:
            print(f"Resume load failed: {exc}")

    for sigma in args.sigma_values:
        print(f"\nProcessing {args.forward_op} with sigma_y={sigma}")
        for batch_iter, (x_batch, _) in enumerate(test_loader):
            x_batch    = x_batch.to(device)
            ref_images = x_batch.clamp(0.0, 1.0)

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
                k = max(1, int(mask_frac * latent_dim))
                if args.resume:
                    if key_cols is None:
                        op_c = sorted(f"op_{kk}" for kk in A_op.params)
                        key_cols = ["forward_op", "batch_iter", "mask_frac", "k", "sigma"] + op_c
                    key = tuple([A_op.name, batch_iter, mask_frac, k, sigma]
                                + [A_op.params.get(c[3:]) for c in key_cols[5:]])
                    if key in existing_keys:
                        continue
                recon = flow_ictm_solve(
                    model=model, A_op=A_op, y=y_batch,
                    latent_shape=latent_shape, NFE=args.NFE,
                    eta=args.eta, lamda=args.lamda,
                    num_iter=args.num_iter, n_trace=args.n_trace,
                    zeta=args.zeta, nita=args.nita, beta=args.beta,
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
                    "k":            k,
                    "psnr":         psnr_val,
                    "ssim":         ssim_val,
                    "lpips":        lpips_val,
                    "sigma":        sigma,
                    "NFE":          args.NFE,
                    "eta":          args.eta,
                    "lamda":        args.lamda,
                    "num_iter":     args.num_iter,
                    "n_trace":      args.n_trace,
                    "zeta":         args.zeta,
                    "nita":         args.nita,
                    "beta":         args.beta,
                    "shift_factor": args.shift_factor,
                    "vae_scale":    scale,
                    **{f"op_{kk}": vv for kk, vv in A_op.params.items()},
                })
                if args.resume:
                    existing_keys.add(tuple(results[-1].get(c, None) for c in key_cols))

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
        description="FlowICTM: Iterative Corrupted Trajectory Matching adapted to latent flow LDM")

    # Model / data
    parser.add_argument("-c", "--config",       required=True)
    parser.add_argument("--vae_ckpt",           required=True)
    parser.add_argument("--unet_ckpt",          required=True)
    parser.add_argument("-r", "--result_name",  required=True)
    parser.add_argument("-o", "--csv",          required=True)
    parser.add_argument("--resume", action="store_true",
                        help="Skip rows already present in the output CSV")
    parser.add_argument("--im_path",    type=str,  default=None)
    parser.add_argument("--unet_arch",  choices=["new", "old"], default="new")
    parser.add_argument("-b", "--batch_size", type=int, default=1)
    parser.add_argument("--device",     type=str,  default="cuda")

    # Flow sampler
    parser.add_argument("--NFE",           type=int,   default=100)
    parser.add_argument("--shift_factor",  type=float, default=1.0)
    parser.add_argument("--vae_scale",     type=float, default=1.0)

    # ICTM hyperparameters
    parser.add_argument("--eta",       type=float, default=1e-2,
                        help="Adam learning rate for inner optimization loop")
    parser.add_argument("--lamda",     type=float, default=100.0,
                        help="Measurement consistency weight")
    parser.add_argument("--num_iter",  type=int,   default=1,
                        help="Inner Adam steps per ODE step")
    parser.add_argument("--n_trace",   type=int,   default=1,
                        help="Hutchinson trace samples (only used when zeta > 0)")
    parser.add_argument("--zeta",      type=float, default=1.0,
                        help="Prior (Hutchinson trace) weight; set 0 to disable")
    parser.add_argument("--nita",      type=float, default=1.0,
                        help="Tweedie trajectory correction weight")
    parser.add_argument("--beta",      type=float, default=0.0,
                        help="Noise injection scale per inner step (0 = off)")

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
    evaluate_flow_ictm(args)
