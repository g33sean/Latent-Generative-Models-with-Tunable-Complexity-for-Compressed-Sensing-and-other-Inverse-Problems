import os
import argparse
import yaml
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import inspect
from typing import Optional, Tuple, List, Dict, Any
from diffusers import DDPMScheduler, DDIMScheduler
import piq
from .img_util import save_images_batch, param_tag_str
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")

from .forward_operators import (
    IdentityOp,
    InpaintOp,
    GaussianBlurOp,
    SuperResDownOp,
    LinearMatOp,
    PhaseRetrievalOp,
    HighDynamicRangeOp,
    make_gaussian_A,
    add_gaussian_noise,
)
from .dataset_loader import get_dataset
from models.vae import VAE_MODELS
from diffusers.training_utils import EMAModel


# ─────────────────────────────────────────────────────────────────────────────
# Model loading  (identical to Post_Approx.py)
# ─────────────────────────────────────────────────────────────────────────────

def load_autoencoder(path: str, model: nn.Module, device: torch.device) -> nn.Module:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)
    return model.to(device).eval()


def load_unet(ckpt_path: str, model: nn.Module, device: torch.device) -> nn.Module:
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


# ─────────────────────────────────────────────────────────────────────────────
# Utilities  (identical to Post_Approx.py)
# ─────────────────────────────────────────────────────────────────────────────

def apply_k_mask(z: torch.Tensor, k: int) -> torch.Tensor:
    B, C, H, W = z.shape
    z_flat = z.view(B, -1)
    mask = torch.zeros_like(z_flat)
    mask[:, :k] = 1.0
    return (z_flat * mask).view(B, C, H, W)


def _mask_adam_state_(optimizer: torch.optim.Optimizer, k: Optional[int]) -> None:
    """Zero Adam moments in inactive dims so momentum cannot leak off the submanifold."""
    if k is None:
        return
    for g in optimizer.param_groups:
        for p in g["params"]:
            st = optimizer.state.get(p, None)
            if not st:
                continue
            for key in ("exp_avg", "exp_avg_sq"):
                if key in st and st[key] is not None:
                    v = st[key].view(p.shape[0], -1)
                    if k < v.shape[1]:
                        v[:, k:] = 0.0


def make_scheduler(kind: str, num_train_timesteps: int, beta_schedule: str, clip_sample: bool):
    kind = kind.lower()
    if kind == "ddim":
        return DDIMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule=beta_schedule,
            clip_sample=clip_sample,
            set_alpha_to_one=False,
            prediction_type="epsilon",
        )
    elif kind == "ddpm":
        return DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule=beta_schedule,
            clip_sample=clip_sample,
        )
    else:
        raise ValueError(f"Unknown sampler '{kind}'. Use 'ddpm' or 'ddim'.")


def scheduler_step(scheduler, noise_pred, t, z, eta: float = 0.0):
    sig = inspect.signature(scheduler.step)
    kwargs = {}
    if "eta" in sig.parameters:
        kwargs["eta"] = eta
    return scheduler.step(noise_pred, t, z, **kwargs)


def batch_mse_per_image(x01: torch.Tensor, ref01: torch.Tensor) -> torch.Tensor:
    return ((x01 - ref01) ** 2).flatten(1).mean(dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# ReSample core helpers
# Ported from Research/resample/ldm/models/diffusion/ddim.py
# ─────────────────────────────────────────────────────────────────────────────

def _pixel_optimization(
    y: torch.Tensor,
    x_prime: torch.Tensor,
    A_op: nn.Module,
    iters: int = 2000,
    lr: float = 1e-2,
    eps: float = 1e-3,
) -> torch.Tensor:
    """
    argmin_{x} ||y - A((x+1)/2)||²  starting from x_prime in [-1,1].

    Port of ddim.py::pixel_optimization (lines 337-370).
    Our A_op takes [0,1] images, so we convert opt_var from [-1,1] → [0,1].
    """
    opt_var = x_prime.detach().clone().requires_grad_(True)
    optimizer = torch.optim.AdamW([opt_var], lr=lr)
    y_d = y.detach()
    for _ in range(iters):
        optimizer.zero_grad()
        loss = F.mse_loss(y_d, A_op((opt_var + 1.0) / 2.0))
        loss.backward()
        optimizer.step()
        if loss.item() < eps ** 2:
            break
    return opt_var.detach()


def _latent_optimization(
    z_init: torch.Tensor,
    vae: nn.Module,
    A_op: nn.Module,
    y: torch.Tensor,
    k: int,
    iters: int,
    lr: float,
    eps: float = 1e-3,
    adam_betas: tuple = (0.9, 0.999),
) -> torch.Tensor:
    """
    argmin_z  ||y - A(decode(z))||²  subject to z on the k-dim submanifold.

    apply_k_mask lives ONLY here. Port of ddim.py::latent_optimization (lines 373-432).
    """
    B = z_init.shape[0]
    z = apply_k_mask(z_init.detach(), k).requires_grad_(True)
    optimizer = torch.optim.Adam([z], lr=lr, betas=adam_betas)

    for _ in range(iters):
        optimizer.zero_grad(set_to_none=True)
        x = vae.decode(z).sample          # [-1, 1]
        y_hat = A_op((x + 1.0) / 2.0)    # [0, 1]
        loss = F.mse_loss(y_hat, y.detach())
        loss.backward()

        if z.grad is not None:
            g = z.grad.view(B, -1)
            g[:, k:] = 0.0

        optimizer.step()
        _mask_adam_state_(optimizer, k)

        with torch.no_grad():
            z.data.copy_(apply_k_mask(z.detach(), k))

        if loss.item() < eps ** 2:
            break
        if torch.isnan(z).any():
            return z_init.detach()

    return z.detach()


def _stochastic_resample(
    z0hat: torch.Tensor,
    z_t: torch.Tensor,
    a_prev: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """
    Map measurement-consistent z0hat back onto the noisy manifold at t-1.

    Port of ddim.py::stochastic_resample (lines 435-441).
    Called with a_t=a_prev in the original, so the formula uses a_prev throughout:
        (σ·√ᾱ_prev·z0hat + (1−ᾱ_prev)·z_t) / (σ + 1−ᾱ_prev)
         + noise · √(1/(1/σ + 1/(1−ᾱ_prev)))
    sigma is precomputed in the caller: 40*(1-a_prev)/(1-a_t)*(1-a_t/a_prev)
    """
    ap = a_prev.view(1, 1, 1, 1).to(z0hat.device)
    noise = torch.randn_like(z0hat)
    numer = sigma * ap.sqrt() * z0hat + (1.0 - ap) * z_t
    denom = sigma + 1.0 - ap
    noise_std = torch.sqrt(1.0 / (1.0 / sigma + 1.0 / (1.0 - ap + 1e-8)))
    return numer / denom + noise * noise_std


# ─────────────────────────────────────────────────────────────────────────────
# Core: ReSample reconstruction
#
# Faithful port of Research/resample/ldm/models/diffusion/ddim.py::resample_sampling
#
# Three stages (index_split = num_steps // 3):
#   First 1/3  (index > num_steps - index_split): pure DDIM + DPS
#   Middle 1/3 (index_split ≤ index ≤ num_steps-index_split): pixel opt every 10 steps
#   Last 1/3   (0 < index < index_split): latent opt every 10 steps
#
# apply_k_mask only inside _latent_optimization.
# ─────────────────────────────────────────────────────────────────────────────

def resample_inverse_diffusion(
    vae: nn.Module,
    unet: nn.Module,
    scheduler,
    x_init: torch.Tensor,
    A_op: nn.Module,
    y: torch.Tensor,
    num_steps: int,
    k: int,
    device: torch.device,
    eta: float = 0.0,
    guidance_scale_factor: float = 0.5,
    fidelity_iters: int = 50,
    latent_lr: float = 5e-3,
    latent_eps: float = 1e-3,
    pixel_iters: int = 2000,
    pixel_lr: float = 1e-2,
    pixel_eps: float = 1e-3,
    sigma_scale: float = 40.0,
    adam_betas: tuple = (0.9, 0.999),
    ref_images: Optional[torch.Tensor] = None,
    track_reference: bool = False,
) -> Tuple[
    torch.Tensor,           # recon01  (B,C,H,W)
    torch.Tensor,           # z final latents
    Optional[torch.Tensor], # step_mse (T,B)
    Optional[torch.Tensor], # best_recon01 (B,C,H,W)
    Optional[torch.Tensor], # best_mse (B,)
    Optional[torch.Tensor], # best_step (B,)
]:
    vae.eval()
    unet.eval()

    with torch.no_grad():
        dummy = vae.encode(x_init).latent_dist.sample()
    init_sigma = getattr(scheduler, "init_noise_sigma", 1.0)

    # Full latent noise — no k-masking here (mask only in latent opt)
    z = torch.randn_like(dummy, device=device) * init_sigma
    z = z.requires_grad_(True)

    scheduler.set_timesteps(num_steps, device=device)

    inter_timesteps = 5            # hardcoded in ddim.py line 206
    index_split = num_steps // 3   # splits=3, ddim.py line 265

    # Reference tracking
    do_track = track_reference and (ref_images is not None)
    if do_track:
        ref01 = ref_images.clamp(0, 1).to(device)
        B = ref01.size(0)
        step_mse_list: List[torch.Tensor] = []
        best_mse = torch.full((B,), float("inf"), device=device)
        best_step = torch.zeros(B, dtype=torch.long, device=device)
        best_recon01 = torch.zeros_like(ref01)
    else:
        ref01 = step_mse_list = best_mse = best_step = best_recon01 = None

    for i, t in enumerate(tqdm(scheduler.timesteps, desc="ReSample")):
        index = num_steps - i - 1

        alpha_bar_t = scheduler.alphas_cumprod[t].to(device)
        if i + 1 < num_steps:
            alpha_bar_prev = scheduler.alphas_cumprod[scheduler.timesteps[i + 1]].to(device)
        else:
            alpha_bar_prev = torch.zeros(1, device=device)

        a_t   = alpha_bar_t.view(1, 1, 1, 1)
        a_prev = alpha_bar_prev.view(1, 1, 1, 1)

        # ── 1. UNet forward — NO torch.no_grad so grad flows through for DPS ──
        model_in = scheduler.scale_model_input(z, t) \
            if hasattr(scheduler, "scale_model_input") else z
        noise_pred = unet(model_in, t).sample

        # pseudo_x0 — ddim.py line 541: (x - (1-ab)*e_t) / sqrt(ab)
        # NOTE: sqrt_one_minus_at**2 = (1 - alpha_bar), NOT sqrt(1-alpha_bar)
        pseudo_x0 = (z - (1.0 - a_t) * noise_pred) / a_t.sqrt()

        # DDIM x_{t-1} — detached, just the position update
        with torch.no_grad():
            step_out = scheduler_step(scheduler, noise_pred.detach(), t, z.detach(), eta=eta)
        z_next = step_out.prev_sample.detach()

        # ── 2. DPS correction — grad through UNet → pseudo_x0 → VAE → A_op ──
        # ddim.py line 255-261: measurement_cond_fn(x_prev=img, x_0_hat=pseudo_x0, scale=a_t*0.5)
        x0_hat = vae.decode(pseudo_x0).sample    # [-1, 1]
        y_hat  = A_op((x0_hat + 1.0) / 2.0)     # [0, 1]
        norm   = torch.linalg.norm((y_hat - y).flatten())
        norm_grad = torch.autograd.grad(norm, z)[0]

        dps_scale = float(alpha_bar_t) * guidance_scale_factor
        with torch.no_grad():
            z_next = z_next - dps_scale * norm_grad.detach()
            # NO apply_k_mask — mask only inside latent opt

        # Save DPS-corrected z_{t-1} for stochastic resample
        # ddim.py line 269: x_t = img.detach().clone()  (after DPS correction)
        z_t_saved = z_next.detach().clone()

        # ── 3. Time travel + optimization every 10 steps after first 1/3 ──
        # ddim.py line 268: if index <= (total_steps - index_split) and index > 0
        if index <= (num_steps - index_split) and index > 0 and index % 10 == 0:
            z_travel = z_next.detach().clone()
            pseudo_x0_travel = pseudo_x0.detach()

            # Time travel: 5 extra DDIM steps (ddim.py lines 274-285)
            with torch.no_grad():
                for k_step in range(i, min(i + inter_timesteps, num_steps - 1)):
                    t_inner = scheduler.timesteps[k_step + 1]
                    ab_inner = scheduler.alphas_cumprod[t_inner].view(1, 1, 1, 1).to(device)
                    model_in_inner = scheduler.scale_model_input(z_travel, t_inner) \
                        if hasattr(scheduler, "scale_model_input") else z_travel
                    np_inner = unet(model_in_inner, t_inner).sample
                    step_inner = scheduler_step(scheduler, np_inner, t_inner, z_travel, eta=eta)
                    z_travel = step_inner.prev_sample
                    pseudo_x0_travel = (z_travel - (1.0 - ab_inner) * np_inner) / ab_inner.sqrt()

            # sigma from schedule — ddim.py line 289
            a_t_f   = float(alpha_bar_t)
            a_prev_f = float(alpha_bar_prev)
            sigma = sigma_scale * (1.0 - a_prev_f) / (1.0 - a_t_f + 1e-8) * (1.0 - a_t_f / (a_prev_f + 1e-8))
            sigma = max(sigma, 1e-8)

            if index >= index_split:
                # Middle stage: pixel optimization → encode → stochastic resample
                # ddim.py lines 294-307
                pseudo_x0_pixel = vae.decode(pseudo_x0_travel).sample   # [-1, 1]
                opt_pixel = _pixel_optimization(
                    y, pseudo_x0_pixel, A_op,
                    iters=pixel_iters, lr=pixel_lr, eps=pixel_eps,
                )
                with torch.no_grad():
                    opt_latent = vae.encode(opt_pixel).latent_dist.sample()
                    # NO apply_k_mask on opt_latent
                z_next = _stochastic_resample(opt_latent, z_t_saved, a_prev, sigma)
                z_next = z_next.requires_grad_(True)   # ddim.py line 307

            else:
                # Late stage: latent optimization → stochastic resample
                # ddim.py lines 309-320  (apply_k_mask is INSIDE _latent_optimization)
                z0hat_opt = _latent_optimization(
                    pseudo_x0_travel, vae, A_op, y, k,
                    iters=fidelity_iters, lr=latent_lr,
                    eps=latent_eps, adam_betas=adam_betas,
                )
                z_next = _stochastic_resample(z0hat_opt, z_t_saved, a_prev, sigma)

        # Always keep requires_grad for next iteration's DPS
        z = z_next.detach().requires_grad_(True)

        # ── 4. Reference tracking ──────────────────────────────────────────────
        if do_track:
            with torch.no_grad():
                x01_step = (vae.decode(z).sample + 1.0) / 2.0
                per_img_mse = batch_mse_per_image(x01_step, ref01)
                step_mse_list.append(per_img_mse.detach())
                better = per_img_mse < best_mse
                if better.any():
                    best_mse = torch.where(better, per_img_mse, best_mse)
                    best_recon01[better] = x01_step[better]
                    best_step[better] = i

    # ── Final latent optimization — ddim.py lines 329-332 ────────────────────
    if fidelity_iters > 0:
        z = _latent_optimization(
            z.detach(), vae, A_op, y, k,
            iters=fidelity_iters, lr=latent_lr,
            eps=latent_eps, adam_betas=adam_betas,
        )

    with torch.no_grad():
        recon = vae.decode(z).sample
    recon01 = (recon + 1.0) / 2.0

    if do_track:
        step_mse = torch.stack(step_mse_list, dim=0)
    else:
        step_mse = best_mse = best_step = best_recon01 = None

    return recon01, z, step_mse, best_recon01, best_mse, best_step


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation loop  (identical structure to Post_Approx.py)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_resample(args):
    device = torch.device(args.device)
    print("Running ReSample on", device)

    lpips_metric = piq.LPIPS().to(device)

    cfg = yaml.safe_load(open(args.config))
    ds_cfg = cfg["dataset_params"]
    if args.im_path:
        ds_cfg["im_path"] = args.im_path

    out_dir = os.path.join(os.getcwd(), "Inverse_Problems", args.result_name)
    os.makedirs(out_dir, exist_ok=True)
    shutil.copy(args.config, os.path.join(out_dir, "config.yaml"))

    scheduler = make_scheduler(
        kind=args.sampler,
        num_train_timesteps=args.sample_steps,
        beta_schedule=args.beta_schedule,
        clip_sample=False,
    )
    print(f"Sampler: {args.sampler.upper()}  eta={args.eta if args.sampler == 'ddim' else 'n/a'}")

    ds_test = get_dataset(ds_cfg, train=False)
    test_loader = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False)

    if args.unet_arch == "old":
        from models.unet_old import UNET_MODELS
    else:
        from models.unet import UNET_MODELS

    vae  = load_autoencoder(args.vae_ckpt,  VAE_MODELS().create_autoencoder_from_dataset(ds_cfg), device)
    unet = load_unet(args.unet_ckpt, UNET_MODELS().create_unet_from_dataset(ds_cfg), device)

    x0, _ = next(iter(test_loader))
    with torch.no_grad():
        lat0 = vae.encode(x0.to(device)).latent_dist.sample()
    latent_dim = int(np.prod(lat0.shape[1:]))
    print(f"Latent dim: {latent_dim}")

    def build_op_and_measurements(op_name, x_ref, params):
        B_, C_, H_, W_ = x_ref.shape
        n = C_ * H_ * W_
        if op_name == "denoise":
            A_op_ = IdentityOp().to(device)
        elif op_name == "blur":
            A_op_ = GaussianBlurOp(
                kernel_size=int(params["kernel_size"]), sigma=float(params["sigma"]),
                device=device, channels=C_).to(device)
        elif op_name == "superres":
            A_op_ = SuperResDownOp(scale=int(params["scale"])).to(device)
        elif op_name == "inpaint":
            A_op_ = InpaintOp(missing_frac=float(params["missing_frac"]), example_x=x_ref).to(device)
        elif op_name == "compressive":
            m = max(1, int(float(params["meas_frac"]) * n))
            A_op_ = LinearMatOp(make_gaussian_A(m, n, device)).to(device)
        elif op_name == "phase":
            m = max(1, int(float(params["meas_frac"]) * n))
            A_op_ = PhaseRetrievalOp(num_measurements=m, example_x=x_ref).to(device)
        elif op_name == "high_dynamic_range":
            A_op_ = HighDynamicRangeOp(scale=float(params["hdr_scale"])).to(device)
        else:
            raise ValueError(f"Unknown forward_op: {op_name}")
        return A_op_, A_op_(x_ref)

    def param_grid():
        if args.forward_op == "denoise":
            for sigma in args.sigma_values:
                yield {"sigma": sigma}
        elif args.forward_op == "blur":
            for ks in args.kernel_sizes:
                for bsig in args.blur_sigmas:
                    for sigma in args.sigma_values:
                        yield {"kernel_size": int(ks), "sigma_blur": float(bsig), "sigma": float(sigma)}
        elif args.forward_op == "superres":
            for s in args.sr_scales:
                for sigma in args.sigma_values:
                    yield {"scale": int(s), "sigma": float(sigma)}
        elif args.forward_op == "inpaint":
            for mf in args.inpaint_fracs:
                for sigma in args.sigma_values:
                    yield {"missing_frac": float(mf), "sigma": float(sigma)}
        elif args.forward_op in ["compressive", "phase"]:
            for mf in args.meas_fracs:
                for sigma in args.sigma_values:
                    yield {"meas_frac": float(mf), "sigma": float(sigma)}
        elif args.forward_op == "high_dynamic_range":
            for sigma in args.sigma_values:
                yield {"hdr_scale": float(args.hdr_scale), "sigma": float(sigma)}
        else:
            raise ValueError

    # Resume support
    results = []
    existing_keys = set()
    key_cols = None
    out_csv = os.path.join(out_dir, args.csv)
    if args.resume and os.path.exists(out_csv):
        try:
            df_existing = pd.read_csv(out_csv)
            results = df_existing.to_dict(orient="records")
            base_cols = ["forward_op", "sampler", "eta", "batch_iter",
                         "mask_frac", "k", "sample_steps", "beta_schedule"]
            p_cols  = [c for c in df_existing.columns if c.startswith("p_")]
            op_cols = [c for c in df_existing.columns if c.startswith("op_")]
            key_cols = base_cols + sorted(p_cols) + sorted(op_cols)
            for _, row in df_existing.iterrows():
                existing_keys.add(tuple(row.get(c, None) for c in key_cols))
            print(f"Resume: loaded {len(results)} existing rows from {out_csv}")
        except Exception as exc:
            print(f"Resume failed: {exc}")

    for p in param_grid():
        print(f"\nProcessing {args.forward_op} with params {p}")

        for batch_iter, (x_batch, _) in enumerate(test_loader):
            x_batch = x_batch.to(device)
            ref_images = x_batch.clamp(0, 1)

            if args.forward_op == "denoise":
                build_params = {}
            elif args.forward_op == "blur":
                build_params = {"kernel_size": p["kernel_size"], "sigma": p["sigma_blur"]}
            elif args.forward_op == "superres":
                build_params = {"scale": p["scale"]}
            elif args.forward_op == "inpaint":
                build_params = {"missing_frac": p["missing_frac"]}
            elif args.forward_op == "high_dynamic_range":
                build_params = {"hdr_scale": p["hdr_scale"]}
            else:
                build_params = {"meas_frac": p["meas_frac"]}

            A_op, y_clean = build_op_and_measurements(args.forward_op, ref_images, build_params)
            y_batch = add_gaussian_noise(y_clean, p["sigma"])

            for mask_frac in args.mask_fracs:
                k = max(1, int(mask_frac * latent_dim))

                row_preview = {
                    "forward_op": A_op.name, "sampler": args.sampler,
                    "eta": args.eta if args.sampler == "ddim" else 0.0,
                    "batch_iter": batch_iter, "mask_frac": mask_frac, "k": k,
                    "sample_steps": args.sample_steps, "beta_schedule": args.beta_schedule,
                    **{f"p_{k_}": v_ for k_, v_ in p.items()},
                    **{f"op_{k_}": v_ for k_, v_ in A_op.params.items()},
                }
                if args.resume:
                    if key_cols is None:
                        p_cols  = [c for c in row_preview if c.startswith("p_")]
                        op_cols = [c for c in row_preview if c.startswith("op_")]
                        key_cols = ["forward_op", "sampler", "eta", "batch_iter",
                                    "mask_frac", "k", "sample_steps", "beta_schedule"
                                    ] + sorted(p_cols) + sorted(op_cols)
                    if tuple(row_preview.get(c, None) for c in key_cols) in existing_keys:
                        continue

                recon, _, step_mse, best_recon, best_mse_t, best_step_t = resample_inverse_diffusion(
                    vae=vae, unet=unet, scheduler=scheduler,
                    x_init=ref_images, A_op=A_op, y=y_batch,
                    num_steps=args.sample_steps, k=k, device=device,
                    eta=args.eta,
                    guidance_scale_factor=args.guidance_scale_factor,
                    fidelity_iters=args.fidelity_iters,
                    latent_lr=args.latent_lr,
                    latent_eps=args.latent_eps,
                    pixel_iters=args.pixel_iters,
                    pixel_lr=args.pixel_lr,
                    pixel_eps=args.pixel_eps,
                    sigma_scale=args.sigma_scale,
                    adam_betas=(args.adam_beta1, args.adam_beta2),
                    ref_images=ref_images,
                    track_reference=True,
                )

                rec = recon.clamp(0, 1)
                ref = ref_images

                psnr_val  = piq.psnr(rec, ref, data_range=1.0, reduction="mean").item()
                ssim_val  = piq.ssim(rec, ref, data_range=1.0, reduction="mean").item()
                lpips_val = lpips_metric(rec, ref).mean().item()

                if step_mse is not None and best_mse_t is not None:
                    ref_mse_last = step_mse[-1].mean().item()
                    ref_mse_best = best_mse_t.mean().item()
                    if best_recon is not None:
                        rb = best_recon.clamp(0, 1)
                        psnr_best  = piq.psnr(rb, ref, data_range=1.0, reduction="mean").item()
                        ssim_best  = piq.ssim(rb, ref, data_range=1.0, reduction="mean").item()
                        lpips_best = lpips_metric(rb, ref).mean().item()
                    else:
                        psnr_best = ssim_best = lpips_best = float("nan")
                else:
                    ref_mse_last = ref_mse_best = psnr_best = ssim_best = lpips_best = float("nan")

                print(f"  img {batch_iter:03d} | mask {mask_frac:.2f} | "
                      f"PSNR {psnr_val:.2f} dB  LPIPS {lpips_val:.4f}")

                row = {
                    "forward_op": A_op.name, "sampler": args.sampler,
                    "eta": args.eta if args.sampler == "ddim" else 0.0,
                    "batch_iter": batch_iter, "mask_frac": mask_frac, "k": k,
                    "psnr": psnr_val, "ssim": ssim_val, "lpips": lpips_val,
                    "ref_mse_last": ref_mse_last, "ref_mse_best": ref_mse_best,
                    "psnr_best_over_steps": psnr_best,
                    "ssim_best_over_steps": ssim_best,
                    "lpips_best_over_steps": lpips_best,
                    "guidance_scale_factor": args.guidance_scale_factor,
                    "fidelity_iters": args.fidelity_iters,
                    "latent_lr": args.latent_lr,
                    "pixel_iters": args.pixel_iters,
                    "pixel_lr": args.pixel_lr,
                    "sigma_scale": args.sigma_scale,
                    "sample_steps": args.sample_steps,
                    "beta_schedule": args.beta_schedule,
                    **{f"p_{k_}": v_ for k_, v_ in p.items()},
                    **{f"op_{k_}": v_ for k_, v_ in A_op.params.items()},
                }
                results.append(row)
                if args.resume and key_cols is not None:
                    existing_keys.add(tuple(row.get(c, None) for c in key_cols))

                save_images_batch(
                    recon=rec, best_recon=best_recon, y=y_batch,
                    ref_images=ref_images, A_op=A_op, params=p,
                    mask_frac=mask_frac, batch_iter=batch_iter, out_dir=out_dir,
                )

                df = pd.DataFrame(results)
                df.to_csv(out_csv, index=False)
                print(f"Saved → {out_csv}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ReSample: faithful port of ddim.py::resample_sampling "
                    "with DDIM + DPS (grad through UNet) + time travel + "
                    "pixel/latent optimization + stochastic resample")

    # ── Model / data ─────────────────────────────────────────────────────────
    parser.add_argument("-c", "--config",      required=True)
    parser.add_argument("--vae_ckpt",          required=True)
    parser.add_argument("--unet_ckpt",         required=True)
    parser.add_argument("-r", "--result_name", required=True)
    parser.add_argument("-o", "--csv",         required=True)
    parser.add_argument("--im_path",    type=str,  default=None)
    parser.add_argument("--unet_arch",  choices=["new", "old"], default="new",
                        help="'old' for unet_old.py (LDM_Trial checkpoints)")
    parser.add_argument("-b", "--batch_size",  type=int, default=1)
    parser.add_argument("--device",            type=str, default="cuda")
    parser.add_argument("--beta_schedule",     type=str, default="scaled_linear")

    # ── Sampler ──────────────────────────────────────────────────────────────
    parser.add_argument("--sampler",      choices=["ddpm", "ddim"], default="ddim")
    parser.add_argument("--sample_steps", type=int,   default=100)
    parser.add_argument("--eta",          type=float, default=0.0,
                        help="DDIM stochasticity (0=deterministic). Ignored for DDPM.")

    # ── DPS guidance ─────────────────────────────────────────────────────────
    parser.add_argument("--guidance_scale_factor", type=float, default=0.5,
                        help="DPS scale = alpha_bar_t * guidance_scale_factor (original: a_t*0.5)")

    # ── Latent optimization (inside _latent_optimization) ────────────────────
    parser.add_argument("--fidelity_iters", type=int,   default=50,
                        help="Adam steps per latent optimization call")
    parser.add_argument("--latent_lr",     type=float, default=5e-3)
    parser.add_argument("--latent_eps",    type=float, default=1e-3,
                        help="Early-stop threshold for latent opt (loss < eps²)")
    parser.add_argument("--adam_beta1",    type=float, default=0.9)
    parser.add_argument("--adam_beta2",    type=float, default=0.999)

    # ── Pixel optimization (inside _pixel_optimization) ──────────────────────
    parser.add_argument("--pixel_iters", type=int,   default=2000,
                        help="AdamW steps per pixel optimization call (ddim.py max_iters=2000)")
    parser.add_argument("--pixel_lr",    type=float, default=1e-2,
                        help="AdamW lr for pixel optimization")
    parser.add_argument("--pixel_eps",   type=float, default=1e-3,
                        help="Early-stop threshold for pixel opt (loss < eps²)")

    # ── Stochastic resample ───────────────────────────────────────────────────
    parser.add_argument("--sigma_scale", type=float, default=40.0,
                        help="Multiplier in sigma=scale*(1-a_prev)/(1-a_t)*(1-a_t/a_prev)")

    # ── Measurement / mask ───────────────────────────────────────────────────
    parser.add_argument("--forward_op", required=True,
                        choices=["denoise", "blur", "superres", "inpaint",
                                 "compressive", "phase", "high_dynamic_range"])
    parser.add_argument("--sigma_values",  nargs="+", type=float, default=[0.05])
    parser.add_argument("--mask_fracs",    nargs="+", type=float,
                        default=[1.0, 0.8, 0.6, 0.4, 0.2, 0.1])
    parser.add_argument("--meas_fracs",    nargs="+", type=float, default=[0.1])
    parser.add_argument("--kernel_sizes",  nargs="+", type=int,   default=[9])
    parser.add_argument("--blur_sigmas",   nargs="+", type=float, default=[1.5])
    parser.add_argument("--sr_scales",     nargs="+", type=int,   default=[4])
    parser.add_argument("--inpaint_fracs", nargs="+", type=float, default=[0.5])
    parser.add_argument("--hdr_scale",     type=float, default=2.0)

    # ── Resume ───────────────────────────────────────────────────────────────
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing CSV, skip completed rows")

    args = parser.parse_args()
    evaluate_resample(args)
