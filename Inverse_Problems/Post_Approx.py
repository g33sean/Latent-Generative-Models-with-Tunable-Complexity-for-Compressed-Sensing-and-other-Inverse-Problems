import os
import argparse
import yaml
import shutil
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import inspect
import scipy.ndimage
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
    make_gaussian_A,
    add_gaussian_noise,
)


# ─── dataset utilities ────────────────────────────────────────────────
from .dataset_loader import get_dataset

# ─── model utilities ──────────────────────────────────────────────────
from models.vae import VAE_MODELS
from models.unet import UNET_MODELS
from diffusers.training_utils import EMAModel



def load_autoencoder(path: str, model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    # Explicitly set weights_only=False to suppress warning
    # Use weights_only=True if you trust the checkpoint files
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)
    return model.to(device).eval()


def load_unet(ckpt_path: str, model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    # Explicitly set weights_only=False to suppress warning
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


# ─── must remain (do not change) ──────────────────────────────────────
def apply_k_mask(z: torch.Tensor, k: int) -> torch.Tensor:
    B, C, H, W = z.shape
    z_flat = z.view(B, -1)
    mask = torch.zeros_like(z_flat)
    mask[:, :k] = 1.0
    z_masked = z_flat * mask
    return z_masked.view(B, C, H, W)


def _mask_adam_state_(optimizer: torch.optim.Optimizer, k: int) -> None:
    """
    Zero Adam moments in forbidden flattened dims so momentum cannot leak
    off the nested-dropout submanifold.
    """
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
                        v[:, k:] = 0


# ─── scheduler helpers to support both DDPM and DDIM ──────────────────
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
    """
    Calls scheduler.step(...) and passes eta only if supported (DDIM).
    """
    sig = inspect.signature(scheduler.step)
    kwargs = {}
    if "eta" in sig.parameters:
        kwargs["eta"] = eta
    return scheduler.step(noise_pred, t, z, **kwargs)


def get_sigma_t(scheduler, t_idx, device):
    if hasattr(scheduler, "get_variance"):
        t = scheduler.timesteps[t_idx]
        var = scheduler.get_variance(t)
        return torch.sqrt(var if torch.is_tensor(var) else torch.tensor(float(var), device=device))
    elif hasattr(scheduler, "sigmas"):
        return scheduler.sigmas[t_idx].to(device)
    else:
        return torch.tensor(1.0, device=device)


# =================== Fidelity & Diffusion-loss helpers ================= #
def data_mse_from_latents(z, vae, A_op: nn.Module, y, scale):
    with torch.no_grad():
        x = vae.decode(z / scale).sample
        x = (x + 1) / 2.0  # [0,1]
        y_hat = A_op(x)
        return F.mse_loss(y_hat, y, reduction="mean")


def fidelity_loss(
    z_cur: torch.Tensor,
    z_pred: torch.Tensor,
    vae: torch.nn.Module,
    A_op: nn.Module,
    y: torch.Tensor,
    lambda_f: float,
    scheduler,
    idx: int,
    scale: float,
    k: int,  # NEW: mask index for nested-dropout trust
) -> torch.Tensor:
    # data term
    x_cur = vae.decode(z_cur / scale).sample  # [-1,1]
    x_cur = (x_cur + 1) / 2  # [0,1]
    y_hat = A_op(x_cur)
    mse = F.mse_loss(y_hat, y, reduction="mean")

    # scheduler-aware quadratic pull to z_pred — MASKED the same way you trained
    sigma = get_sigma_t(scheduler, idx, z_cur.device)
    zc = apply_k_mask(z_cur,  k)
    zp = apply_k_mask(z_pred, k)
    reg = 0.5 * F.mse_loss(zc, zp, reduction="mean") / (sigma**2 + 1e-8)
    return lambda_f * mse + reg


# ─── NEW: per-image MSE helper in [0,1] space ─────────────────────────
def batch_mse_per_image(x01: torch.Tensor, ref01: torch.Tensor) -> torch.Tensor:
    """
    Per-image MSE in [0,1].
    Returns shape [B], each entry is MSE for one image.
    """
    return ((x01 - ref01) ** 2).flatten(1).mean(dim=1)


# ─── projected, trust-region, orthogonalized update helpers ───────────
def prepare_candidate(
    z_prev: torch.Tensor,
    z_raw: torch.Tensor,
    k: int,
    t_idx: int,
    scheduler,
    u_prev: torch.Tensor or None,
    trust_c: float,
    ortho_lambda: float,
) -> torch.Tensor:
    """
    Takes a raw candidate z_raw, applies:
      1) hard mask projection,
      2) orthogonalization of (z_raw - z_prev) w.r.t u_prev,
      3) trust-radius scaling based on sigma_t.
    Returns masked candidate.
    """
    device = z_prev.device
    dz = (z_raw - z_prev).detach()
    dz = apply_k_mask(dz, k)

    if (u_prev is not None) and (u_prev.pow(2).sum() > 0) and (dz.pow(2).sum() > 0) and (ortho_lambda != 0.0):
        coeff = (dz * u_prev).sum() / (u_prev.pow(2).sum() + 1e-8)
        dz = dz - ortho_lambda * coeff * u_prev

    sigma_t = float(get_sigma_t(scheduler, t_idx, device))
    trust_radius = trust_c * sigma_t
    dz_norm = dz.flatten(1).norm(dim=1).mean().item() + 1e-8
    alpha = min(1.0, trust_radius / dz_norm)
    z_cand = apply_k_mask(z_prev + alpha * dz, k)
    return z_cand


# ========================== Core Reconstruction ======================== #
def resample_inverse_diffusion(
    vae: torch.nn.Module,
    unet: torch.nn.Module,
    scheduler,
    x_init: torch.Tensor,
    A_op: nn.Module,
    y: torch.Tensor,
    num_steps: int,
    fidelity_iters: int,
    lambda_f: float,
    learning_rate: float,
    k: int,
    scale: float,
    device: torch.device,
    eta: float = 0.0,       # used by DDIM; ignored by DDPM
    trust_c: float = 1.0,   # trust-radius coefficient
    accept_eps: float = 0.0,
    ortho_lambda: float = 1.0,
    adam_betas=(0.9, 0.999),
    adam_eps=1e-8,
    adam_weight_decay=0.0,
    # NEW: reference tracking
    ref_images: Optional[torch.Tensor] = None,     # ground-truth in [0,1]
    track_reference: bool = False,
) -> Tuple[
    torch.Tensor,              # recon01 (B,C,H,W)
    torch.Tensor,              # z
    Optional[torch.Tensor],    # step_mse (T,B)
    Optional[torch.Tensor],    # best_recon01 (B,C,H,W)
    Optional[torch.Tensor],    # best_mse (B,)
    Optional[torch.Tensor],    # best_step (B,)
]:
    """
    Single reconstruction with projected-Adam inner refinements.
    - Keep hard mask at all times (exact nested-dropout ordering).
    - Trust-region scaling ~ sigma_t.
    - Accept strictly monotone data-fidelity improvements only.
    - Optional orthogonalization against last accepted update.
    - Track per-step reference MSE vs ground-truth and best-over-steps recon.

    Returns:
      recon01      : final reconstruction in [0,1], shape [B,C,H,W]
      z            : final latents
      step_mse     : (T,B) tensor of per-step per-image reference MSEs (or None)
      best_recon01 : (B,C,H,W) per-image best-over-steps recon (or None)
      best_mse     : (B,) best-over-steps MSE per image (or None)
      best_step    : (B,) step index achieving best MSE (or None)
    """
    vae.eval()
    unet.eval()
    with torch.no_grad():
        dummy = vae.encode(x_init).latent_dist.sample()
    init_sigma = getattr(scheduler, "init_noise_sigma", 1.0)
    z = torch.randn_like(dummy, device=device) * init_sigma
    z = apply_k_mask(z, k)  # start ON the k-dim manifold

    scheduler.set_timesteps(num_steps, device=device)
    u_prev = None  # last accepted (masked) update direction

    # NEW: reference tracking buffers
    do_track = track_reference and (ref_images is not None)
    if do_track:
        ref01 = ref_images.clamp(0, 1).to(device)
        B = ref01.size(0)
        step_mse_list: List[torch.Tensor] = []
        best_mse = torch.full((B,), float("inf"), device=device)
        best_step = torch.full((B,), -1, dtype=torch.long, device=device)
        best_recon01 = torch.zeros_like(ref01)
    else:
        ref01 = None
        step_mse_list = None
        best_mse = None
        best_step = None
        best_recon01 = None

    for t_idx, t in enumerate(tqdm(scheduler.timesteps, desc="Inverse Diffusion")):
        with torch.no_grad():
            # UNet sees masked latent (tidy; consistent with training subspace)
            model_in = scheduler.scale_model_input(apply_k_mask(z, k), t) \
                if hasattr(scheduler, "scale_model_input") else apply_k_mask(z, k)
            noise_pred = unet(model_in, t).sample
        z_pred = scheduler_step(scheduler, noise_pred, t, z, eta=eta).prev_sample
        z_pred = apply_k_mask(z_pred, k)  # trust anchor must be on-submanifold

        # Adam refinement around z_pred
        z_cur = z_pred.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam(
            [z_cur],
            lr=learning_rate,
            betas=adam_betas,
            eps=adam_eps,
            weight_decay=adam_weight_decay,
        )

        # baseline fidelity at previous accepted z
        mse_prev = float(data_mse_from_latents(apply_k_mask(z, k), vae, A_op, y, scale))
        best_mse_data = mse_prev
        best_z = z.clone().detach()

        for _ in range(max(1, fidelity_iters)):
            optimizer.zero_grad(set_to_none=True)
            loss = fidelity_loss(z_cur, z_pred, vae, A_op, y, lambda_f, scheduler, t_idx, scale, k)
            loss.backward()

            # Mask gradients so forbidden dims never accumulate momentum
            if z_cur.grad is not None:
                g = z_cur.grad.view(z_cur.size(0), -1)
                if k < g.shape[1]:
                    g[:, k:] = 0

            optimizer.step()
            _mask_adam_state_(optimizer, k)

            # After Adam step, project & enforce trust region + optional orthogonalization
            with torch.no_grad():
                z_raw = z_cur.detach()
                z_proj = prepare_candidate(
                    z_prev=z,
                    z_raw=z_raw,
                    k=k,
                    t_idx=t_idx,
                    scheduler=scheduler,
                    u_prev=u_prev,
                    trust_c=trust_c,
                    ortho_lambda=ortho_lambda,
                )
                # overwrite z_cur data with the projected candidate
                z_cur.data.copy_(z_proj)

                # track best candidate by data MSE (monotone acceptance later)
                mse_now = float(data_mse_from_latents(z_proj, vae, A_op, y, scale))
                if mse_now < best_mse_data - 0.0:  # strictly better
                    best_mse_data = mse_now
                    best_z = z_proj.clone().detach()

        # Monotone accept: only move if fidelity improved
        if best_mse_data <= mse_prev - accept_eps:
            du = apply_k_mask(best_z - z, k)
            z = best_z
            if du.pow(2).sum() > 0:
                u_prev = du
        else:
            # no improvement → keep z
            pass

        # Ensure hard mask after each outer step
        z = apply_k_mask(z, k)

        # NEW: per-step reference MSE vs ground-truth (in [0,1])
        if do_track:
            with torch.no_grad():
                x01_step = vae.decode(z / scale).sample
                x01_step = (x01_step + 1) / 2
                per_img_mse = batch_mse_per_image(x01_step, ref01)  # shape [B]
                step_mse_list.append(per_img_mse.detach())

                # update best-over-steps (per image)
                better = per_img_mse < best_mse
                if better.any():
                    best_mse = torch.where(better, per_img_mse, best_mse)
                    best_recon01[better] = x01_step[better]
                    best_step[better] = t_idx

    with torch.no_grad():
        recon = vae.decode(z / scale).sample
    recon01 = (recon + 1) / 2  # [0,1]

    # NEW: finalize tracking tensors
    if do_track:
        step_mse = torch.stack(step_mse_list, dim=0)  # (T,B)
    else:
        step_mse = None
        best_mse = None
        best_step = None
        best_recon01 = None

    return recon01, z, step_mse, best_recon01, best_mse, best_step


# ========================== Evaluation / Sweeps ======================== #
def param_tag_str(d: Dict[str, Any]) -> str:

    parts = []
    for k in sorted(d.keys()):
        v = d[k]
        s = str(v).replace(" ", "").replace("{", "").replace("}", "").replace(":", "").replace(",", "_")
        parts.append(f"{k}_{s}")
    return "_".join(parts) if parts else "none"


def evaluate_inverse(args):
    device = torch.device(args.device)
    print("Running on", device)

    # initialize LPIPS
    lpips_metric = piq.LPIPS().to(device)

    cfg = yaml.safe_load(open(args.config))
    ds_cfg = cfg["dataset_params"]
    sigma_values = args.sigma_values
    mask_fracs = args.mask_fracs

    out_dir = os.path.join(os.getcwd(), "Inverse_Problems", args.result_name)
    os.makedirs(out_dir, exist_ok=True)
    shutil.copy(args.config, os.path.join(out_dir, "config.yaml"))

    scheduler = make_scheduler(
        kind=args.sampler,
        num_train_timesteps=args.sample_steps,
        beta_schedule=args.beta_schedule,
        clip_sample=False,
    )
    print(f"Using sampler: {args.sampler.upper()} (eta={args.eta if args.sampler=='ddim' else 'n/a'})")

    ds_test = get_dataset(ds_cfg, train=False)
    total_test = len(ds_test)

    vae = load_autoencoder(args.vae_ckpt, VAE_MODELS().create_autoencoder_from_dataset(ds_cfg), device)
    unet = load_unet(args.unet_ckpt, UNET_MODELS().create_unet_from_dataset(ds_cfg), device)

    # compute dims and scale
    test_loader = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False)
    scale = 1.0

    dl0 = test_loader
    x0, _ = next(iter(dl0))
    B, C, H, W = x0.shape
    image_dim = C * H * W
    with torch.no_grad():
        lat0 = vae.encode(x0.to(device)).latent_dist.sample()
    latent_dim = int(np.prod(lat0.shape[1:]))

    # Helper: Build operator and corresponding noiseless measurement y
    def build_op_and_measurements(op_name: str,
                                  x_ref: torch.Tensor,
                                  device: torch.device,
                                  params: Dict[str, Any]) -> Tuple[nn.Module, torch.Tensor]:
        B_, C_, H_, W_ = x_ref.shape
        n = C_ * H_ * W_

        if op_name == "denoise":
            A_op_ = IdentityOp().to(device)
            y_ = A_op_(x_ref)
            return A_op_, y_

        if op_name == "blur":
            ks = int(params["kernel_size"])
            sig = float(params["sigma"])
            A_op_ = GaussianBlurOp(kernel_size=ks, sigma=sig, device=device, channels=C_).to(device)
            y_ = A_op_(x_ref)
            return A_op_, y_

        if op_name == "superres":
            s = int(params["scale"])
            A_op_ = SuperResDownOp(scale=s).to(device)
            y_ = A_op_(x_ref)
            return A_op_, y_

        if op_name == "inpaint":
            mf = float(params["missing_frac"])
            A_op_ = InpaintOp(missing_frac=mf, example_x=x_ref).to(device)
            y_ = A_op_(x_ref)
            return A_op_, y_

        if op_name == "compressive":
            frac = float(params["meas_frac"])
            m = max(1, int(frac * n))
            A = make_gaussian_A(m, n, device)
            A_op_ = LinearMatOp(A).to(device)
            y_ = A_op_(x_ref)
            return A_op_, y_

        if op_name == "phase":
            frac = float(params["meas_frac"])
            m = max(1, int(frac * n))
            A = make_gaussian_A(m, n, device)
            A_op_ = PhaseRetrievalOp(num_measurements=m,example_x=x_ref).to(device)
            y_ = A_op_(x_ref)
            return A_op_, y_

        raise ValueError(f"Unknown forward_op: {op_name}")

    # Parameter grid per operator
    def param_grid():
        if args.forward_op == "denoise":
            # only noise sigma sweep
            for sigma in sigma_values:
                yield {"sigma": sigma}
        elif args.forward_op == "blur":
            for ks in args.kernel_sizes:
                for bsig in args.blur_sigmas:
                    for sigma in sigma_values:
                        yield {"kernel_size": int(ks), "sigma_blur": float(bsig), "sigma": float(sigma)}
        elif args.forward_op == "superres":
            for s in args.sr_scales:
                for sigma in sigma_values:
                    yield {"scale": int(s), "sigma": float(sigma)}
        elif args.forward_op == "inpaint":
            for mf in args.inpaint_fracs:
                for sigma in sigma_values:
                    yield {"missing_frac": float(mf), "sigma": float(sigma)}
        elif args.forward_op in ["compressive", "phase"]:
            for mf in args.meas_fracs:
                for sigma in sigma_values:
                    yield {"meas_frac": float(mf), "sigma": float(sigma)}
        else:
            raise ValueError

    # Resume support: load existing CSV (if present) and build a key set
    results = []
    existing_keys = set()
    key_cols = None
    out_csv = os.path.join(out_dir, args.csv)
    if args.resume and os.path.exists(out_csv):
        try:
            df_existing = pd.read_csv(out_csv)
            results = df_existing.to_dict(orient="records")
            base_cols = [
                "forward_op",
                "sampler",
                "eta",
                "batch_iter",
                "mask_frac",
                "k",
                "sample_steps",
                "beta_schedule",
            ]
            p_cols = [c for c in df_existing.columns if c.startswith("p_")]
            op_cols = [c for c in df_existing.columns if c.startswith("op_")]
            key_cols = base_cols + sorted(p_cols) + sorted(op_cols)
            for _, row in df_existing.iterrows():
                existing_keys.add(tuple(row.get(c, None) for c in key_cols))
            print(f"Resume enabled: loaded {len(results)} existing rows from {out_csv}")
        except Exception as exc:
            print(f"Resume enabled but failed to read existing CSV: {exc}")

    for p in param_grid():
        print(f"\nProcessing {args.forward_op} with params {p}")
        
        # Fix: Add batch iteration
        for batch_iter, (x_batch, _) in enumerate(test_loader):
            x_batch = x_batch.to(device)
            ref_images = x_batch.clamp(0, 1)

            # Map generic p -> build_op params
            if args.forward_op == "denoise":
                build_params = {}
            elif args.forward_op == "blur":
                build_params = {"kernel_size": p["kernel_size"], "sigma": p["sigma_blur"]}
            elif args.forward_op == "superres":
                build_params = {"scale": p["scale"]}
            elif args.forward_op == "inpaint":
                build_params = {"missing_frac": p["missing_frac"]}
            elif args.forward_op in ["compressive", "phase"]:
                build_params = {"meas_frac": p["meas_frac"]}
            else:
                raise ValueError

            # Build operator & noiseless measurement, then add measurement noise
            A_op, y_clean = build_op_and_measurements(args.forward_op, ref_images, device, build_params)
            y_batch = add_gaussian_noise(y_clean, p["sigma"])


            for mask_frac in mask_fracs:
                k = max(1, int(mask_frac * latent_dim))

                # Build a unique row key for skip logic
                row_preview = {
                    "forward_op": A_op.name,
                    "sampler": args.sampler,
                    "eta": args.eta if args.sampler == "ddim" else 0.0,
                    "batch_iter": batch_iter,
                    "mask_frac": mask_frac,
                    "k": k,
                    "sample_steps": args.sample_steps,
                    "beta_schedule": args.beta_schedule,
                    **{f"p_{k_}": v_ for k_, v_ in p.items()},
                    **{f"op_{k_}": v_ for k_, v_ in A_op.params.items()},
                }
                if args.resume:
                    if key_cols is None:
                        p_cols = [c for c in row_preview.keys() if c.startswith("p_")]
                        op_cols = [c for c in row_preview.keys() if c.startswith("op_")]
                        key_cols = [
                            "forward_op",
                            "sampler",
                            "eta",
                            "batch_iter",
                            "mask_frac",
                            "k",
                            "sample_steps",
                            "beta_schedule",
                        ] + sorted(p_cols) + sorted(op_cols)
                    if tuple(row_preview.get(c, None) for c in key_cols) in existing_keys:
                        continue
                    continue

                recon, _, ref_step_mse, ref_best_recon, ref_best_mse, ref_best_step = resample_inverse_diffusion(
                    vae, unet, scheduler,
                    x_init=ref_images,
                    A_op=A_op,
                    y=y_batch,
                    num_steps=args.sample_steps,
                    fidelity_iters=args.fidelity_iters,
                    lambda_f=args.lambda_f,
                    learning_rate=args.learning_rate,
                    k=k,
                    scale=scale,
                    device=device,
                    eta=args.eta,  # will be ignored by DDPM
                    trust_c=args.trust_c,
                    accept_eps=args.accept_eps,
                    ortho_lambda=args.ortho_lambda,
                    adam_betas=(args.adam_beta1, args.adam_beta2),
                    adam_eps=args.adam_eps,
                    adam_weight_decay=args.adam_weight_decay,
                    # NEW: track per-step reference MSE vs ground-truth
                    ref_images=ref_images,
                    track_reference=True,
                )

                # metrics in [0,1]
                rec = recon.clamp(0, 1)
                ref = ref_images

                psnr_val = piq.psnr(rec, ref, data_range=1.0, reduction="mean").item()
                ssim_val = piq.ssim(rec, ref, data_range=1.0, reduction="mean").item()

                # FIXED: LPIPS expects inputs in [0,1] range for piq
                # piq.LPIPS expects inputs in [0, 1] range
                lpips_val = lpips_metric(rec, ref).mean().item()

                # NEW: reference MSE summaries (per-step and best-over-steps)
                if ref_step_mse is not None:
                    # ref_step_mse: (T,B), ref_best_mse: (B,)
                    ref_mse_last_mean = ref_step_mse[-1].mean().item()
                    ref_mse_best_mean = ref_best_mse.mean().item()
                    ref_mse_best_min = ref_best_mse.min().item()


                    # (Optionally) compute metrics on the best-over-steps recon
                    if ref_best_recon is not None:
                        rec_best = ref_best_recon.clamp(0, 1)
                        psnr_best = piq.psnr(rec_best, ref, data_range=1.0, reduction="mean").item()
                        ssim_best = piq.ssim(rec_best, ref, data_range=1.0, reduction="mean").item()
                        # FIXED: LPIPS for best reconstruction
                        lpips_best = lpips_metric(rec_best, ref).mean().item()
                    else:
                        psnr_best = ssim_best = lpips_best = float("nan")
                else:
                    ref_mse_last_mean = ref_mse_best_mean = ref_mse_best_min = float("nan")
                    psnr_best = ssim_best = lpips_best = float("nan")
                
                row = {
                    "forward_op": A_op.name,
                    "sampler": args.sampler,
                    "eta": args.eta if args.sampler == "ddim" else 0.0,
                    "batch_iter": batch_iter,
                    "mask_frac": mask_frac,
                    "k": k,
                    "psnr": psnr_val,
                    "ssim": ssim_val,
                    "lpips": lpips_val,
                    # NEW fields
                    "ref_mse_last_mean": ref_mse_last_mean,
                    "ref_mse_best_mean": ref_mse_best_mean,
                    "ref_mse_best_min": ref_mse_best_min,
                    "psnr_best_over_steps": psnr_best,
                    "ssim_best_over_steps": ssim_best,
                    "lpips_best_over_steps": lpips_best,
                    # Optimization hyperparameters
                    "fidelity_iters": args.fidelity_iters,
                    "lambda_f": args.lambda_f,
                    "learning_rate": args.learning_rate,
                    "trust_c": args.trust_c,
                    "accept_eps": args.accept_eps,
                    "ortho_lambda": args.ortho_lambda,
                    # Other hyperparameters
                    "sample_steps": args.sample_steps,
                    "beta_schedule": args.beta_schedule,
                    # op-specific hyperparams we swept
                    **{f"p_{k_}": v_ for k_, v_ in p.items()},
                    **{f"op_{k_}": v_ for k_, v_ in A_op.params.items()},
                    }
                results.append(row)
                if args.resume and key_cols is not None:
                    existing_keys.add(tuple(row.get(c, None) for c in key_cols))
                save_images_batch(
                        recon=rec,
                        best_recon=ref_best_recon if ref_best_recon is not None else None,
                        y=y_batch,
                        ref_images=ref_images,
                        A_op=A_op,
                        params=p,
                        mask_frac=mask_frac,
                        batch_iter=batch_iter,
                        out_dir=out_dir,
                    )



                df = pd.DataFrame(results)
                df.to_csv(out_csv, index=False)
                print(f"\nSaved results to: {out_csv}")


# ================================ Main ================================= #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="General inverse problems via masked diffusion (DDPM/DDIM) with Adam refinement")
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--vae_ckpt", required=True)
    parser.add_argument("--unet_ckpt", required=True)
    parser.add_argument("-r", "--result_name", required=True)
    parser.add_argument("-o", "--csv", required=True)
    parser.add_argument("-b", "--batch_size", type=int, default=100)
    parser.add_argument("--sample_steps", type=int, default=500)
    parser.add_argument("--fidelity_iters", type=int, default=3)
    parser.add_argument("--lambda_f", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=1e-2)  # Adam often likes smaller LR than LBFGS
    parser.add_argument("--beta_schedule", type=str, default="scaled_linear")
    parser.add_argument("--device", type=str, default="cuda")

    # noise levels for additive Gaussian on measurements
    parser.add_argument("--sigma_values", nargs="+", type=float, default=[0.01, 0.05, 0.1, 0.2, 0.3])
    # latent mask fractions
    parser.add_argument("--mask_fracs", nargs="+", type=float, default=[1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.01])

    # Sampler
    parser.add_argument("--sampler", type=str, choices=["ddpm", "ddim"], default="ddim",
                        help="Choose sampler/scheduler: DDPM or DDIM")
    parser.add_argument("--eta", type=float, default=0.1,
                        help="DDIM stochasticity (0.0 = deterministic). Ignored for DDPM.")

    # Trust-region + monotone accept + orthogonalization knobs
    parser.add_argument("--trust_c", type=float, default=1.0, help="Trust radius coefficient (multiplies sigma_t)")
    parser.add_argument("--accept_eps", type=float, default=0.0, help="Minimum MSE improvement to accept an update")
    parser.add_argument("--ortho_lambda", type=float, default=1.0, help="Strength of orthogonalization vs previous update")

    # Adam knobs
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_eps", type=float, default=1e-8)
    parser.add_argument("--adam_weight_decay", type=float, default=0.0)

    # NEW: Forward operator selection & sweep params
    parser.add_argument("--forward_op", choices=["denoise", "blur", "superres", "inpaint", "compressive", "phase"], required=True)
    parser.add_argument("--meas_fracs", nargs="+", type=float, default=[1.0, 0.75, 0.5, 0.25])  # for compressive/phase (m/n)
    parser.add_argument("--kernel_sizes", nargs="+", type=int, default=[11, 21, 31])           # for blur
    parser.add_argument("--blur_sigmas", nargs="+", type=float, default=[1.5, 3.0])             # for blur
    parser.add_argument("--sr_scales", nargs="+", type=int, default=[2, 4])                     # for super-res down
    parser.add_argument("--inpaint_fracs", nargs="+", type=float, default=[0.5, 0.7, 0.9])      # for inpainting (missing %)
    parser.add_argument("--resume", action="store_true", help="Resume from existing CSV and skip completed rows")

    args = parser.parse_args()
    evaluate_inverse(args)
