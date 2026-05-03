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
from typing import Optional, Dict, Any
from diffusers import DDPMScheduler
import piq
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")

from .forward_operators import (
    IdentityOp, InpaintOp, GaussianBlurOp, SuperResDownOp,
    LinearMatOp, PhaseRetrievalOp, HighDynamicRangeOp, make_gaussian_A, add_gaussian_noise,
)
from .dataset_loader import get_dataset
from models.vae import VAE_MODELS
from diffusers.training_utils import EMAModel
from .img_util import save_images_batch


def apply_k_mask(z: torch.Tensor, k: int) -> torch.Tensor:
    B, C, H, W = z.shape
    z_flat = z.view(B, -1)
    mask = torch.zeros_like(z_flat)
    mask[:, :k] = 1.0
    return (z_flat * mask).view(B, C, H, W)


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


# ─────────────────────────────────────────────────────────────────────────────
# LDM Wrapper
# ─────────────────────────────────────────────────────────────────────────────

class LDMWrapper(nn.Module):
    """
    Wraps our diffusers VAE + UNet2DModel (DDPM-trained, epsilon prediction) to expose
    a DAPS-compatible tweedie / score / encode / decode interface.

    All outer-loop computations are in EDM space: z = z0 + σ·ε  (s = 1, no scaling).
    VP↔EDM bridge in tweedie():
        z_vp = √ᾱ · z,  ᾱ = 1/(1+σ²)   (identical to VPPrecond c_in in DAPS reference)
    This is mathematically equivalent to DAPS's VPPrecond wrapper but adapted for a
    diffusers UNet trained with standard DDPM (unscaled input, epsilon target).
    """

    def __init__(self, vae: nn.Module, unet: nn.Module,
                 alphas_cumprod: torch.Tensor, device: torch.device, latent_shape: tuple):
        super().__init__()
        self.vae = vae.eval().requires_grad_(False)
        self.unet = unet.eval().requires_grad_(False)
        self.device = device
        self._latent_shape = latent_shape
        self.register_buffer("alphas_cumprod", alphas_cumprod.float())
        sigmas = ((1.0 - alphas_cumprod) / alphas_cumprod).sqrt()
        self.sigma_min = float(sigmas.min().clamp(min=1e-4))
        self.sigma_max = float(sigmas.max())

    def _sigma_to_t(self, sigma: float) -> int:
        sigma_c = float(np.clip(sigma, self.sigma_min, self.sigma_max))
        target_abar = 1.0 / (1.0 + sigma_c ** 2)
        return int((self.alphas_cumprod - target_abar).abs().argmin().item())

    @torch.no_grad()
    def tweedie(self, z: torch.Tensor, sigma: float) -> torch.Tensor:
        """Denoised estimate ẑ₀ in EDM space given noisy z at noise level σ."""
        t = self._sigma_to_t(sigma)
        abar = self.alphas_cumprod[t]
        t_batch = torch.full((z.shape[0],), t, device=z.device, dtype=torch.long)
        # Convert EDM → VP space for the UNet (c_in = √ᾱ)
        eps = self.unet(abar.sqrt() * z, t_batch, return_dict=False)[0]
        # Tweedie in EDM space: ẑ₀ = z − σ·ε  (VPPrecond: c_skip·z + c_out·F_x)
        return z - sigma * eps

    @torch.no_grad()
    def score(self, z: torch.Tensor, sigma: float) -> torch.Tensor:
        return (self.tweedie(z, sigma) - z) / (sigma ** 2)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x in [-1, 1] → latent (EDM space ≈ z0)."""
        return self.vae.encode(x).latent_dist.sample()

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """latent → image in [-1, 1]. No @no_grad so autograd can flow for MCMC."""
        return self.vae.decode(z, return_dict=False)[0]

    def get_in_shape(self) -> tuple:
        return self._latent_shape

    def parameters(self):
        return self.unet.parameters()


# ─────────────────────────────────────────────────────────────────────────────
# Annealing Scheduler
# ─────────────────────────────────────────────────────────────────────────────

class AnnealingScheduler:
    """EDM polynomial sigma annealing schedule (poly-p)."""

    def __init__(self, num_steps: int, sigma_max: float, sigma_min: float, p: int = 7):
        self.num_steps = num_steps
        r = np.linspace(0.0, 1.0, num_steps + 1)
        steps = (sigma_max ** (1.0 / p) + r * (sigma_min ** (1.0 / p) - sigma_max ** (1.0 / p))) ** p
        # Append σ=0 as the final clean target
        self.sigma_steps = torch.tensor(np.append(steps, 0.0), dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Latent Operator Wrapper
# ─────────────────────────────────────────────────────────────────────────────

class LatentForwardOpWrapper:
    """
    Bridges a [0,1]-pixel-space ForwardOp to EDM latent space.
    Pipeline: z (EDM) → decode → [-1,1] → (x+1)/2 → [0,1] → A_op → y
    gradient() lets autograd flow through the VAE decoder to compute dL/dz.
    """

    def __init__(self, A_op, decode_fn):
        self.A_op = A_op
        self.decode_fn = decode_fn  # LDMWrapper.decode — no @no_grad

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.decode_fn(z)
        return self.A_op((x + 1.0) / 2.0)

    def gradient(self, z: torch.Tensor, y: torch.Tensor,
                 return_loss: bool = False):
        z_tmp = z.clone().detach().requires_grad_(True)
        x = self.decode_fn(z_tmp)              # gradient flows through VAE
        res = self.A_op((x + 1.0) / 2.0) - y
        loss = (res ** 2).flatten(1).sum(-1).sum()
        g = torch.autograd.grad(loss, z_tmp)[0].clamp(-1.0, 1.0)
        if return_loss:
            return g, loss.detach()
        return g

    def loss(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.decode_fn(z)
            return ((self.A_op((x + 1.0) / 2.0) - y) ** 2).flatten(1).sum(-1)


# ─────────────────────────────────────────────────────────────────────────────
# MCMC Sampler
# ─────────────────────────────────────────────────────────────────────────────

class MCMCSampler:
    """
    Langevin / HMC posterior sampler in EDM latent space.
    Adapted from Research/DAPS/cores/mcmc.py.

    Score decomposition at each step:
        ∇ log p(z | y, z_t) = data_term + xt_term + prior_term
        data_term  = -∇_z ||A(decode(z)) − y||² / τ²
        xt_term    = (z_t − z) / σ²
        prior_term = (ẑ₀ − z_t) / σ²  [gaussian solver; others available]
    """

    def __init__(self, num_steps: int, lr: float, tau: float = 0.05,
                 lr_min_ratio: float = 0.01, prior_solver: str = "gaussian",
                 prior_sigma_min: float = 1e-2, mc_algo: str = "langevin",
                 momentum: float = 0.9):
        self.num_steps = num_steps
        self.lr = lr
        self.tau = tau
        self.lr_min_ratio = lr_min_ratio
        self.prior_solver = prior_solver
        self.prior_sigma_min = prior_sigma_min
        self.mc_algo = mc_algo
        self.momentum = momentum

    def _get_lr(self, ratio: float) -> float:
        # Linear decay from lr down to lr * lr_min_ratio over annealing
        return (1.0 + ratio * (self.lr_min_ratio - 1.0)) * self.lr

    def _prepare_prior(self, x0hat: torch.Tensor, xt: torch.Tensor,
                       model: LDMWrapper, sigma: float):
        if self.prior_solver == "gaussian":
            self.prior_score = (x0hat - xt).detach() / (sigma ** 2)
        elif self.prior_solver == "score-t":
            self.prior_score = model.score(xt, sigma).detach()
        elif self.prior_solver == "score-min":
            self.prior_score = model.score(x0hat, self.prior_sigma_min).detach()
        elif self.prior_solver == "exact":
            pass  # recomputed per-step in _prior_term
        else:
            raise ValueError(f"Unknown prior_solver: {self.prior_solver}")

    def _prior_term(self, x: torch.Tensor, x0hat: torch.Tensor,
                    xt: torch.Tensor, model: LDMWrapper, sigma: float) -> torch.Tensor:
        if self.prior_solver == "exact":
            return model.score(x, self.prior_sigma_min).detach()
        return self.prior_score

    def _score_fn(self, x, x0hat, xt, model, operator, y, sigma,
                  k: Optional[int] = None):
        data_grad, data_loss = operator.gradient(x, y, return_loss=True)
        if k is not None:
            data_grad = apply_k_mask(data_grad, k)
        data_term = -data_grad / (self.tau ** 2)
        xt_term = (xt - x) / (sigma ** 2)
        prior_term = self._prior_term(x, x0hat, xt, model, sigma)
        return data_term + xt_term + prior_term, data_loss

    def _mc_update(self, x: torch.Tensor, score: torch.Tensor,
                   lr: float, eps: torch.Tensor,
                   k: Optional[int] = None) -> torch.Tensor:
        if self.mc_algo == "langevin":
            return x + lr * score + np.sqrt(2.0 * lr) * eps
        elif self.mc_algo == "hmc":
            step = np.sqrt(lr)
            self.velocity = (self.momentum * self.velocity
                             + step * score
                             + np.sqrt(2.0 * (1.0 - self.momentum)) * eps)
            if k is not None:
                self.velocity = apply_k_mask(self.velocity, k)
            return x + self.velocity * step
        else:
            raise ValueError(f"Unknown mc_algo: {self.mc_algo}")

    def sample(self, xt: torch.Tensor, model: LDMWrapper, x0hat: torch.Tensor,
               operator: LatentForwardOpWrapper, y: torch.Tensor,
               sigma: float, ratio: float, k: Optional[int] = None) -> torch.Tensor:
        lr = self._get_lr(ratio)
        if self.mc_algo == "hmc":
            self.velocity = torch.randn_like(x0hat)
            if k is not None:
                self.velocity = apply_k_mask(self.velocity, k)
        self._prepare_prior(x0hat, xt, model, sigma)
        x = x0hat.clone().detach()
        if k is not None:
            x = apply_k_mask(x, k)
        for _ in range(self.num_steps):
            score, _ = self._score_fn(x, x0hat, xt, model, operator, y, sigma, k=k)
            eps = torch.randn_like(x)
            if k is not None:
                eps = apply_k_mask(eps, k)
            x = self._mc_update(x, score, lr, eps, k=k)
            if k is not None:
                x = apply_k_mask(x, k)
            if torch.isnan(x).any():
                return torch.zeros_like(x0hat)
        return x.detach()


# ─────────────────────────────────────────────────────────────────────────────
# DAPS Solver
# ─────────────────────────────────────────────────────────────────────────────

class DAPSSolver:
    """
    Latent Decoupled Annealing Posterior Sampling.

    Per annealing step at noise level σ:
      1. Euler PF-ODE:  z_t  →  ẑ₀          (reverse diffusion, no_grad)
      2. MCMC:          ẑ₀   →  z₀^y         (posterior sample guided by measurement)
      3. Noise inject:  z    =  z₀^y + ε·σ_{t+1}
    Nested-dropout masking is applied after MCMC when mask_frac < 1.
    """

    def __init__(self, model: LDMWrapper, annealing: AnnealingScheduler,
                 mcmc: MCMCSampler, num_ode_steps: int = 5, sigma_min_ode: float = 0.01):
        self.model = model
        self.annealing = annealing
        self.mcmc = mcmc
        self.num_ode_steps = num_ode_steps
        self.sigma_min_ode = sigma_min_ode

    def _euler_ode(self, z: torch.Tensor, sigma_t: float,
                   k: Optional[int] = None) -> torch.Tensor:
        """
        Heun (2nd-order) integration of the EDM PF-ODE:  dz/dσ = (z − tweedie(z,σ)) / σ
        Runs poly-7 sub-schedule from sigma_t down to sigma_min_ode.
        Two tweedie calls per step for O(h²) accuracy vs Euler's O(h).
        """
        sigma_end = max(self.sigma_min_ode, 1e-4)
        p = 7
        r = np.linspace(0.0, 1.0, self.num_ode_steps + 1)
        sub_sigmas = [(sigma_t ** (1 / p) + ri * (sigma_end ** (1 / p) - sigma_t ** (1 / p))) ** p
                      for ri in r]
        for i in range(len(sub_sigmas) - 1):
            s, s_next = sub_sigmas[i], sub_sigmas[i + 1]
            ds = s_next - s
            # Predictor (Euler)
            z0_s = self.model.tweedie(z, s)
            if k is not None:
                z0_s = apply_k_mask(z0_s, k)
            d_s = (z - z0_s) / s
            z_pred = z + ds * d_s
            if k is not None:
                z_pred = apply_k_mask(z_pred, k)
            # Corrector (use derivative at predicted point)
            z0_next = self.model.tweedie(z_pred, s_next)
            if k is not None:
                z0_next = apply_k_mask(z0_next, k)
            d_next = (z_pred - z0_next) / s_next
            z = z + ds * 0.5 * (d_s + d_next)
            if k is not None:
                z = apply_k_mask(z, k)
        return z

    def solve(self, A_op, y: torch.Tensor, mask_frac: float = 1.0,
              latent_dim: Optional[int] = None, adaptive_k: bool = False,
              verbose: bool = True) -> torch.Tensor:
        device = self.model.device
        y = y.to(device)
        B = y.shape[0]
        in_shape = self.model.get_in_shape()
        sigma_steps = self.annealing.sigma_steps
        n = self.annealing.num_steps

        # Initialize in EDM space: z ~ N(0, σ_max² I)
        z = torch.randn(B, *in_shape, device=device) * float(sigma_steps[0])

        k_end: Optional[int] = None
        k_start: Optional[int] = None
        if mask_frac < 1.0 and latent_dim is not None:
            k_end = max(1, int(mask_frac * latent_dim))
            if adaptive_k:
                k_start = max(1, k_end // 10)
                z = apply_k_mask(z, k_start)
            else:
                z = apply_k_mask(z, k_end)

        wrapped = LatentForwardOpWrapper(A_op, self.model.decode)
        pbar = tqdm(range(n), desc="DAPS", disable=not verbose)

        for step in pbar:
            sigma = float(sigma_steps[step])
            sigma_next = float(sigma_steps[step + 1])
            ratio = step / n

            # Compute k for this step (grows linearly when adaptive_k=True)
            if adaptive_k and k_end is not None:
                k: Optional[int] = max(k_start, int(k_start + (k_end - k_start) * ratio))
            else:
                k = k_end

            # 1. Reverse PF-ODE
            z0_hat = self._euler_ode(z, sigma, k=k)

            # mask z0_hat so MCMC starts on the k-dim submanifold
            if k is not None:
                z0_hat = apply_k_mask(z0_hat, k)

            # 2. MCMC posterior update
            z0_y = self.mcmc.sample(z, self.model, z0_hat, wrapped, y, sigma, ratio, k=k)

            # 3. Nested dropout projection
            if k is not None:
                z0_y = apply_k_mask(z0_y, k)

            # 4. Forward noise injection
            if step < n - 1 and sigma_next > 0.0:
                z = z0_y + torch.randn_like(z0_y) * sigma_next
                if k is not None:
                    z = apply_k_mask(z, k)
            else:
                z = z0_y

            if verbose:
                postfix = {"σ": f"{sigma:.3f}"}
                if adaptive_k and k is not None:
                    postfix["k"] = k
                pbar.set_postfix(postfix)

        with torch.no_grad():
            x = self.model.decode(z)
            x = ((x + 1.0) / 2.0).clamp(0.0, 1.0)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation entry point
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_daps(args):
    device = torch.device(args.device)
    print(f"Running DAPS on {device}")

    lpips_metric = piq.LPIPS().to(device)

    cfg = yaml.safe_load(open(args.config))
    ds_cfg = cfg["dataset_params"]
    if args.im_path:
        ds_cfg["im_path"] = args.im_path

    out_dir = os.path.join(os.getcwd(), "Inverse_Problems", args.result_name)
    os.makedirs(out_dir, exist_ok=True)
    shutil.copy(args.config, os.path.join(out_dir, "config.yaml"))

    ds_test = get_dataset(ds_cfg, train=False)
    test_loader = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False)

    if args.unet_arch == "old":
        from models.unet_old import UNET_MODELS
    else:
        from models.unet import UNET_MODELS

    vae = load_autoencoder(args.vae_ckpt,
                           VAE_MODELS().create_autoencoder_from_dataset(ds_cfg), device)
    unet = load_unet(args.unet_ckpt,
                     UNET_MODELS().create_unet_from_dataset(ds_cfg), device)

    # Alphas_cumprod from the reference VP schedule (used only for σ → t lookup)
    ref_sched = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule=args.beta_schedule,
        clip_sample=False,
    )
    alphas_cumprod = ref_sched.alphas_cumprod.to(device)

    # Compute latent shape from first batch
    x0, _ = next(iter(test_loader))
    with torch.no_grad():
        lat0 = vae.encode(x0.to(device)).latent_dist.sample()
    latent_shape = tuple(lat0.shape[1:])
    latent_dim = int(np.prod(latent_shape))
    print(f"Latent shape: {latent_shape}  dim: {latent_dim}")
    print(f"Annealing σ range: [{args.sigma_min_anneal}, {args.sigma_max_anneal}]  "
          f"steps: {args.num_annealing_steps}")

    model = LDMWrapper(vae, unet, alphas_cumprod, device, latent_shape)
    print(f"Model VP σ range: [{model.sigma_min:.4f}, {model.sigma_max:.2f}]")

    annealing = AnnealingScheduler(
        num_steps=args.num_annealing_steps,
        sigma_max=args.sigma_max_anneal,
        sigma_min=args.sigma_min_anneal,
        p=args.anneal_p,
    )
    mcmc = MCMCSampler(
        num_steps=args.num_mcmc_steps,
        lr=args.mcmc_lr,
        tau=args.tau,
        lr_min_ratio=args.mcmc_lr_min_ratio,
        prior_solver=args.prior_solver,
        mc_algo=args.mc_algo,
        momentum=args.momentum,
    )
    solver = DAPSSolver(
        model=model,
        annealing=annealing,
        mcmc=mcmc,
        num_ode_steps=args.num_ode_steps,
        sigma_min_ode=args.sigma_min_ode,
    )

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
            x_batch = x_batch.to(device)
            ref_images = x_batch.clamp(0.0, 1.0)   # [0, 1] from dataloader

            if args.forward_op == "denoise":
                A_op = IdentityOp().to(device)
            elif args.forward_op == "blur":
                A_op = GaussianBlurOp(
                    kernel_size=args.kernel_sizes[0],
                    sigma=args.blur_sigmas[0],
                    device=device,
                    channels=ref_images.shape[1],
                ).to(device)
            elif args.forward_op == "superres":
                A_op = SuperResDownOp(scale=args.sr_scales[0]).to(device)
            elif args.forward_op == "inpaint":
                A_op = InpaintOp(
                    missing_frac=args.inpaint_fracs[0], example_x=ref_images
                ).to(device)
            elif args.forward_op == "compressive":
                npix = ref_images.shape[1] * ref_images.shape[2] * ref_images.shape[3]
                m = max(1, int(args.meas_fracs[0] * npix))
                A_mat = make_gaussian_A(m, npix, device)
                A_op = LinearMatOp(A_mat).to(device)
            elif args.forward_op == "phase":
                npix = ref_images.shape[1] * ref_images.shape[2] * ref_images.shape[3]
                m = max(1, int(args.meas_fracs[0] * npix))
                A_op = PhaseRetrievalOp(num_measurements=m, example_x=ref_images).to(device)
            elif args.forward_op == "high_dynamic_range":
                A_op = HighDynamicRangeOp(scale=args.hdr_scale).to(device)
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
                recon = solver.solve(
                    A_op=A_op,
                    y=y_batch,
                    mask_frac=mask_frac,
                    latent_dim=latent_dim,
                    adaptive_k=args.adaptive_k,
                    verbose=True,
                )

                rec = recon.clamp(0.0, 1.0)
                ref = ref_images.clamp(0.0, 1.0)

                psnr_val = piq.psnr(rec, ref, data_range=1.0, reduction="mean").item()
                ssim_val = piq.ssim(rec, ref, data_range=1.0, reduction="mean").item()
                lpips_val = lpips_metric(rec, ref).mean().item()

                print(f"  img {batch_iter:03d} | mask_frac {mask_frac:.1f} | "
                      f"PSNR {psnr_val:.2f} dB  LPIPS {lpips_val:.4f}")

                results.append({
                    "forward_op": A_op.name,
                    "batch_iter": batch_iter,
                    "mask_frac": mask_frac,
                    "k": k,
                    "psnr": psnr_val,
                    "ssim": ssim_val,
                    "lpips": lpips_val,
                    "sigma": sigma,
                    "tau": args.tau,
                    "num_annealing_steps": args.num_annealing_steps,
                    "num_ode_steps": args.num_ode_steps,
                    "num_mcmc_steps": args.num_mcmc_steps,
                    "sigma_max_anneal": args.sigma_max_anneal,
                    "sigma_min_anneal": args.sigma_min_anneal,
                    "mcmc_lr": args.mcmc_lr,
                    "mc_algo": args.mc_algo,
                    "prior_solver": args.prior_solver,
                    "adaptive_k": args.adaptive_k,
                    "f_op": args.forward_op,
                    **{f"op_{kk}": vv for kk, vv in A_op.params.items()},
                })
                if args.resume:
                    existing_keys.add(tuple(results[-1].get(c, None) for c in key_cols))

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

                df = pd.DataFrame(results)
                df.to_csv(os.path.join(out_dir, args.csv), index=False)
                print(f"Saved results to: {os.path.join(out_dir, args.csv)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DAPS for latent diffusion inverse problems")

    # ── Model / data ────────────────────────────────────────────────────────
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--vae_ckpt", required=True)
    parser.add_argument("--unet_ckpt", required=True)
    parser.add_argument("-r", "--result_name", required=True)
    parser.add_argument("-o", "--csv", required=True)
    parser.add_argument("--resume", action="store_true",
                        help="Skip rows already present in the output CSV")
    parser.add_argument("--forward_op", required=True,
                        choices=["denoise", "blur", "superres", "inpaint",
                                 "compressive", "phase", "high_dynamic_range"])
    parser.add_argument("--im_path", type=str, default=None)
    parser.add_argument("--unet_arch", choices=["new", "old"], default="new",
                        help="'old' uses unet_old.py (layers_per_block=2) for LDM_Trial checkpoints")
    parser.add_argument("-b", "--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--beta_schedule", type=str, default="scaled_linear")

    # ── Measurement ─────────────────────────────────────────────────────────
    parser.add_argument("--sigma_values", type=float, nargs="+", default=[0.05])
    parser.add_argument("--meas_fracs", type=float, nargs="+", default=[0.1])
    parser.add_argument("--mask_fracs", type=float, nargs="+", default=[1.0])
    parser.add_argument("--adaptive_k", action="store_true",
                        help="Grow k linearly from k//10 → k over annealing steps. "
                             "Without this flag, k is fixed throughout (default).")

    # ── Operator-specific ───────────────────────────────────────────────────
    parser.add_argument("--kernel_sizes", type=int, nargs="+", default=[9])
    parser.add_argument("--blur_sigmas", type=float, nargs="+", default=[1.5])
    parser.add_argument("--sr_scales", type=int, nargs="+", default=[4])
    parser.add_argument("--inpaint_fracs", type=float, nargs="+", default=[0.5])
    parser.add_argument("--hdr_scale",     type=float, default=2.0)

    # ── Annealing schedule ───────────────────────────────────────────────────
    parser.add_argument("--num_annealing_steps", type=int, default=100,
                        help="Number of outer DAPS annealing iterations")
    parser.add_argument("--sigma_max_anneal", type=float, default=80.0,
                        help="Starting noise level (must be ≤ model σ_max ≈ 149 for our VP schedule)")
    parser.add_argument("--sigma_min_anneal", type=float, default=0.1,
                        help="Ending noise level for outer annealing")
    parser.add_argument("--anneal_p", type=int, default=7,
                        help="Polynomial degree for EDM sigma schedule")

    # ── Inner PF-ODE ────────────────────────────────────────────────────────
    parser.add_argument("--num_ode_steps", type=int, default=5,
                        help="Euler steps for the inner PF-ODE at each annealing level")
    parser.add_argument("--sigma_min_ode", type=float, default=0.01,
                        help="ODE denoising target σ (≈ model σ_min)")

    # ── MCMC ────────────────────────────────────────────────────────────────
    parser.add_argument("--num_mcmc_steps", type=int, default=100,
                        help="Langevin / HMC steps per annealing level")
    parser.add_argument("--tau", type=float, default=0.05,
                        help="Measurement noise std for MCMC score (should match sigma_y)")
    parser.add_argument("--mcmc_lr", type=float, default=1e-3,
                        help="Base Langevin step size (decays to mcmc_lr * mcmc_lr_min_ratio)")
    parser.add_argument("--mcmc_lr_min_ratio", type=float, default=0.01)
    parser.add_argument("--mc_algo", type=str, default="langevin",
                        choices=["langevin", "hmc"])
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="HMC momentum coefficient (ignored for Langevin)")
    parser.add_argument("--prior_solver", type=str, default="gaussian",
                        choices=["gaussian", "score-t", "score-min", "exact"],
                        help="How to compute the prior score term in MCMC")

    args = parser.parse_args()
    evaluate_daps(args)
