# Latent Generative Models with Tunable Complexity

Research code for training and evaluating **latent diffusion models whose effective
latent dimensionality can be adjusted at inference time**, with applications to
compressed sensing and general linear inverse problems.

The central idea is to apply **Nested Dropout** during training so that the first *k*
latent dimensions carry the most information. At inference, truncating to *k* dimensions
yields a graceful quality–complexity trade-off with no retraining required.

---

## Method Overview

Training proceeds in two stages:

**Stage 1 — Autoencoder (VAE).** Encodes images into a structured latent space using a
combination of reconstruction loss (L1), KL divergence, LPIPS perceptual loss, a PatchGAN
discriminator (hinge loss + R1 penalty), and a nested-dropout LPIPS regularization term
that enforces ordered latent representations.

**Stage 2 — Latent Diffusion Model (LDM).** A denoising UNet or flow-matching network
trained entirely in the frozen latent space. Supports DDPM/DPM++ sampling schedules,
classifier-free guidance for class-conditional generation, and nested-dropout augmentation
during denoising.

Both stages support nested dropout, so the trained model generates at selectable complexity
levels without any fine-tuning.

---

## Repository Structure

```
.
├── models/
│   ├── vae.py              # AutoencoderKL factory (CelebA, FFHQ, CIFAR, MNIST, ImageNet, LSUN)
│   ├── unet.py             # UNet2DModel factory
│   ├── dit.py              # DiT (Diffusion Transformer) factory
│   ├── discriminator.py    # PatchGAN discriminator for adversarial VAE training
│   └── lpips.py            # LPIPS perceptual loss (VGG16-based)
│
├── train/
│   ├── base_ldm_trainer.py # Shared LDM training infrastructure (base class)
│   ├── vae.py              # VAE trainer (GAN + LPIPS + R1 + nested dropout)
│   ├── unet.py             # DDPM latent diffusion trainer
│   ├── unet_imagenet.py    # Class-conditional LDM for ImageNet (CFG)
│   └── flow.py             # Flow-matching LDM trainer
│
├── inference/
│   └── inference.py        # VAE reconstruction and LDM generation evaluation (FID/SSIM/PSNR)
│
├── Inverse_Problems/
│   ├── forward_operators.py  # CS, inpainting, SR, blur, phase retrieval, HDR operators
│   ├── DAPS.py               # Diffusion Approximation for Posterior Sampling
│   ├── PSLD.py               # Pseudo-inverse Guided Latent Diffusion (PSLD)
│   ├── flow_chef.py          # Flow-CHEF solver
│   ├── flow_dps.py           # Flow-DPS solver
│   ├── flow_ictm.py          # Flow-ICTM solver
│   └── resample.py           # Resampling-based posterior sampler
│
├── utils/
│   └── nd.py               # Nested Dropout (geometric distribution masking)
│
├── data/
│   └── dataset_loader.py   # Unified DataLoader for all supported datasets
│
└── config/
    ├── AutoEncoder/        # VAE training configs per dataset
    └── LDM/                # LDM training configs per dataset
```

---

## Supported Datasets

| Dataset   | Resolution | Latent shape   |
|-----------|------------|----------------|
| MNIST     | 1×32×32    | pixel space    |
| CIFAR-10  | 3×32×32    | 8×8×8          |
| CelebA    | 3×64×64    | 16×8×8         |
| FFHQ      | 3×256×256  | 24×64×64       |
| ImageNet  | 3×256×256  | 12×64×64       |
| LSUN      | 3×256×256  | 24×64×64       |

---

## Installation

```bash
conda env create -f environment.yml
conda activate <env_name>
```

---

## Usage

### 1. Train the Autoencoder

```bash
python train/vae.py --config config/AutoEncoder/FFHQ_ND_24x64x64.yaml
```

Key config fields: `train_params.drop_p` sets the nested dropout probability;
`train_params.lambda_reg` weights the nested-dropout LPIPS regularization.

### 2. Train the Latent Diffusion Model

**DDPM (unconditional):**
```bash
python train/unet.py \
    --config config/LDM/FFHQ_ND_24x64x64.yaml \
    --vae_ckpt_path <path/to/vae.pt>
```

**Flow matching:**
```bash
python train/flow.py \
    --config config/LDM/ND_FLOW.yaml \
    --vae_ckpt_path <path/to/vae.pt>
```

**Class-conditional (ImageNet):**
```bash
python train/unet_imagenet.py \
    --config config/LDM/IMG_ND_24x64x64.yaml \
    --vae_ckpt_path <path/to/vae.pt>
```

### 3. Evaluate Reconstruction and Generation

```bash
# VAE reconstruction sweep over k (nested dropout mask fractions)
python inference/inference.py vae \
    --config config/AutoEncoder/FFHQ_ND_24x64x64.yaml \
    --ckpt <path/to/vae.pt>

# LDM generation + FID
python inference/inference.py unet \
    --config config/LDM/FFHQ_ND_24x64x64.yaml \
    --vae_ckpt <path/to/vae.pt> \
    --unet_ckpt <path/to/unet.pt>
```

### 4. Inverse Problems

```bash
python Inverse_Problems/DAPS.py \
    --config config/LDM/FFHQ_ND_24x64x64.yaml \
    --vae_ckpt <path/to/vae.pt> \
    --ldm_ckpt <path/to/unet.pt> \
    --operator cs --sparsity 0.1
```

Supported operators: `cs` (compressed sensing), `inpainting`, `super_resolution`,
`gaussian_blur`, `motion_blur`, `phase_retrieval`, `hdr`.

**Supported samplers:**

| Script | Method |
|--------|--------|
| `DAPS.py` | DAPS: Improving Diffusion Inverse Problem Solving with Decoupled Noise Annealing |
| `PSLD.py` | Posterior Sampling Latent Diffusion (PSLD) |
| `flow_chef.py` | Flow-CHEF |
| `flow_dps.py` | Flow-DPS |
| `flow_ictm.py` | Flow-ICTM |
| `resample.py` | Resampling-based posterior sampler |

---

## Nested Dropout

`utils/nd.py` implements nested dropout via a geometric distribution over latent
dimension indices. During training each sample is masked to its first *k* dimensions,
where *k ~ Geometric(p)*. This induces an ordered representation in which dimension
importance decreases monotonically — analogous to PCA but learned end-to-end alongside
the generative model.

At inference, setting *k < D* recovers a lower-complexity model at a reduced latent
budget, enabling direct trade-offs between reconstruction fidelity and model capacity
without any additional training.

---

## Citation

```bibtex
@article{gunn2026latent,
  title   = {Latent generative models with tunable complexity for compressed sensing and other inverse problems},
  author  = {Gunn, Sean and Cocola, Jorio and De Candido, Oliver and Chatziafratis, Vaggos and Hand, Paul},
  journal = {arXiv preprint arXiv:2603.07357},
  year    = {2026},
}
```
