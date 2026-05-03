import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.ndimage
import math
from typing import Dict, Any, Union

# ======================== General Forward Operators ==================== #
class ForwardOp(nn.Module):
    """
    Callable measurement operator y = A(x). Must be differentiable for backprop.
    x: (B,C,H,W) in [0,1] float32
    Returns y with any shape; MSE compares to y (same shape).
    """
    def __init__(self, name: str, params: Dict[str, Any]):
        super().__init__()
        self._name = name
        self._params = params

    @property
    def name(self) -> str:
        return self._name

    @property
    def params(self) -> Dict[str, Any]:
        return self._params

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def transpose(self, y: torch.Tensor) -> torch.Tensor:
        """
        Computes x = A^T(y).
        """
        raise NotImplementedError


class IdentityOp(ForwardOp):
    def __init__(self):
        super().__init__("denoise", {})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
    
    def transpose(self, y: torch.Tensor) -> torch.Tensor:
        return y


class InpaintOp(ForwardOp):
    """
    Bernoulli missing pixels. Keeps (1-missing_frac) of pixels; missing set to 0.
    Mask is fixed per instance (per resolution) to match y consistently.
    """
    def __init__(self, missing_frac: float, example_x: torch.Tensor):
        assert 0.0 <= missing_frac < 1.0
        B, C, H, W = example_x.shape
        super().__init__("inpaint", {"missing_frac": float(missing_frac)})
        mask = torch.bernoulli(torch.full((1, 1, H, W), 1.0 - missing_frac, device=example_x.device))
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.mask  # broadcast over batch & channels

    def transpose(self, y: torch.Tensor) -> torch.Tensor:
        """For masking, transpose is the same as forward (idempotent)"""
        return y * self.mask


class GaussianBlurOp(ForwardOp):
    """
    Gaussian blur operator with symmetric Gaussian kernel.
    Uses zero-padding (same as functional gaussian_blur_operator).
    For Gaussian kernels (symmetric), transpose = forward.
    """
    def __init__(self, kernel_size: int, sigma: float, device: torch.device, channels: int = 3):
        assert kernel_size % 2 == 1, "kernel_size should be odd"
        super().__init__("blur", {"kernel_size": int(kernel_size), "sigma": float(sigma)})

        self.channels = channels
        self.kernel_size = kernel_size
        self.sigma = sigma

        # Create 2D Gaussian kernel
        ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
        xx, yy = torch.meshgrid([ax, ax], indexing="ij")
        kernel = torch.exp(-(xx**2 + yy**2) / (2. * sigma**2))
        kernel = kernel / kernel.sum()

        # Reshape to (channels, 1, k, k) for depthwise conv
        kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1).to(device)
        self.register_buffer("kernel", kernel)

        # Depthwise conv with zero-padding
        self.conv = nn.Conv2d(
            channels, channels, kernel_size,
            stride=1, padding=kernel_size // 2,  # 👈 zero padding
            bias=False, groups=channels
        ).to(device)

        with torch.no_grad():
            self.conv.weight.copy_(kernel)
            self.conv.weight.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

    def transpose(self, y: torch.Tensor) -> torch.Tensor:
        """
        For Gaussian blur with symmetric kernel, transpose = forward.
        """
        return self.conv(y)


class SuperResDownOp(ForwardOp):
    """
    Downsample by factor s using bicubic. Measurement is low-res image.
    """
    def __init__(self, scale: int):
        assert scale >= 2
        super().__init__("superres_down", {"scale": int(scale)})
        self.scale = int(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Downsample by factor of scale"""
        H, W = x.shape[-2:]
        h, w = H // self.scale, W // self.scale
        return F.interpolate(x, size=(h, w), mode="bicubic",
                             align_corners=False, antialias=True)

    def transpose(self, y: torch.Tensor) -> torch.Tensor:
        """
        Upsample - approximate adjoint of downsampling.
        Not exactly transpose but works well for PSLD.
        """
        h, w = y.shape[-2:]
        H, W = h * self.scale, w * self.scale
        return F.interpolate(y, size=(H, W), mode="bicubic", align_corners=False)


# ===========================================================
#    Efficient Partial Circulant Implementation (Replacing Dense)
# ===========================================================

def make_gaussian_A(m: int, n: int, device: torch.device) -> Dict[str, Any]:
    """
    Replaces the dense Gaussian matrix creation with Partial Circulant parameters.
    Returns a dictionary bundle instead of a raw Tensor.
    """
    # 1. Random filter h ~ N(0, 1/m)
    filt = torch.randn(n, device=device) / math.sqrt(m)

    # 2. Rademacher ±1 sign pattern (D)
    sign_pattern = torch.randint(0, 2, (n,), device=device, dtype=torch.float32)
    sign_pattern = 2.0 * sign_pattern - 1.0 

    # 3. Random subset of n indices (rows)
    indices = torch.randperm(n, device=device)[:m]

    # Return as a param bundle to be consumed by LinearMatOp
    return {
        "type": "partial_circulant",
        "filt": filt,
        "indices": indices,
        "sign_pattern": sign_pattern,
        "m": m,
        "n": n
    }


class LinearMatOp(ForwardOp):
    """
    Previously a dense matrix operator. Now adapted to handle Partial Circulant
    logic seamlessly while maintaining the same class name for integration.
    """
    def __init__(self, A: Union[torch.Tensor, Dict[str, Any]]):
        # Handle the legacy/dense case if a raw tensor is passed (fallback)
        if torch.is_tensor(A):
            m, n = A.shape
            super().__init__("compressive_dense", {"m": int(m), "n": int(n)})
            self.register_buffer("A", A)
            self.is_dense = True
        else:
            # Efficient Partial Circulant Case
            m = A["m"]
            n = A["n"]
            super().__init__("compressive", {"m": int(m), "n": int(n)})
            
            self.register_buffer("filt", A["filt"])
            self.register_buffer("indices", A["indices"].long())
            self.register_buffer("sign_pattern", A["sign_pattern"])
            # Precompute FFT of the filter
            self.register_buffer("filt_fft", torch.fft.fft(A["filt"]))
            
            self.is_dense = False
            self.n = n
            self.m = m

        self.original_shape = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Capture shape on first pass for transpose reconstruction
        if self.original_shape is None:
            self.original_shape = x.shape
            
        if self.is_dense:
            # Legacy dense path
            B = x.shape[0]
            x_flat = x.view(B, -1)
            return x_flat @ self.A.t()
        else:
            # Efficient Partial Circulant path
            return self._forward_circulant(x)

    def transpose(self, y: torch.Tensor) -> torch.Tensor:
        if self.is_dense:
            B = y.shape[0]
            x_flat = y @ self.A
            if self.original_shape is not None:
                return x_flat.view(self.original_shape)
            else:
                # Fallback inference
                n_pixels = x_flat.shape[1]
                hw = int(math.sqrt(n_pixels / 3))
                return x_flat.view(B, 3, hw, hw)
        else:
            return self._transpose_circulant(y)

    # --- Internal Partial Circulant Logic ---
    
    def _forward_circulant(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x_flat = x.view(B, -1)
        
        # 1. Apply diagonal sign pattern
        x_signed = x_flat * self.sign_pattern

        # 2. FFT
        Xf = torch.fft.fft(x_signed, dim=-1)

        # 3. Convolve (multiply in freq domain)
        Yf = Xf * self.filt_fft

        # 4. Inverse FFT (Real part)
        y_full = torch.fft.ifft(Yf, dim=-1).real

        # 5. Subsample
        return y_full[:, self.indices]

    def _transpose_circulant(self, y: torch.Tensor) -> torch.Tensor:
        B = y.shape[0]
        
        # 1. Zero-fill
        z_full = torch.zeros(B, self.n, device=y.device, dtype=y.dtype)
        z_full[:, self.indices] = y

        # 2. FFT
        Zf = torch.fft.fft(z_full, dim=-1)

        # 3. Adjoint Convolution (multiply by conjugate)
        Xf = Zf * torch.conj(self.filt_fft)

        # 4. Inverse FFT
        x_signed = torch.fft.ifft(Xf, dim=-1).real

        # 5. Apply sign pattern
        x_flat = x_signed * self.sign_pattern

        # 6. Reshape
        if self.original_shape is not None:
            return x_flat.view(self.original_shape)
        else:
            # Fallback inference for square image
            hw = int(math.sqrt(self.n / 3))
            return x_flat.view(B, 3, hw, hw)

class PhaseRetrievalOp(ForwardOp):
    """
    Non-linear phase retrieval operator: y = |A(x)|
    where A is a GPU-friendly partial circulant operator with m measurements.

    x: (B, C, H, W)
    y: (B, m)
    """
    def __init__(self, num_measurements: int, example_x: torch.Tensor):
        # example_x just used to infer n and device
        assert example_x.ndim == 4, "example_x must be (B, C, H, W)"
        B, C, H, W = example_x.shape
        n = C * H * W
        m = int(num_measurements)
        assert 1 <= m <= n, "num_measurements must be between 1 and total number of pixels"

        super().__init__("phase_retrieval", {"m": m, "n": n})

        device = example_x.device
        dtype = example_x.dtype

        # Diagonal ±1 sign pattern (Rademacher)
        sign_pattern = torch.randint(
            0, 2, (n,), device=device, dtype=dtype
        )
        sign_pattern = 2.0 * sign_pattern - 1.0  # {−1, +1}

        # Random filter h ~ N(0, 1/m)
        filt = torch.randn(n, device=device, dtype=dtype) / math.sqrt(m)

        # Random subset of indices (rows)
        indices = torch.randperm(n, device=device)[:m]

        # Register as buffers so they move with .to(device) and save in state_dict
        self.register_buffer("sign_pattern", sign_pattern)
        self.register_buffer("filt_fft", torch.fft.fft(filt))
        self.register_buffer("indices", indices.long())

        self.n = n
        self.m = m
        # Store spatial shape for transpose reshaping
        self.original_shape = (C, H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        returns: |A x|  with shape (B, m)
        """
        B = x.shape[0]
        # Flatten spatial+channel dims
        x_flat = x.view(B, -1)  # (B, n)

        # 1. Apply diagonal sign pattern
        x_signed = x_flat * self.sign_pattern  # (B, n)

        # 2. FFT
        Xf = torch.fft.fft(x_signed, dim=-1)   # (B, n), complex

        # 3. Convolution in frequency: multiply by filt_fft
        Yf = Xf * self.filt_fft                # (B, n), complex

        # 4. Inverse FFT → real domain
        y_full = torch.fft.ifft(Yf, dim=-1).real  # (B, n), real

        # 5. Subsample m rows
        y_lin = y_full[:, self.indices]        # (B, m)

        # 6. Phase retrieval measurements: magnitude
        return torch.abs(y_lin)  

# ======================== Helper Utils ======================== #

def add_gaussian_noise(y: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return y
    return y + sigma * torch.randn_like(y)


# ======================== Custom Blur Utils ==================== #

class MotionBlurOp(ForwardOp):
    """
    Linear motion blur operator.
    Forward: convolve with motion kernel.
    Transpose: convolve with flipped kernel (proper adjoint).
    """
    def __init__(self, length: int = 31, angle: float = 0.0,
                 device: torch.device = torch.device("cuda"), channels: int = 3):
        assert length >= 3, "Blur length must be at least 3"
        super().__init__("motion_blur", {"length": int(length), "angle": float(angle)})

        self.length = length
        self.angle = angle
        self.channels = channels

        kernel_1d = torch.zeros(length, device=device)
        kernel_1d[:] = 1.0 / length

        size = length
        kernel_2d = torch.zeros(size, size, device=device)
        center = size // 2
        kernel_2d[center, :] = kernel_1d

        if angle != 0.0:
            kernel_2d = self._rotate_kernel(kernel_2d, angle)

        kernel_2d = kernel_2d / (kernel_2d.sum() + 1e-8)

        kernel_forward = kernel_2d.view(1, 1, size, size).expand(channels, 1, -1, -1).clone()
        kernel_transpose = torch.flip(kernel_2d, dims=[-1, -2]).view(1, 1, size, size).expand(channels, 1, -1, -1).clone()

        self.register_buffer("kernel_forward", kernel_forward)
        self.register_buffer("kernel_transpose", kernel_transpose)
        self.pad = size // 2

    def _rotate_kernel(self, kernel: torch.Tensor, angle: float) -> torch.Tensor:
        h, w = kernel.shape
        theta_rad = torch.tensor(angle * np.pi / 180.0, device=kernel.device)
        cos_t, sin_t = torch.cos(theta_rad), torch.sin(theta_rad)
        rot_matrix = torch.tensor([
            [cos_t, -sin_t, 0],
            [sin_t,  cos_t, 0]
        ], device=kernel.device, dtype=kernel.dtype).unsqueeze(0)
        grid = F.affine_grid(rot_matrix, (1, 1, h, w), align_corners=False)
        kernel_4d = kernel.unsqueeze(0).unsqueeze(0)
        rotated = F.grid_sample(kernel_4d, grid, mode='bilinear',
                                padding_mode='zeros', align_corners=False)
        return rotated.squeeze()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_padded = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode='reflect')
        return F.conv2d(x_padded, self.kernel_forward, groups=self.channels)

    def transpose(self, y: torch.Tensor) -> torch.Tensor:
        y_padded = F.pad(y, (self.pad, self.pad, self.pad, self.pad), mode='reflect')
        return F.conv2d(y_padded, self.kernel_transpose, groups=self.channels)


class HighDynamicRangeOp(ForwardOp):
    def __init__(self, scale: float = 2.0):
        super().__init__("high_dynamic_range", {"scale": scale})
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x * self.scale, -1.0, 1.0)

    def transpose(self, y: torch.Tensor) -> torch.Tensor:
        # Adjoint of the scaling component; clip has no linear adjoint so y/scale is the standard approximation
        return y / self.scale


class Blurkernel(nn.Module):
    def __init__(self, blur_type='gaussian', kernel_size=31, std=3.0, channels=3, device=None):
        super().__init__()
        assert kernel_size % 2 == 1
        self.blur_type = blur_type
        self.kernel_size = kernel_size
        self.std = std
        self.pad = nn.ReflectionPad2d(kernel_size // 2)
        self.conv = nn.Conv2d(channels, channels, kernel_size,
                              stride=1, padding=0, bias=False, groups=channels)
        if device is not None:
            self.to(device)
        self.weights_init()

    def forward(self, x):
        return self.conv(self.pad(x))

    def weights_init(self):
        ks, sigma = self.kernel_size, self.std
        if self.blur_type == "gaussian":
            n = np.zeros((ks, ks), dtype=np.float32)
            n[ks // 2, ks // 2] = 1.0
            k = scipy.ndimage.gaussian_filter(n, sigma=sigma).astype(np.float32)
        else:
            k = np.zeros((ks, ks), dtype=np.float32)
            k[ks // 2, :] = 1.0
        k = torch.from_numpy(k)
        k = (k / (k.sum() + 1e-8)).view(1, 1, ks, ks).repeat(self.conv.in_channels, 1, 1, 1).to(self.conv.weight.device)
        with torch.no_grad():
            self.conv.weight.copy_(k)
            self.conv.weight.requires_grad_(False)


