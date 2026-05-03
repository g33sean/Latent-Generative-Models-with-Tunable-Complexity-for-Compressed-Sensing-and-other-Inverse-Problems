import torch
import torch.nn as nn

class NestedDropout(nn.Module):
    def __init__(self, latent_dim, p, device):
        """
        Implements Nested Dropout using a Geometric Distribution.
        Args:
            latent_dim (int): Total number of latent dimensions (e.g., 4*8*8=256).
            p (float): Probability parameter for the geometric distribution.
            device (torch.device or str): device for tensors.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.p = float(p)            # ensure Python float
        self.device = torch.device(device)
        # create a tensor once to pass as `probs`
        self.register_buffer('_probs', torch.tensor(self.p, device=self.device))

    def forward(self, latents):
        # latents: (B, C, H, W) or similar
        batch_size = latents.size(0)
        # flatten to (B, latent_dim)
        flat = latents.reshape(batch_size, -1)
        D = flat.size(1)

        # sample k ~ Geometric(probs) for each batch element
        # now using the tensor _probs for broadcast safety
        geom = torch.distributions.Geometric(self._probs)
        k = geom.sample((batch_size,)).clamp(1, D).long()   # (B,)

        # build mask: [i < k[b]] -> keep first k dims
        arange = torch.arange(D, device=self.device).unsqueeze(0)  # (1, D)
        mask = (arange < k.unsqueeze(1)).float()                  # (B, D)

        dropped = flat * mask
        return dropped.view_as(latents).contiguous()
