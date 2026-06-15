from collections.abc import Sequence

import torch
import torch.nn as nn


class SIGReg(nn.Module):
    """Sketched Isotropic Gaussian Regularization from LeJEPA."""

    def __init__(self, num_slices: int = 1024, num_knots: int = 17, t_max: float = 5.0):
        super().__init__()
        if num_knots < 2:
            raise ValueError("num_knots must be at least 2")

        self.num_slices = num_slices

        t = torch.linspace(0, t_max, num_knots, dtype=torch.float32)
        dt = t_max / (num_knots - 1)
        weights = torch.full((num_knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        gaussian_cf = torch.exp(-t.square() / 2)

        self.register_buffer("t", t)
        self.register_buffer("gaussian_cf", gaussian_cf)
        self.register_buffer("weights", weights * gaussian_cf)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Measure how far embeddings are from an isotropic Gaussian.

        The last two dimensions must be [batch, embedding_dim]. Any leading
        dimensions, such as a view dimension, are averaged in the result.
        """

        # Trigonometric functions are more stable in float32 than in low precision.
        dtype = torch.float64 if embeddings.dtype == torch.float64 else torch.float32
        embeddings = embeddings.to(dtype=dtype)

        directions = torch.randn(
            embeddings.shape[-1],
            self.num_slices,
            device=embeddings.device,
            dtype=dtype,
        )
        directions = directions / directions.norm(dim=0, keepdim=True).clamp_min(1e-12)

        t = self.t.to(dtype=dtype)
        gaussian_cf = self.gaussian_cf.to(dtype=dtype)
        weights = self.weights.to(dtype=dtype)

        projected = embeddings @ directions
        x_t = projected.unsqueeze(-1) * t
        error = (
            (x_t.cos().mean(dim=-3) - gaussian_cf).square()
            + x_t.sin().mean(dim=-3).square()
        )
        statistic = (error @ weights) * embeddings.shape[-2]
        return statistic.mean()


class LeJEPALoss(nn.Module):
    """LeJEPA invariance loss combined with SIGReg."""

    def __init__(
        self,
        lambd: float = 0.05,
        num_global_views: int = 2,
        num_slices: int = 1024,
        num_knots: int = 17,
        t_max: float = 5.0,
    ):
        super().__init__()
        if not 0 <= lambd <= 1:
            raise ValueError("lambd must be between 0 and 1")
        if num_global_views < 1:
            raise ValueError("num_global_views must be at least 1")

        self.lambd = lambd
        self.num_global_views = num_global_views
        self.sigreg = SIGReg(
            num_slices=num_slices,
            num_knots=num_knots,
            t_max=t_max,
        )

    def forward(
        self,
        embeddings: torch.Tensor | Sequence[torch.Tensor],
        *,
        return_components: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Compute the LeJEPA loss for multiple augmented views.

        Args:
            embeddings: A tensor shaped [views, batch, embedding_dim], or a
                sequence of tensors shaped [batch, embedding_dim]. The first
                num_global_views entries are treated as global views.
            return_components: Return the invariance and SIGReg terms for logging.
        """
        if not isinstance(embeddings, torch.Tensor):
            embeddings = torch.stack(tuple(embeddings))
        if embeddings.ndim != 3:
            raise ValueError("embeddings must have shape [views, batch, embedding_dim]")
        if embeddings.shape[0] < 2:
            raise ValueError("LeJEPA requires at least two views")
        if self.num_global_views > embeddings.shape[0]:
            raise ValueError("num_global_views cannot exceed the number of views")

        center = embeddings[: self.num_global_views].mean(dim=0, keepdim=True)
        invariance = (center - embeddings).square().mean()
        sigreg = self.sigreg(embeddings)
        loss = (1 - self.lambd) * invariance + self.lambd * sigreg

        if return_components:
            return loss, {"invariance": invariance, "sigreg": sigreg}
        return loss
