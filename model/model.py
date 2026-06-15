from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_2tuple(value: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, tuple):
        return value
    return (value, value)


class DropPath(nn.Module):
    """Stochastic depth per sample."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)
        return x.div(keep_prob) * mask


class PatchEmbed(nn.Module):
    """Image to patch-token embedding."""

    def __init__(
        self,
        image_size: int | tuple[int, int] = 96,
        patch_size: int | tuple[int, int] = 8,
        in_channels: int = 3,
        embed_dim: int = 384,
    ):
        super().__init__()
        self.image_size = _to_2tuple(image_size)
        self.patch_size = _to_2tuple(patch_size)
        self.grid_size = (
            self.image_size[0] // self.patch_size[0],
            self.image_size[1] // self.patch_size[1],
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        if x.ndim != 4:
            raise ValueError("input must have shape [batch, channels, height, width]")

        height, width = x.shape[-2:]
        if height % self.patch_size[0] != 0 or width % self.patch_size[1] != 0:
            raise ValueError("image height and width must be divisible by patch_size")

        x = self.proj(x)
        grid_size = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)
        return x, grid_size


class MLP(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.drop_path1 = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), dropout=dropout)
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, need_weights=False)
        x = x + self.drop_path1(attn_out)
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    """A compact ViT backbone that can return tokens or pooled embeddings."""

    def __init__(
        self,
        image_size: int | tuple[int, int] = 96,
        patch_size: int | tuple[int, int] = 8,
        in_channels: int = 3,
        embed_dim: int = 384,
        depth: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        drop_path: float = 0.0,
        use_cls_token: bool = False,
        pool: str = "mean",
    ):
        super().__init__()
        if pool not in {"mean", "cls", "none"}:
            raise ValueError("pool must be one of: 'mean', 'cls', 'none'")
        if pool == "cls" and not use_cls_token:
            raise ValueError("pool='cls' requires use_cls_token=True")

        self.patch_embed = PatchEmbed(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        self.embed_dim = embed_dim
        self.use_cls_token = use_cls_token
        self.pool = pool

        num_extra_tokens = 1 if use_cls_token else 0
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if use_cls_token else None
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches + num_extra_tokens, embed_dim)
        )
        self.pos_drop = nn.Dropout(dropout)

        drop_path_rates = torch.linspace(0, drop_path, depth).tolist()
        self.blocks = nn.ModuleList(
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attention_dropout=attention_dropout,
                drop_path=drop_path_rates[i],
            )
            for i in range(depth)
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def _interpolate_pos_embed(self, grid_size: tuple[int, int]) -> torch.Tensor:
        base_grid = self.patch_embed.grid_size
        if grid_size == base_grid:
            return self.pos_embed

        extra_tokens = self.pos_embed[:, :1] if self.use_cls_token else self.pos_embed[:, :0]
        patch_pos = self.pos_embed[:, extra_tokens.shape[1] :]
        patch_pos = patch_pos.reshape(1, base_grid[0], base_grid[1], self.embed_dim)
        patch_pos = patch_pos.permute(0, 3, 1, 2)
        patch_pos = F.interpolate(
            patch_pos,
            size=grid_size,
            mode="bicubic",
            align_corners=False,
        )
        patch_pos = patch_pos.permute(0, 2, 3, 1).flatten(1, 2)
        return torch.cat((extra_tokens, patch_pos), dim=1)

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        x, grid_size = self.patch_embed(x)

        if self.use_cls_token:
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        x = x + self._interpolate_pos_embed(grid_size)
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        return self.norm(x)

    def forward_masked_tokens(self, x: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """Encode only the patch tokens selected by masks.

        Args:
            x: Images shaped [batch, channels, height, width].
            masks: Patch indices shaped [batch, num_visible_patches].
        """
        if self.use_cls_token:
            raise ValueError("forward_masked_tokens requires use_cls_token=False")
        if masks.ndim != 2:
            raise ValueError("masks must have shape [batch, num_visible_patches]")

        x, grid_size = self.patch_embed(x)
        pos_embed = self._interpolate_pos_embed(grid_size)

        gather_index = masks.to(device=x.device).unsqueeze(-1).expand(-1, -1, x.shape[-1])
        x = x.gather(dim=1, index=gather_index)
        x = x + pos_embed.expand(x.shape[0], -1, -1).gather(dim=1, index=gather_index)
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        return self.norm(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_tokens(x)
        if self.pool == "none":
            return x
        if self.pool == "cls":
            return x[:, 0]
        if self.use_cls_token:
            return x[:, 1:].mean(dim=1)
        return x.mean(dim=1)


class ProjectionHead(nn.Module):
    """Small MLP projection head for self-supervised embeddings."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int | None = None,
    ):
        super().__init__()
        hidden_dim = hidden_dim or in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ViTEncoder(nn.Module):
    """ViT image encoder returning one embedding per image."""

    def __init__(
        self,
        image_size: int | tuple[int, int] = 96,
        patch_size: int | tuple[int, int] = 8,
        in_channels: int = 3,
        embed_dim: int = 384,
        depth: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        drop_path: float = 0.0,
        projection_dim: int | None = 256,
        projection_hidden_dim: int | None = None,
    ):
        super().__init__()
        self.backbone = VisionTransformer(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attention_dropout=attention_dropout,
            drop_path=drop_path,
            use_cls_token=False,
            pool="mean",
        )
        self.projection = (
            ProjectionHead(embed_dim, projection_dim, projection_hidden_dim)
            if projection_dim is not None
            else nn.Identity()
        )
        self.output_dim = projection_dim or embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(self.backbone(x))

    def encode_views(self, views: Iterable[torch.Tensor]) -> torch.Tensor:
        """Encode a sequence of augmented views into [views, batch, output_dim]."""
        return torch.stack([self.forward(view) for view in views])


class IJEPAPredictor(nn.Module):
    """Predict target patch embeddings from context patch embeddings."""

    def __init__(
        self,
        num_patches: int,
        encoder_dim: int = 384,
        predictor_dim: int = 384,
        depth: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.predictor_dim = predictor_dim

        self.context_proj = nn.Linear(encoder_dim, predictor_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, predictor_dim))

        drop_path_rates = torch.linspace(0, drop_path, depth).tolist()
        self.blocks = nn.ModuleList(
            TransformerBlock(
                embed_dim=predictor_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attention_dropout=attention_dropout,
                drop_path=drop_path_rates[i],
            )
            for i in range(depth)
        )
        self.norm = nn.LayerNorm(predictor_dim)
        self.output_proj = nn.Linear(predictor_dim, encoder_dim)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(VisionTransformer._init_weights)

    def _gather_pos(self, masks: torch.Tensor, batch_size: int) -> torch.Tensor:
        masks = masks.to(device=self.pos_embed.device)
        gather_index = masks.unsqueeze(-1).expand(-1, -1, self.predictor_dim)
        pos_embed = self.pos_embed.expand(batch_size, -1, -1)
        return pos_embed.gather(dim=1, index=gather_index)

    def forward(
        self,
        context_tokens: torch.Tensor,
        context_masks: torch.Tensor,
        target_masks: torch.Tensor,
    ) -> torch.Tensor:
        if context_tokens.ndim != 3:
            raise ValueError("context_tokens must have shape [batch, context_len, encoder_dim]")
        if context_masks.ndim != 2 or target_masks.ndim != 2:
            raise ValueError("context_masks and target_masks must have shape [batch, num_patches]")

        batch_size = context_tokens.shape[0]
        context_tokens = self.context_proj(context_tokens)
        context_tokens = context_tokens + self._gather_pos(context_masks, batch_size)

        target_tokens = self.mask_token.expand(batch_size, target_masks.shape[1], -1)
        target_tokens = target_tokens + self._gather_pos(target_masks, batch_size)

        x = torch.cat((context_tokens, target_tokens), dim=1)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.output_proj(x[:, -target_masks.shape[1] :])
