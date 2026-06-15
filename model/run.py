import argparse
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

try:
    from .dataset import make_ijepa_stl10_loader
    from .lejepa import LeJEPALoss
    from .model import VisionTransformer
except ImportError:
    from dataset import make_ijepa_stl10_loader
    from lejepa import LeJEPALoss
    from model import VisionTransformer


@dataclass
class IJEPATrainConfig:
    data_root: str = "datasets/stl10/stl10_binary"
    split: str = "unlabeled"
    output_dir: str = "checkpoints"
    image_size: int = 96
    patch_size: int = 8
    batch_size: int = 64
    num_workers: int = 0
    epochs: int = 1
    max_steps: int | None = None
    learning_rate: float = 1e-4
    min_learning_rate: float = 1e-6
    weight_decay: float = 0.04
    warmup_steps: int = 100
    embed_dim: int = 192
    encoder_depth: int = 6
    encoder_heads: int = 6
    mlp_ratio: float = 4.0
    log_every: int = 20
    save_every: int = 0
    seed: int = 0
    device: str = "auto"
    compile: bool = False
    amp: bool = False
    amp_dtype: str = "float16"
    profile: bool = False
    profile_steps: int = 5


def pick_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cosine_schedule(
    step: int,
    total_steps: int,
    base_value: float,
    final_value: float,
    warmup_steps: int = 0,
) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return base_value * float(step + 1) / float(warmup_steps)

    if total_steps <= warmup_steps:
        return final_value

    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return final_value + 0.5 * (base_value - final_value) * (1.0 + math.cos(math.pi * progress))


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def gather_tokens(tokens: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    masks = masks.to(device=tokens.device)
    gather_index = masks.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
    return tokens.gather(dim=1, index=gather_index)


def pool_block_tokens(tokens: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    return gather_tokens(tokens, masks).mean(dim=1)


def build_encoder(config: IJEPATrainConfig, device: torch.device) -> VisionTransformer:
    encoder = VisionTransformer(
        image_size=config.image_size,
        patch_size=config.patch_size,
        embed_dim=config.embed_dim,
        depth=config.encoder_depth,
        num_heads=config.encoder_heads,
        mlp_ratio=config.mlp_ratio,
        use_cls_token=False,
        pool="none",
    )
    encoder.to(device)
    return encoder


def resolve_compile(config: IJEPATrainConfig, device: torch.device) -> bool:
    """Decide whether torch.compile is worth it, and say so when it is not.

    For this workload (small ViT, mask lengths that vary every batch) the
    inductor backend only pays off on CUDA. On MPS/CPU it adds a ~10s+ one-time
    compile and measures *slower* steady-state steps than eager, so we skip it.
    """
    if not config.compile:
        return False
    if device.type != "cuda":
        print(
            f"--compile ignored on device={device.type}: torch.compile adds a slow "
            "one-time compile and runs slower than eager for this model outside CUDA; "
            "training eagerly."
        )
        return False
    print("compiling encoder with dynamic shapes")
    return True


def maybe_compile(module: nn.Module, enabled: bool) -> nn.Module:
    if not enabled:
        return module
    return torch.compile(module, dynamic=True)


def unwrap_compiled(module: nn.Module) -> nn.Module:
    return getattr(module, "_orig_mod", module)


def resolve_amp(config: IJEPATrainConfig, device: torch.device) -> tuple[bool, torch.dtype]:
    if config.amp_dtype == "float16":
        amp_dtype = torch.float16
    elif config.amp_dtype == "bfloat16":
        amp_dtype = torch.bfloat16
    else:
        raise ValueError("amp_dtype must be 'float16' or 'bfloat16'")

    if not config.amp:
        return False, amp_dtype
    if device.type != "cuda":
        print(f"--amp ignored on device={device.type}: CUDA AMP is only enabled on CUDA.")
        return False, amp_dtype
    return True, amp_dtype


def build_grad_scaler(
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> torch.amp.GradScaler:
    return torch.amp.GradScaler(
        device="cuda" if device.type == "cuda" else "cpu",
        enabled=amp_enabled and device.type == "cuda" and amp_dtype == torch.float16,
    )


def train_one_step(
    batch: dict[str, torch.Tensor | list[torch.Tensor]],
    encoder: VisionTransformer,
    criterion: LeJEPALoss,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    images = batch["images"].to(device, non_blocking=True)
    context_masks = batch["context_masks"].to(device, non_blocking=True)
    target_masks = [mask.to(device, non_blocking=True) for mask in batch["target_masks"]]

    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
        tokens = encoder.forward_tokens(images)
        context_embedding = pool_block_tokens(tokens, context_masks)
        target_embeddings = [pool_block_tokens(tokens, mask) for mask in target_masks]
        embeddings = torch.stack([context_embedding, *target_embeddings])
        loss, components = criterion(embeddings, return_components=True)

    if scaler.is_enabled():
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()

    # Keep the loss on-device; syncing to host here would stall the pipeline every
    # step. The caller materializes it with .item() only when it actually logs.
    detached_components = {name: value.detach() for name, value in components.items()}
    return loss.detach(), detached_components


def profile_training(config: IJEPATrainConfig) -> None:
    torch.manual_seed(config.seed)
    device = pick_device(config.device)
    loader = make_ijepa_stl10_loader(
        root=config.data_root,
        split=config.split,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        patch_size=config.patch_size,
        return_labels=False,
        seed=config.seed,
    )

    amp_enabled, amp_dtype = resolve_amp(config, device)
    print(
        f"profile device={device} compile={config.compile} "
        f"amp={amp_enabled} amp_dtype={config.amp_dtype}"
    )
    print(f"profile_steps={config.profile_steps} batch_size={config.batch_size}")

    iterator = iter(loader)
    data_times = []
    batches = []
    for _ in range(config.profile_steps):
        start = time.perf_counter()
        batch = next(iterator)
        data_times.append(time.perf_counter() - start)
        batches.append(batch)

    print(
        "data_loader_seconds "
        f"avg={sum(data_times) / len(data_times):.4f} "
        f"min={min(data_times):.4f} max={max(data_times):.4f}"
    )
    for index, batch in enumerate(batches):
        print(
            f"mask_shapes step={index} "
            f"context={tuple(batch['context_masks'].shape)} "
            f"targets={[tuple(mask.shape) for mask in batch['target_masks']]}"
        )

    encoder = build_encoder(config, device)
    criterion = LeJEPALoss(num_global_views=1).to(device)
    do_compile = resolve_compile(config, device)
    encoder = maybe_compile(encoder, do_compile)

    optimizer = torch.optim.AdamW(
        encoder.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scaler = build_grad_scaler(device, amp_enabled, amp_dtype)

    encoder.train()

    for step, batch in enumerate(batches):
        sync_device(device)
        start = time.perf_counter()

        images = batch["images"].to(device, non_blocking=True)
        context_masks = batch["context_masks"].to(device, non_blocking=True)
        target_masks = [mask.to(device, non_blocking=True) for mask in batch["target_masks"]]
        sync_device(device)
        after_transfer = time.perf_counter()

        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            tokens = encoder.forward_tokens(images)
        sync_device(device)
        after_encoder = time.perf_counter()

        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            context_embedding = pool_block_tokens(tokens, context_masks)
            target_embeddings = [pool_block_tokens(tokens, mask) for mask in target_masks]
            embeddings = torch.stack([context_embedding, *target_embeddings])
        sync_device(device)
        after_pool = time.perf_counter()

        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            loss, components = criterion(embeddings, return_components=True)
        sync_device(device)
        after_loss = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        sync_device(device)
        after_backward = time.perf_counter()

        print(
            f"profile_step={step} loss={float(loss.detach().cpu()):.6f} "
            f"invariance={float(components['invariance'].detach().cpu()):.6f} "
            f"sigreg={float(components['sigreg'].detach().cpu()):.6f} "
            f"amp_scale={scaler.get_scale():.1f} "
            f"transfer={after_transfer - start:.4f} "
            f"encoder={after_encoder - after_transfer:.4f} "
            f"pool={after_pool - after_encoder:.4f} "
            f"loss_calc={after_loss - after_pool:.4f} "
            f"backward={after_backward - after_loss:.4f} "
            f"total={after_backward - start:.4f}"
        )

    if do_compile:
        try:
            from torch._dynamo.utils import compile_times

            print("torch_compile_times")
            print(compile_times())
        except Exception as error:
            print(f"could not read torch compile times: {error}")


def save_checkpoint(
    path: Path,
    config: IJEPATrainConfig,
    encoder: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    epoch: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": asdict(config),
            "encoder": unwrap_compiled(encoder).state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "epoch": epoch,
        },
        path,
    )


def train(config: IJEPATrainConfig) -> None:
    if config.profile:
        profile_training(config)
        return

    torch.manual_seed(config.seed)
    device = pick_device(config.device)

    loader = make_ijepa_stl10_loader(
        root=config.data_root,
        split=config.split,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        patch_size=config.patch_size,
        return_labels=False,
        seed=config.seed,
    )
    total_steps = config.epochs * len(loader)
    if config.max_steps is not None:
        total_steps = min(total_steps, config.max_steps)

    encoder = build_encoder(config, device)
    criterion = LeJEPALoss(num_global_views=1).to(device)
    do_compile = resolve_compile(config, device)
    amp_enabled, amp_dtype = resolve_amp(config, device)
    encoder = maybe_compile(encoder, do_compile)

    optimizer = torch.optim.AdamW(
        encoder.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scaler = build_grad_scaler(device, amp_enabled, amp_dtype)

    output_dir = Path(config.output_dir)
    print(f"device={device} split={config.split} batches_per_epoch={len(loader)}")
    print(
        f"training_steps={total_steps} batch_size={config.batch_size} "
        f"amp={amp_enabled} amp_dtype={config.amp_dtype}"
    )

    global_step = 0
    encoder.train()

    for epoch in range(config.epochs):
        for batch in loader:
            learning_rate = cosine_schedule(
                step=global_step,
                total_steps=total_steps,
                base_value=config.learning_rate,
                final_value=config.min_learning_rate,
                warmup_steps=config.warmup_steps,
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = learning_rate

            loss, components = train_one_step(
                batch=batch,
                encoder=encoder,
                criterion=criterion,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
            )

            if global_step % config.log_every == 0:
                print(
                    f"epoch={epoch + 1} step={global_step} "
                    f"loss={loss.item():.6f} "
                    f"invariance={components['invariance'].item():.6f} "
                    f"sigreg={components['sigreg'].item():.6f} "
                    f"lr={learning_rate:.2e}"
                )

            if config.save_every > 0 and global_step > 0 and global_step % config.save_every == 0:
                save_checkpoint(
                    output_dir / f"ijepa_step_{global_step}.pt",
                    config,
                    encoder,
                    optimizer,
                    global_step,
                    epoch,
                )

            global_step += 1
            if config.max_steps is not None and global_step >= config.max_steps:
                break

        if config.max_steps is not None and global_step >= config.max_steps:
            break

    save_checkpoint(
        output_dir / "ijepa_latest.pt",
        config,
        encoder,
        optimizer,
        global_step,
        config.epochs,
    )
    print(f"saved {output_dir / 'ijepa_latest.pt'}")


def parse_args() -> IJEPATrainConfig:
    parser = argparse.ArgumentParser(description="Train LeJEPA over I-JEPA block masks on local STL-10.")
    parser.add_argument("--data-root", default=IJEPATrainConfig.data_root)
    parser.add_argument("--split", default=IJEPATrainConfig.split)
    parser.add_argument("--output-dir", default=IJEPATrainConfig.output_dir)
    parser.add_argument("--image-size", type=int, default=IJEPATrainConfig.image_size)
    parser.add_argument("--patch-size", type=int, default=IJEPATrainConfig.patch_size)
    parser.add_argument("--batch-size", type=int, default=IJEPATrainConfig.batch_size)
    parser.add_argument("--num-workers", type=int, default=IJEPATrainConfig.num_workers)
    parser.add_argument("--epochs", type=int, default=IJEPATrainConfig.epochs)
    parser.add_argument("--max-steps", type=int, default=IJEPATrainConfig.max_steps)
    parser.add_argument("--learning-rate", type=float, default=IJEPATrainConfig.learning_rate)
    parser.add_argument("--min-learning-rate", type=float, default=IJEPATrainConfig.min_learning_rate)
    parser.add_argument("--weight-decay", type=float, default=IJEPATrainConfig.weight_decay)
    parser.add_argument("--warmup-steps", type=int, default=IJEPATrainConfig.warmup_steps)
    parser.add_argument("--embed-dim", type=int, default=IJEPATrainConfig.embed_dim)
    parser.add_argument("--encoder-depth", type=int, default=IJEPATrainConfig.encoder_depth)
    parser.add_argument("--encoder-heads", type=int, default=IJEPATrainConfig.encoder_heads)
    parser.add_argument("--mlp-ratio", type=float, default=IJEPATrainConfig.mlp_ratio)
    parser.add_argument("--log-every", type=int, default=IJEPATrainConfig.log_every)
    parser.add_argument("--save-every", type=int, default=IJEPATrainConfig.save_every)
    parser.add_argument("--seed", type=int, default=IJEPATrainConfig.seed)
    parser.add_argument("--device", default=IJEPATrainConfig.device)
    parser.add_argument(
        "--compile",
        action="store_true",
        default=IJEPATrainConfig.compile,
        help="Enable torch.compile (used only on CUDA; ignored on MPS/CPU where it is slower).",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        default=IJEPATrainConfig.amp,
        help="Enable CUDA automatic mixed precision for forward/loss and optimizer stepping.",
    )
    parser.add_argument(
        "--amp-dtype",
        choices=["float16", "bfloat16"],
        default=IJEPATrainConfig.amp_dtype,
        help="Autocast dtype to use when --amp is enabled on CUDA.",
    )
    parser.add_argument("--profile", action="store_true", default=IJEPATrainConfig.profile)
    parser.add_argument("--profile-steps", type=int, default=IJEPATrainConfig.profile_steps)
    return IJEPATrainConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    train(parse_args())
