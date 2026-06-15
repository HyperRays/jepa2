import argparse
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from .dataset import STL10BinaryDataset
    from .model import VisionTransformer
    from .run import IJEPATrainConfig, pick_device
except ImportError:
    from dataset import STL10BinaryDataset
    from run import IJEPATrainConfig, pick_device

    from model import VisionTransformer


@dataclass
class LinearProbeConfig:
    checkpoint: str = "checkpoints/ijepa_latest.pt"
    data_root: str = "datasets/stl10/stl10_binary"
    encoder: str = "encoder"
    output_path: str = "checkpoints/linear_probe.pt"
    batch_size: int = 256
    probe_batch_size: int = 256
    num_workers: int = 0
    epochs: int = 50
    max_train_steps: int | None = None
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    log_every: int = 20
    seed: int = 0
    device: str = "auto"


def collate_labeled_batch(
    batch: list[dict[str, torch.Tensor | int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    images = torch.stack([item["image"] for item in batch])
    labels = torch.tensor([int(item["label"]) for item in batch], dtype=torch.long)
    return images, labels


def build_encoder_from_checkpoint(
    checkpoint: dict[str, Any],
    encoder_name: str,
    device: torch.device,
) -> tuple[VisionTransformer, str]:
    valid_encoder_names = {"encoder", "target_encoder", "context_encoder"}
    if encoder_name not in valid_encoder_names:
        raise ValueError(
            "encoder must be 'encoder', 'target_encoder', or 'context_encoder'"
        )
    if encoder_name == "encoder":
        for checkpoint_key in ("encoder", "target_encoder", "context_encoder"):
            if checkpoint_key in checkpoint:
                encoder_name = checkpoint_key
                break
        else:
            raise KeyError("checkpoint does not contain an encoder state dict")
    elif encoder_name not in checkpoint:
        raise KeyError(f"checkpoint does not contain {encoder_name!r}")

    config_fields = {field.name for field in fields(IJEPATrainConfig)}
    train_config = IJEPATrainConfig(
        **{
            key: value
            for key, value in checkpoint["config"].items()
            if key in config_fields
        }
    )
    encoder = VisionTransformer(
        image_size=train_config.image_size,
        patch_size=train_config.patch_size,
        embed_dim=train_config.embed_dim,
        depth=train_config.encoder_depth,
        num_heads=train_config.encoder_heads,
        mlp_ratio=train_config.mlp_ratio,
        use_cls_token=False,
        pool="none",
    )
    encoder.load_state_dict(checkpoint[encoder_name])
    encoder.to(device)
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    return encoder, encoder_name


@torch.no_grad()
def encode_images(
    encoder: VisionTransformer,
    images: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    tokens = encoder.forward_tokens(images.to(device, non_blocking=True))
    return tokens.mean(dim=1)

@torch.no_grad()
def cache_features(
    encoder: VisionTransformer,
    loader: DataLoader,
    device: torch.device,
    name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []

    for batch_index, (images, batch_labels) in enumerate(loader):
        batch_features = encode_images(encoder, images, device)
        features.append(batch_features.cpu())
        labels.append(batch_labels.cpu())
        if batch_index == 0:
            print(f"caching {name}: feature_dim={batch_features.shape[-1]}")

    return torch.cat(features), torch.cat(labels)


def standardize_features(
    train_features: torch.Tensor,
    test_features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    train_features = torch.nan_to_num(train_features, nan=0.0, posinf=0.0, neginf=0.0)
    test_features = torch.nan_to_num(test_features, nan=0.0, posinf=0.0, neginf=0.0)
    mean = train_features.mean(dim=0, keepdim=True)
    std = train_features.std(dim=0, keepdim=True).clamp_min(1e-6)
    train_features = ((train_features - mean) / std).clamp_(-10.0, 10.0)
    test_features = ((test_features - mean) / std).clamp_(-10.0, 10.0)
    return train_features, test_features


def evaluate(
    classifier: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    classifier.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = classifier(features)
            loss = F.cross_entropy(logits, labels)

            total_loss += float(loss.cpu()) * labels.numel()
            total_correct += int((logits.argmax(dim=1) == labels).sum().cpu())
            total_seen += labels.numel()

    return total_loss / total_seen, total_correct / total_seen


def train_linear_probe(config: LinearProbeConfig) -> None:
    torch.manual_seed(config.seed)
    device = pick_device(config.device)
    checkpoint_path = Path(config.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    encoder, encoder_name = build_encoder_from_checkpoint(
        checkpoint, config.encoder, device
    )

    train_dataset = STL10BinaryDataset(
        root=config.data_root,
        split="train",
        return_labels=True,
    )
    test_dataset = STL10BinaryDataset(
        root=config.data_root,
        split="test",
        return_labels=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_labeled_batch,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_labeled_batch,
    )

    print(f"device={device} checkpoint={checkpoint_path} encoder={encoder_name}")
    print(f"caching frozen features with encoder_batch_size={config.batch_size}")
    train_features, train_labels = cache_features(
        encoder, train_loader, device, "train"
    )
    test_features, test_labels = cache_features(encoder, test_loader, device, "test")
    train_features, test_features = standardize_features(train_features, test_features)
    print(
        f"cached train={tuple(train_features.shape)} test={tuple(test_features.shape)} "
        f"probe_batch_size={config.probe_batch_size}"
    )
    probe_device = torch.device("cpu") if device.type == "mps" else device
    print(f"linear_probe_device={probe_device}")

    train_feature_loader = DataLoader(
        TensorDataset(train_features, train_labels),
        batch_size=config.probe_batch_size,
        shuffle=True,
    )
    test_feature_loader = DataLoader(
        TensorDataset(test_features, test_labels),
        batch_size=config.probe_batch_size,
        shuffle=False,
    )

    classifier = nn.Linear(encoder.embed_dim, 10).to(probe_device)
    optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    global_step = 0
    best_accuracy = 0.0

    for epoch in range(config.epochs):
        classifier.train()
        for features, labels in train_feature_loader:
            features = features.to(probe_device, non_blocking=True)
            labels = labels.to(probe_device, non_blocking=True)
            logits = classifier(features)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if global_step % config.log_every == 0:
                accuracy = float((logits.argmax(dim=1) == labels).float().mean().cpu())
                print(
                    f"epoch={epoch + 1} step={global_step} "
                    f"train_loss={float(loss.detach().cpu()):.4f} train_acc={accuracy:.4f}"
                )

            global_step += 1
            if (
                config.max_train_steps is not None
                and global_step >= config.max_train_steps
            ):
                break

        test_loss, test_accuracy = evaluate(
            classifier, test_feature_loader, probe_device
        )
        best_accuracy = max(best_accuracy, test_accuracy)
        print(
            f"epoch={epoch + 1} test_loss={test_loss:.4f} "
            f"test_acc={test_accuracy:.4f} best_acc={best_accuracy:.4f}"
        )

        if config.max_train_steps is not None and global_step >= config.max_train_steps:
            break

    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "checkpoint": str(checkpoint_path),
            "encoder": encoder_name,
            "classifier": classifier.state_dict(),
            "best_accuracy": best_accuracy,
            "config": vars(config),
        },
        output_path,
    )
    print(f"saved {output_path}")


def parse_args() -> LinearProbeConfig:
    parser = argparse.ArgumentParser(
        description="Train a linear probe on frozen I-JEPA STL-10 features."
    )
    parser.add_argument("--checkpoint", default=LinearProbeConfig.checkpoint)
    parser.add_argument("--data-root", default=LinearProbeConfig.data_root)
    parser.add_argument(
        "--encoder",
        choices=["encoder", "target_encoder", "context_encoder"],
        default=LinearProbeConfig.encoder,
    )
    parser.add_argument("--output-path", default=LinearProbeConfig.output_path)
    parser.add_argument("--batch-size", type=int, default=LinearProbeConfig.batch_size)
    parser.add_argument(
        "--probe-batch-size", type=int, default=LinearProbeConfig.probe_batch_size
    )
    parser.add_argument(
        "--num-workers", type=int, default=LinearProbeConfig.num_workers
    )
    parser.add_argument("--epochs", type=int, default=LinearProbeConfig.epochs)
    parser.add_argument(
        "--max-train-steps", type=int, default=LinearProbeConfig.max_train_steps
    )
    parser.add_argument(
        "--learning-rate", type=float, default=LinearProbeConfig.learning_rate
    )
    parser.add_argument(
        "--weight-decay", type=float, default=LinearProbeConfig.weight_decay
    )
    parser.add_argument("--log-every", type=int, default=LinearProbeConfig.log_every)
    parser.add_argument("--seed", type=int, default=LinearProbeConfig.seed)
    parser.add_argument("--device", default=LinearProbeConfig.device)
    return LinearProbeConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    train_linear_probe(parse_args())
