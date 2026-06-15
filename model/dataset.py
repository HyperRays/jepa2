from bisect import bisect_right
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
import tarfile
import urllib.request

import torch
from torch.utils.data import DataLoader, Dataset


STL10_IMAGE_SIZE = 96
STL10_CHANNELS = 3
STL10_IMAGE_BYTES = STL10_CHANNELS * STL10_IMAGE_SIZE * STL10_IMAGE_SIZE
STL10_URL = "http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz"
STL10_REQUIRED_FILES = (
    "train_X.bin",
    "train_y.bin",
    "test_X.bin",
    "test_y.bin",
    "unlabeled_X.bin",
)


def stl10_is_present(root: str | Path = "datasets/stl10/stl10_binary") -> bool:
    root = Path(root)
    return all((root / name).exists() for name in STL10_REQUIRED_FILES)


def _safe_extract_tar(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            member_path = (destination / member.name).resolve()
            if destination != member_path and destination not in member_path.parents:
                raise RuntimeError(f"refusing to extract path outside destination: {member.name}")
        tar.extractall(destination)


def download_stl10_if_needed(
    root: str | Path = "datasets/stl10/stl10_binary",
    *,
    url: str = STL10_URL,
) -> Path:
    """Download and extract STL-10 binary files if root is incomplete."""
    root = Path(root)
    if stl10_is_present(root):
        return root

    data_dir = root.parent
    archive = data_dir / "stl10_binary.tar.gz"
    data_dir.mkdir(parents=True, exist_ok=True)

    if not archive.exists():
        print(f"downloading STL-10 binary dataset to {archive}")
        urllib.request.urlretrieve(url, archive)
    else:
        print(f"using cached STL-10 archive {archive}")

    print(f"extracting STL-10 into {data_dir}")
    _safe_extract_tar(archive, data_dir)

    missing = [name for name in STL10_REQUIRED_FILES if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(f"STL-10 extraction did not produce expected files: {missing}")
    return root


@dataclass(frozen=True)
class _STL10Source:
    data: torch.Tensor
    labels: torch.Tensor | None
    length: int


class STL10BinaryDataset(Dataset):
    """Memory-mapped STL-10 binary dataset from ./datasets/stl10/stl10_binary."""

    valid_splits = {"train", "test", "unlabeled", "train+unlabeled"}

    def __init__(
        self,
        root: str | Path = "datasets/stl10/stl10_binary",
        split: str = "unlabeled",
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        return_labels: bool = True,
        download: bool = True,
    ):
        super().__init__()
        if split not in self.valid_splits:
            raise ValueError(f"split must be one of {sorted(self.valid_splits)}")

        self.root = Path(root)
        if download:
            self.root = download_stl10_if_needed(self.root)
        self.split = split
        self.transform = transform
        self.return_labels = return_labels
        self.sources = self._build_sources(split)

        total = 0
        self.cumulative_lengths: list[int] = []
        for source in self.sources:
            total += source.length
            self.cumulative_lengths.append(total)

    def _build_sources(self, split: str) -> list[_STL10Source]:
        if split == "train":
            return [self._load_source("train_X.bin", "train_y.bin")]
        if split == "test":
            return [self._load_source("test_X.bin", "test_y.bin")]
        if split == "unlabeled":
            return [self._load_source("unlabeled_X.bin")]
        return [
            self._load_source("train_X.bin", "train_y.bin"),
            self._load_source("unlabeled_X.bin"),
        ]

    def _load_source(self, image_file: str, label_file: str | None = None) -> _STL10Source:
        image_path = self.root / image_file
        if not image_path.exists():
            raise FileNotFoundError(image_path)

        length = image_path.stat().st_size // STL10_IMAGE_BYTES
        data = torch.from_file(
            str(image_path),
            dtype=torch.uint8,
            size=length * STL10_IMAGE_BYTES,
        ).reshape(length, STL10_CHANNELS, STL10_IMAGE_SIZE, STL10_IMAGE_SIZE)

        labels = None
        if label_file is not None:
            label_path = self.root / label_file
            if not label_path.exists():
                raise FileNotFoundError(label_path)
            labels = torch.from_file(str(label_path), dtype=torch.uint8, size=length)

        return _STL10Source(data=data, labels=labels, length=length)

    def __len__(self) -> int:
        return self.cumulative_lengths[-1]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)

        source_index = bisect_right(self.cumulative_lengths, index)
        previous_length = self.cumulative_lengths[source_index - 1] if source_index > 0 else 0
        local_index = index - previous_length
        source = self.sources[source_index]

        # STL-10 stores each channel transposed relative to ordinary CHW tensors.
        image = source.data[local_index].transpose(1, 2).to(torch.float32).div_(255.0)
        if self.transform is not None:
            image = self.transform(image)

        if not self.return_labels:
            return {"image": image}

        label = -1 if source.labels is None else int(source.labels[local_index].item()) - 1
        return {"image": image, "label": label}


class IJEPAMaskCollator:
    """Batch collator that adds I-JEPA context and target patch-index masks.

    The collator follows the I-JEPA paper's masking setup: sample four
    possibly-overlapping target blocks, sample one large context block, remove
    target-overlapping patches from the context, and return patch indices for
    the context and target views. It does not mask pixels in the image.
    """

    def __init__(
        self,
        image_size: int = STL10_IMAGE_SIZE,
        patch_size: int = 8,
        num_targets: int = 4,
        target_scale: tuple[float, float] = (0.15, 0.2),
        target_aspect_ratio: tuple[float, float] = (0.75, 1.5),
        context_scale: tuple[float, float] = (0.85, 1.0),
        context_aspect_ratio: tuple[float, float] = (1.0, 1.0),
        min_context_patches: int = 1,
        seed: int | None = None,
    ):
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        if num_targets < 1:
            raise ValueError("num_targets must be at least 1")
        if min_context_patches < 1:
            raise ValueError("min_context_patches must be at least 1")

        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.num_targets = num_targets
        self.target_scale = target_scale
        self.target_aspect_ratio = target_aspect_ratio
        self.context_scale = context_scale
        self.context_aspect_ratio = context_aspect_ratio
        self.min_context_patches = min_context_patches
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    def __call__(
        self, batch: Sequence[dict[str, torch.Tensor | int] | tuple[torch.Tensor, int]]
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        images, labels = self._stack_batch(batch)
        target_shapes = [
            self._sample_block_shape(self.target_scale, self.target_aspect_ratio)
            for _ in range(self.num_targets)
        ]
        context_shape = self._sample_block_shape(
            self.context_scale,
            self.context_aspect_ratio,
        )

        context_masks: list[torch.Tensor] = []
        target_masks_by_block: list[list[torch.Tensor]] = [[] for _ in range(self.num_targets)]

        for _ in range(images.shape[0]):
            context_mask, target_masks = self._sample_image_masks(target_shapes, context_shape)
            context_masks.append(context_mask)
            for block_index, target_mask in enumerate(target_masks):
                target_masks_by_block[block_index].append(target_mask)

        context_masks = self._stack_context_masks(context_masks)
        target_masks = [torch.stack(block_masks) for block_masks in target_masks_by_block]

        output = {
            "images": images,
            "context_masks": context_masks,
            "target_masks": target_masks,
        }
        if labels is not None:
            output["labels"] = labels
        return output

    def _stack_batch(
        self, batch: Sequence[dict[str, torch.Tensor | int] | tuple[torch.Tensor, int]]
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        images: list[torch.Tensor] = []
        labels: list[int] = []
        has_labels = True

        for item in batch:
            if isinstance(item, dict):
                images.append(item["image"])
                if "label" in item:
                    labels.append(int(item["label"]))
                else:
                    has_labels = False
            else:
                image, label = item
                images.append(image)
                labels.append(int(label))

        images_tensor = torch.stack(images)
        labels_tensor = torch.tensor(labels, dtype=torch.long) if has_labels else None
        return images_tensor, labels_tensor

    def _sample_image_masks(
        self,
        target_shapes: Sequence[tuple[int, int]],
        context_shape: tuple[int, int],
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        for _ in range(100):
            target_masks = [self._sample_block(*shape) for shape in target_shapes]
            target_union = torch.cat(target_masks).unique()
            context_mask = self._sample_block(*context_shape)
            keep = ~torch.isin(context_mask, target_union)
            context_mask = context_mask[keep]
            if context_mask.numel() >= self.min_context_patches:
                return context_mask, target_masks

        raise RuntimeError("failed to sample a non-empty I-JEPA context mask")

    def _stack_context_masks(self, context_masks: list[torch.Tensor]) -> torch.Tensor:
        mask_length = min(mask.numel() for mask in context_masks)
        if mask_length < self.min_context_patches:
            raise RuntimeError("sampled context masks are smaller than min_context_patches")

        stacked = []
        for mask in context_masks:
            if mask.numel() > mask_length:
                selection = torch.randperm(mask.numel(), generator=self.generator)[:mask_length]
                mask = mask[selection].sort().values
            stacked.append(mask)
        return torch.stack(stacked)

    def _sample_block_shape(
        self,
        scale: tuple[float, float],
        aspect_ratio: tuple[float, float],
    ) -> tuple[int, int]:
        min_scale, max_scale = scale
        min_aspect, max_aspect = aspect_ratio
        if min_scale <= 0 or max_scale <= 0 or min_scale > max_scale:
            raise ValueError("scale must be a positive (min, max) tuple")
        if min_aspect <= 0 or max_aspect <= 0 or min_aspect > max_aspect:
            raise ValueError("aspect_ratio must be a positive (min, max) tuple")

        area = self.grid_size * self.grid_size
        for _ in range(100):
            block_area = area * self._uniform(min_scale, max_scale)
            aspect = self._uniform(min_aspect, max_aspect)
            height = max(1, round((block_area / aspect) ** 0.5))
            width = max(1, round((block_area * aspect) ** 0.5))
            if height <= self.grid_size and width <= self.grid_size:
                return height, width

        height = min(self.grid_size, max(1, round((area * min_scale) ** 0.5)))
        return height, height

    def _sample_block(self, height: int, width: int) -> torch.Tensor:
        max_top = self.grid_size - height
        max_left = self.grid_size - width
        top = int(torch.randint(max_top + 1, (), generator=self.generator).item())
        left = int(torch.randint(max_left + 1, (), generator=self.generator).item())

        rows = torch.arange(top, top + height)
        cols = torch.arange(left, left + width)
        return (rows[:, None] * self.grid_size + cols[None, :]).flatten()

    def _uniform(self, low: float, high: float) -> float:
        value = torch.rand((), generator=self.generator).item()
        return low + (high - low) * value


def make_ijepa_stl10_loader(
    root: str | Path = "datasets/stl10/stl10_binary",
    split: str = "unlabeled",
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
    patch_size: int = 8,
    return_labels: bool = False,
    download: bool = True,
    **collator_kwargs,
) -> DataLoader:
    dataset = STL10BinaryDataset(
        root=root,
        split=split,
        return_labels=return_labels,
        download=download,
    )
    collator = IJEPAMaskCollator(patch_size=patch_size, **collator_kwargs)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
    )
