import os, pickle, mmap
import json, importlib.resources as ir
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Dict

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

ENABLE_FILE_CACHE = False

def load_torchvision_imagenet_wnid_to_idx() -> Dict[str, int]:
    """
    Builds wnid->idx from torchvision's imagenet_class_index.json (0..999).
    """
    p = ir.files("py_src.ml_setup_base").joinpath("imagenet_class_index.json")
    with p.open("r") as f:
        class_index = json.load(f)
    return {wnid: int(k) for k, (wnid, _) in class_index.items()}

class MaskedImageDataset(Dataset):
    """
    Pairs images from `image_root` with PNG masks from `mask_root`.
    For each class subfolder, only files that exist in BOTH sides (same stem) are used.
    In __getitem__, black mask pixels (==0) in the mask replace the image pixels with random noise.

    Returns: (image_tensor, label_index) by default.
    Set return_paths=True to also get (img_path, mask_path).
    """
    def __init__(
        self,
        image_root: str,
        mask_root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        *,
        image_exts: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".jpeg", ".JPEG", ".JPG", ".PNG"),
        mask_exts: Tuple[str, ...] = (".png",),
        return_paths: bool = False,
        unmasked_area_type: str = "random",
        use_imagenet_label: bool = False,
    ):
        self.image_root = Path(image_root)
        self.mask_root = Path(mask_root)
        self.transform = transform
        self.target_transform = target_transform
        self.image_exts = tuple(set(e.lower() for e in image_exts))
        self.mask_exts = tuple(set(e.lower() for e in mask_exts))
        self.return_paths = return_paths
        self.unmasked_area_type = unmasked_area_type

        if not self.image_root.is_dir():
            raise FileNotFoundError(f"image_root not found: {self.image_root}")
        if not self.mask_root.is_dir():
            raise FileNotFoundError(f"mask_root not found: {self.mask_root}")

        # Classes: intersection of subfolders that exist in both roots
        image_classes = sorted([p.name for p in self.image_root.iterdir() if p.is_dir()])
        mask_classes = sorted([p.name for p in self.mask_root.iterdir() if p.is_dir()])
        shared_classes = sorted(set(image_classes) & set(mask_classes))
        if not shared_classes:
            raise RuntimeError("No shared class subfolders between image_root and mask_root.")

        self.classes: List[str] = shared_classes
        if use_imagenet_label:
            self.class_to_idx = load_torchvision_imagenet_wnid_to_idx()
        else:
            self.class_to_idx: Dict[str, int] = {c: i for i, c in enumerate(self.classes)}

        # Build index: only common stems per class (uses the smaller side implicitly)
        self.samples: List[Tuple[Path, Path, int]] = []

        pickle_cache_path = f"{self.mask_root}/mask_list.pickle"
        if ENABLE_FILE_CACHE and os.path.exists(pickle_cache_path):
            print("find mask_list.pickle file in mask folder.")
            with open(pickle_cache_path, "rb") as f:
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                self.samples = pickle.load(mm)
                mm.close()
        else:
            if ENABLE_FILE_CACHE:
                print("generating mask_list and save to pickle.")
            for cls in self.classes:
                img_dir = self.image_root / cls
                msk_dir = self.mask_root / cls
                if not img_dir.is_dir() or not msk_dir.is_dir():
                    continue

                # Map stem -> path
                img_map: Dict[str, Path] = {}
                for p in img_dir.iterdir():
                    if p.is_file() and p.suffix.lower() in self.image_exts:
                        img_map[p.stem] = p
                msk_map: Dict[str, Path] = {}
                for p in msk_dir.iterdir():
                    if p.is_file() and p.suffix.lower() in self.mask_exts:
                        msk_map[p.stem] = p

                common_stems = sorted(set(img_map.keys()) & set(msk_map.keys()))
                # Only pairs that exist on both sides (this naturally equals the smaller count)
                for stem in common_stems:
                    self.samples.append((img_map[stem], msk_map[stem], self.class_to_idx[cls]))
            if not self.samples:
                raise RuntimeError("Found no matching (image, mask) pairs.")
            if ENABLE_FILE_CACHE:
                with open(pickle_cache_path, "wb") as f:
                    pickle.dump(self.samples, f, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self) -> int:
        return len(self.samples)

    def _apply_mask_with_noise(self, img: Image.Image, mask: Image.Image) -> Image.Image:
        """Replace pixels where mask==0 (black) with random noise."""
        # Ensure sizes match (keep NN to respect hard mask edges)
        if mask.size != img.size:
            mask = mask.resize(img.size, resample=Image.Resampling.NEAREST)

        img_arr = np.array(img.convert("RGB"), dtype=np.uint8)
        mask_arr = np.array(mask.convert("L"), dtype=np.uint8)

        # Masked area definition: exactly black (0)
        masked = (mask_arr == 0)  # H x W boolean
        if masked.any():
            if self.unmasked_area_type == "random":
                unmasked = np.random.randint(0, 256, size=img_arr.shape, dtype=np.uint8)
            elif self.unmasked_area_type == "zero":
                unmasked = np.full_like(img_arr, fill_value=0, dtype=np.uint8)
            else:
                raise NotImplementedError
            img_arr = np.where(masked[..., None], unmasked, img_arr)

        return Image.fromarray(img_arr, mode="RGB")

    def __getitem__(self, index: int):
        img_path, msk_path, label = self.samples[index]

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(msk_path)
        img = self._apply_mask_with_noise(img, mask)

        if self.transform is not None:
            img = self.transform(img)
        else:
            # Default: convert to float tensor in [0,1]
            img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0

        if self.target_transform is not None:
            label = self.target_transform(label)

        if self.return_paths:
            return img, label, str(img_path), str(msk_path)
        return img, label


# --- Minimal usage example ---
# from torch.utils.data import DataLoader
# ds = SamMaskedImageDataset(
#     image_root="train",
#     mask_root="train_sam_mask",
#     # e.g., add your own transforms here (Resize/ToTensor/Normalize etc.)
# )
# loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
# for images, labels in loader:
#     ...


