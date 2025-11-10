from typing import Tuple, Optional, Callable, Dict
from pathlib import Path

import random
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF

from utils import find_file_by_prefix


def ensure_min_size_and_pad(img: Image.Image, min_h: int, min_w: int, fill: int = 0) -> Image.Image:
    """
    Ensure the image is at least min_h x min_w by padding symmetrically if needed.

    :param img: PIL image.
    :param min_h: Minimum height required.
    :param min_w: Minimum width required.
    :param fill: Fill value for padding.
    :return: Padded image with at least (min_h, min_w).
    """
    w, h = img.size

    pad_left = pad_top = pad_right = pad_bottom = 0

    if w < min_w:
        diff = min_w - w
        pad_left = diff // 2
        pad_right = diff - pad_left

    if h < min_h:
        diff = min_h - h
        pad_top = diff // 2
        pad_bottom = diff - pad_top

    if any([pad_left, pad_top, pad_right, pad_bottom]):
        img = TF.pad(img, padding=(pad_left, pad_top, pad_right, pad_bottom), fill=fill)

    return img


class SegmentationTransformNoResize:
    """
    Transforms that apply identical geometric ops to three images + mask + region.
    Uses padding if inputs are smaller than crop and performs crop (random for train, center for val).
    """

    def __init__(
        self,
        crop: Tuple[int, int] = (512, 512),
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.0,
        rotation_deg: float = 10.0,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        is_train: bool = True,
    ) -> None:
        self.crop = crop
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.rotation_deg = rotation_deg
        self.mean = mean
        self.std = std
        self.is_train = is_train

    def __call__(self, img1: Image.Image, img2: Image.Image, img3: Image.Image, mask: Image.Image, region_mask: Image.Image) -> Dict[str, torch.Tensor]:
        """
        Apply transforms and return tensors for model input.

        :param img1: PIL image for channel 0.
        :param img2: PIL image for channel 1.
        :param img3: PIL image for channel 2.
        :param mask: PIL mask (nutrient_mask_g0).
        :param region_mask: PIL region mask (boundary_mask).
        :return: Dict with keys 'img1','img2','img3','mask','region_mask' tensors.
        """
        crop_h, crop_w = self.crop

        img1 = ensure_min_size_and_pad(img1, crop_h, crop_w)
        img2 = ensure_min_size_and_pad(img2, crop_h, crop_w)
        img3 = ensure_min_size_and_pad(img3, crop_h, crop_w)
        mask = ensure_min_size_and_pad(mask, crop_h, crop_w, fill=0)
        region_mask = ensure_min_size_and_pad(region_mask, crop_h, crop_w, fill=0)

        if self.is_train:
            if random.random() < self.hflip_prob:
                img1 = TF.hflip(img1)
                img2 = TF.hflip(img2)
                img3 = TF.hflip(img3)
                mask = TF.hflip(mask)
                region_mask = TF.hflip(region_mask)

            if random.random() < self.vflip_prob:
                img1 = TF.vflip(img1)
                img2 = TF.vflip(img2)
                img3 = TF.vflip(img3)
                mask = TF.vflip(mask)
                region_mask = TF.vflip(region_mask)

            if self.rotation_deg and self.rotation_deg > 0:
                angle = random.uniform(-self.rotation_deg, self.rotation_deg)
                img1 = TF.rotate(img1, angle, interpolation=Image.BILINEAR)
                img2 = TF.rotate(img2, angle, interpolation=Image.BILINEAR)
                img3 = TF.rotate(img3, angle, interpolation=Image.BILINEAR)
                mask = TF.rotate(mask, angle, interpolation=Image.NEAREST)
                region_mask = TF.rotate(region_mask, angle, interpolation=Image.NEAREST)

            w, h = img1.size
            left = random.randint(0, max(0, w - crop_w))
            top = random.randint(0, max(0, h - crop_h))

        else:
            w, h = img1.size
            left = max(0, (w - crop_w) // 2)
            top = max(0, (h - crop_h) // 2)

        img1 = TF.crop(img1, top, left, crop_h, crop_w)
        img2 = TF.crop(img2, top, left, crop_h, crop_w)
        img3 = TF.crop(img3, top, left, crop_h, crop_w)
        mask = TF.crop(mask, top, left, crop_h, crop_w)
        region_mask = TF.crop(region_mask, top, left, crop_h, crop_w)

        img1_t = TF.to_tensor(img1)
        img2_t = TF.to_tensor(img2)
        img3_t = TF.to_tensor(img3)

        mask_t = TF.to_tensor(mask)
        mask_t = (mask_t > 0.5).float()

        region_t = TF.to_tensor(region_mask)
        region_t = (region_t > 0.5).float()

        img1_t = TF.normalize(img1_t, mean=self.mean, std=self.std)
        img2_t = TF.normalize(img2_t, mean=self.mean, std=self.std)
        img3_t = TF.normalize(img3_t, mean=self.mean, std=self.std)

        return {"img1": img1_t, "img2": img2_t, "img3": img3_t, "mask": mask_t, "region_mask": region_t}


class TripletImageMaskDatasetFromRoot(Dataset):
    """
    Dataset that reads items from a root directory where each item is a subdirectory.

    Each item folder should contain files starting with:
      - image_i0*
      - image_i1*
      - image_i2*
      - boundary_mask*
      - nutrient_mask_g0*
    """

    def __init__(self, root: str, transforms: Optional[Callable] = None) -> None:
        self.root = Path(root)
        assert self.root.exists(), f"Root {root} does not exist"
        self.items = sorted([p for p in self.root.iterdir() if p.is_dir()])
        self.transforms = transforms

    def __len__(self) -> int:
        """
        :return: Number of items discovered under root.
        """
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load and return tensors for one item.

        :param idx: Index of item.
        :return: Dict with 'img1','img2','img3','mask','region_mask'.
        """
        item_dir = self.items[idx]

        f0 = find_file_by_prefix(item_dir, "image_i0")
        f1 = find_file_by_prefix(item_dir, "image_i1")
        f2 = find_file_by_prefix(item_dir, "image_i2")
        fb = find_file_by_prefix(item_dir, "bounday_mask")
        fm = find_file_by_prefix(item_dir, "nutrient_mask_g0")

        if any(x is None for x in (f0, f1, f2, fb, fm)):
            missing = [name for name, x in (("image_i0", f0), ("image_i1", f1), ("image_i2", f2), ("boundary_mask", fb), ("nutrient_mask_g0", fm)) if x is None]
            raise FileNotFoundError(f"Missing files in {item_dir}: {missing}")

        i1 = Image.open(f0).convert("RGB")
        i2 = Image.open(f1).convert("RGB")
        i3 = Image.open(f2).convert("RGB")

        region = Image.open(fb).convert("L")
        mask = Image.open(fm).convert("L")

        if self.transforms is not None:
            out = self.transforms(i1, i2, i3, mask, region)
            return out

        to_tensor = transforms.ToTensor()

        i1_t = to_tensor(i1)
        i2_t = to_tensor(i2)
        i3_t = to_tensor(i3)

        mask_t = (to_tensor(mask) > 0.5).float()
        region_t = (to_tensor(region) > 0.5).float()

        return {"img1": i1_t, "img2": i2_t, "img3": i3_t, "mask": mask_t, "region_mask": region_t}
