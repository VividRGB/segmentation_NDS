from typing import Tuple, Optional, Dict, Union
from pathlib import Path

import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from tqdm import tqdm

from model import TripletConvNeXtUNet
from utils import find_file_by_prefix


def pad_to_multiple(img: Image.Image, mult: int = 32, fill: int = 0) -> Tuple[Image.Image, Tuple[int, int]]:
    """
    Pad image on right & bottom so height and width are multiples of `mult`.

    :param img: PIL image.
    :param mult: Multiple to pad to.
    :param fill: Padding fill value.
    :return: (padded_image, (pad_h, pad_w)) where pad_h/pad_w are added to bottom/right.
    """
    w, h = img.size
    pad_w = (mult - (w % mult)) % mult
    pad_h = (mult - (h % mult)) % mult
    if pad_h == 0 and pad_w == 0:
        return img, (0, 0)
    padded = TF.pad(img, padding=(0, 0, pad_w, pad_h), fill=fill)
    return padded, (pad_h, pad_w)


class InferenceDataset(Dataset):
    """
    Dataset for inference over item subfolders.

    :param root: root directory containing item_* subfolders.
    :param mean: normalization mean.
    :param std: normalization std.
    """

    def __init__(self, root: str, mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
                 std: Tuple[float, float, float] = (0.229, 0.224, 0.225)):
        self.root = Path(root)
        assert self.root.exists(), f"Input root {root} does not exist"
        self.items = sorted([p for p in self.root.iterdir() if p.is_dir()])
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, Tuple[int, int]]]:
        """
        :param idx: index
        :return: dict with keys:
            'img1','img2','img3' - tensors normalized, padded to multiple of 32
            'orig_size' - (h,w) of original images
            'pads' - (pad_h, pad_w)
            'region_path' - Path to boundary mask or None
            'region_mask' - tensor region mask padded (or None)
            'item_path' - folder name (for naming)
        """
        item_dir = self.items[idx]
        f0 = find_file_by_prefix(item_dir, "image_i0")
        f1 = find_file_by_prefix(item_dir, "image_i1")
        f2 = find_file_by_prefix(item_dir, "image_i2")
        fb = find_file_by_prefix(item_dir, "boundary_mask")

        if any(x is None for x in (f0, f1, f2)):
            missing = [name for name, x in (("image_i0", f0), ("image_i1", f1), ("image_i2", f2)) if x is None]
            raise FileNotFoundError(f"Missing image files in {item_dir}: {missing}")

        i1 = Image.open(f0).convert("RGB")
        i2 = Image.open(f1).convert("RGB")
        i3 = Image.open(f2).convert("RGB")

        orig_w, orig_h = i1.size

        if i2.size != i1.size:
            i2 = i2.resize(i1.size, Image.BILINEAR)
        if i3.size != i1.size:
            i3 = i3.resize(i1.size, Image.BILINEAR)

        i1_p, pads = pad_to_multiple(i1, mult=32, fill=0)
        i2_p, _ = pad_to_multiple(i2, mult=32, fill=0)
        i3_p, _ = pad_to_multiple(i3, mult=32, fill=0)

        i1_t = TF.to_tensor(i1_p)
        i2_t = TF.to_tensor(i2_p)
        i3_t = TF.to_tensor(i3_p)

        i1_t = TF.normalize(i1_t, mean=self.mean, std=self.std)
        i2_t = TF.normalize(i2_t, mean=self.mean, std=self.std)
        i3_t = TF.normalize(i3_t, mean=self.mean, std=self.std)

        region_path = str(fb) if fb is not None else None
        region_t = None
        if region_path is not None:
            region_img = Image.open(region_path).convert("L")
            region_p, _ = pad_to_multiple(region_img, mult=32, fill=0)
            region_t = TF.to_tensor(region_p)
            region_t = (region_t > 0.5).float()

        return {
            "img1": i1_t,
            "img2": i2_t,
            "img3": i3_t,
            "orig_size": (orig_h, orig_w),
            "pads": pads,
            "region_path": region_path,
            "region_mask": region_t,
            "item_path": str(item_dir.name),
        }


def load_checkpoint(model: nn.Module, ckpt_path: str, device: torch.device) -> None:
    """
    Load model_state from checkpoint.

    :param model: model instance.
    :param ckpt_path: path to checkpoint file (may contain 'model_state' key).
    :param device: device to map to.
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state = ckpt["model_state"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt
    model.load_state_dict(state, strict=False)


def save_mask_array_as_png(arr: np.ndarray, out_path: str) -> None:
    """
    Save a 2D float array in [0,1] as 8-bit PNG.

    :param arr: 2D numpy array floats 0..1
    :param out_path: output path
    """
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(out_path)


def run_inference_item_sliding(
        model: nn.Module,
        device: torch.device,
        item: Dict[str, Union[torch.Tensor, str, Tuple[int, int]]],
        output_dir: str,
        patch_size: int = 512,
        overlap: float = 0.25,
        window_batch_size: int = 4,
        threshold: float = 0.5,
        apply_region: bool = True,
        save_prob_as_npy: bool = False,
) -> None:
    """
    Run sliding-window inference on a single item (from dataset[idx]).

    :param model: model on device.
    :param device: torch device.
    :param item: single-item dict returned by dataset.__getitem__.
    :param output_dir: where to save outputs.
    :param patch_size: patch size for sliding window (square).
    :param overlap: fraction overlap between windows (0..0.9).
    :param window_batch_size: how many windows to batch into a single forward pass.
    :param threshold: binarization threshold.
    :param apply_region: if True apply boundary mask.
    :param save_prob_as_npy: if True save .npy probability.
    """
    item_name = item["item_path"]
    img1_t: torch.Tensor = item["img1"]
    img2_t: torch.Tensor = item["img2"]
    img3_t: torch.Tensor = item["img3"]
    orig_h, orig_w = item["orig_size"]
    pad_h, pad_w = item["pads"]
    region_mask_t = item.get("region_mask", None)

    _, Hp, Wp = img1_t.shape

    stride = int(patch_size * (1.0 - overlap))
    stride = max(1, stride)

    ys = list(range(0, max(1, Hp - patch_size + 1), stride))
    xs = list(range(0, max(1, Wp - patch_size + 1), stride))

    # ensure last patch covers the right/bottom border
    if ys[-1] + patch_size < Hp:
        ys.append(max(0, Hp - patch_size))
    if xs[-1] + patch_size < Wp:
        xs.append(max(0, Wp - patch_size))

    accum = np.zeros((Hp, Wp), dtype=np.float32)
    counts = np.zeros((Hp, Wp), dtype=np.float32)

    model.eval()

    windows = []
    coords = []

    # gather windows
    for y in ys:
        for x in xs:
            y1, x1 = y, x
            y2, x2 = min(y + patch_size, Hp), min(x + patch_size, Wp)

            crop1 = img1_t[:, y1:y2, x1:x2]
            crop2 = img2_t[:, y1:y2, x1:x2]
            crop3 = img3_t[:, y1:y2, x1:x2]

            windows.append((crop1, crop2, crop3))
            coords.append((y1, y2, x1, x2))

            if len(windows) >= window_batch_size:
                # process batch
                b_img1 = torch.stack([w[0] for w in windows], dim=0).to(device)
                b_img2 = torch.stack([w[1] for w in windows], dim=0).to(device)
                b_img3 = torch.stack([w[2] for w in windows], dim=0).to(device)

                with torch.no_grad():
                    with torch.amp.autocast(device.type):
                        logits = model(b_img1, b_img2, b_img3)  # (B,1,ph,pw)
                        probs = torch.sigmoid(logits).cpu().numpy()  # B,1,ph,pw

                for i_win, (y1_i, y2_i, x1_i, x2_i) in enumerate(coords[: len(windows)]):
                    ph_np = probs[i_win, 0]
                    accum[y1_i:y2_i, x1_i:x2_i] += ph_np
                    counts[y1_i:y2_i, x1_i:x2_i] += 1.0

                windows = []
                coords = []

    # leftover windows
    if len(windows) > 0:
        b_img1 = torch.stack([w[0] for w in windows], dim=0).to(device)
        b_img2 = torch.stack([w[1] for w in windows], dim=0).to(device)
        b_img3 = torch.stack([w[2] for w in windows], dim=0).to(device)

        with torch.no_grad():
            with torch.amp.autocast(device.type):
                logits = model(b_img1, b_img2, b_img3)
                probs = torch.sigmoid(logits).cpu().numpy()

        for i_win, (y1_i, y2_i, x1_i, x2_i) in enumerate(coords):
            ph_np = probs[i_win, 0]
            accum[y1_i:y2_i, x1_i:x2_i] += ph_np
            counts[y1_i:y2_i, x1_i:x2_i] += 1.0

    counts[counts == 0] = 1.0
    avg = accum / counts

    if pad_h > 0:
        avg = avg[:-pad_h, :]
    if pad_w > 0:
        avg = avg[:, :-pad_w]

    if apply_region and region_mask_t is not None:
        r = region_mask_t.squeeze(0).numpy()
        if pad_h > 0:
            r = r[:-pad_h, :]
        if pad_w > 0:
            r = r[:, :-pad_w]
        avg = avg * (r.astype(avg.dtype))

    # ensure final size matches orig_h, orig_w
    avg = avg[:orig_h, :orig_w]

    # binary mask
    bin_np = (avg > threshold).astype(np.uint8) * 255

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prob_png = out_dir / f"{item_name}_prob.png"
    bin_png = out_dir / f"{item_name}_pred_mask.png"

    save_mask_array_as_png(avg, str(prob_png))
    Image.fromarray(bin_np).save(str(bin_png))
    if save_prob_as_npy:
        prob_npy = out_dir / f"{item_name}_prob.npy"
        np.save(str(prob_npy), avg)

    print(f"Saved (sliding) {prob_png}, {bin_png}" + (f", {prob_npy}" if save_prob_as_npy else ""))


def run_inference(
        model: nn.Module,
        dataset: InferenceDataset,
        device: torch.device,
        output_dir: str,
        threshold: float = 0.5,
        apply_region: bool = True,
        save_prob_as_npy: bool = False,
        patch_size: Optional[int] = None,
        overlap: float = 0.25,
        window_batch_size: int = 4,
) -> None:
    """
    Run inference over all items. The inference uses
    sliding-window inference will be used per-item; otherwise full-image inference is used.

    :param model: model on device.
    :param dataset: InferenceDataset instance.
    :param device: torch device.
    :param output_dir: output directory.
    :param threshold: binarization threshold.
    :param apply_region: apply boundary mask if present.
    :param save_prob_as_npy: also save probability arrays as .npy.
    :param patch_size: if set, use sliding-window with this patch size.
    :param overlap: overlap fraction for sliding windows (0..0.9).
    :param window_batch_size: batch size for window forwards.
    """
    # per-item sliding window inference (process items sequentially)
    for idx in tqdm(range(len(dataset))):
        item = dataset[idx]
        run_inference_item_sliding(
            model=model,
            device=device,
            item=item,
            output_dir=output_dir,
            patch_size=patch_size,
            overlap=overlap,
            window_batch_size=window_batch_size,
            threshold=threshold,
            apply_region=apply_region,
            save_prob_as_npy=save_prob_as_npy,
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inference for Triplet ConvNeXt segmentation (supports sliding-window).")
    p.add_argument("--model-path", type=str, required=True,
                   help="Path to checkpoint (contains 'model_state' or state_dict).")
    p.add_argument("--input-root", type=str, required=True, help="Root folder with item_* subfolders.")
    p.add_argument("--output-dir", type=str, required=True, help="Where to save predicted masks.")
    p.add_argument("--device", type=str, default="cuda", help="Device string, e.g. 'cuda' or 'cpu'.")
    p.add_argument("--threshold", type=float, default=0.5, help="Threshold for binary mask.")
    p.add_argument("--apply-region", action="store_true",
                   help="If set, zero predictions outside the boundary mask (if present).")
    p.add_argument("--save-prob-npy", action="store_true", help="Also save probability maps as .npy.")
    p.add_argument("--pretrained", action="store_true",
                   help="Initialize encoder with pretrained weights (does not affect loading checkpoint).")
    p.add_argument("--decoder-blocks", type=int, default=2, help="Decoder blocks per stage (model init).")
    p.add_argument("--patch-size", type=int, default=512,
                   help="If set, use sliding-window inference with this square patch size.")
    p.add_argument("--overlap", type=float, default=0.25, help="Fractional overlap for sliding windows (0..0.9).")
    p.add_argument("--window-batch-size", type=int, default=4,
                   help="Batch windows into this many for a single forward pass.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    dataset = InferenceDataset(args.input_root, mean=mean, std=std)

    model = TripletConvNeXtUNet(pretrained=args.pretrained, in_ch=3, num_classes=1, encoder_name="convnext_tiny",
                                decoder_blocks_per_stage=args.decoder_blocks)
    load_checkpoint(model, args.model_path, device)

    model.to(device)
    model.eval()

    run_inference(
        model=model,
        dataset=dataset,
        device=device,
        output_dir=args.output_dir,
        threshold=args.threshold,
        apply_region=args.apply_region,
        save_prob_as_npy=args.save_prob_npy,
        patch_size=args.patch_size,
        overlap=args.overlap,
        window_batch_size=args.window_batch_size,
    )


if __name__ == "__main__":
    main()
