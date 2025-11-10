from typing import Tuple, Optional, Union
from pathlib import Path

import argparse
import os
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data_transform import SegmentationTransformNoResize, TripletImageMaskDatasetFromRoot
from model import TripletConvNeXtUNet


def binary_focal_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: Union[float, torch.Tensor] = 0.25,
    gamma: float = 2.0,
    region_mask: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute binary focal loss (optionally inside a region).

    :param logits: Tensor of shape (N,1,H,W) containing raw logits.
    :param targets: Binary tensor same shape as logits (0/1).
    :param alpha: Scalar or tensor broadcastable to targets (per-sample alpha supported).
    :param gamma: Focusing parameter for focal loss.
    :param region_mask: Optional binary mask (N,1,H,W) where loss is computed (1 = valid).
    :param reduction: "mean", "sum", or "none".
    :return: Loss scalar or per-pixel loss tensor.
    """
    probs = torch.sigmoid(logits)

    targets = targets.type_as(probs)

    bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

    p_t = probs * targets + (1 - probs) * (1 - targets)
    modulating = (1 - p_t) ** gamma

    if isinstance(alpha, torch.Tensor):
        alpha_t = alpha.to(dtype=probs.dtype, device=probs.device)
    else:
        alpha_t = float(alpha)

    alpha_factor = alpha_t * targets + (1 - alpha_t) * (1 - targets)

    loss = alpha_factor * modulating * bce_loss

    if region_mask is not None:
        mask = (region_mask > 0.5).type_as(loss)
        loss = loss * mask

        if reduction == "mean":
            denom = mask.sum()
            if denom.item() == 0:
                return torch.tensor(0.0, device=loss.device, dtype=loss.dtype)
            return loss.sum() / (denom + 1e-6)

        elif reduction == "sum":
            return loss.sum()

        else:
            return loss

    else:
        if reduction == "mean":
            return loss.mean()

        elif reduction == "sum":
            return loss.sum()

        else:
            return loss


def soft_dice_loss(
    probs: torch.Tensor,
    targets: torch.Tensor,
    region_mask: Optional[torch.Tensor] = None,
    smooth: float = 1e-6,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute soft Dice loss (optionally inside a region).

    :param probs: Tensor with probabilities (after sigmoid) shape (N,1,H,W).
    :param targets: Binary tensor same shape as probs.
    :param region_mask: Optional binary mask (N,1,H,W) to restrict computation.
    :param smooth: Small constant for numerical stability.
    :param reduction: "mean", "sum", or "none".
    :return: Loss scalar or per-sample loss if reduction='none'.
    """
    N = probs.shape[0]

    probs_flat = probs.view(N, -1)
    targets_flat = targets.view(N, -1).type_as(probs_flat)

    if region_mask is not None:
        mask_flat = (region_mask.view(N, -1)).type_as(probs_flat)
        intersection = (probs_flat * targets_flat * mask_flat).sum(dim=1)
        denom = (probs_flat * mask_flat).sum(dim=1) + (targets_flat * mask_flat).sum(dim=1)
    else:
        intersection = (probs_flat * targets_flat).sum(dim=1)
        denom = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)

    dice = (2.0 * intersection + smooth) / (denom + smooth)
    loss = 1.0 - dice

    if reduction == "mean":
        return loss.mean()

    elif reduction == "sum":
        return loss.sum()

    else:
        return loss


class CombinedLoss(nn.Module):
    """
    Combined Focal + Dice loss with per-sample alpha and optional region masking.

    :param focal_gamma: Gamma parameter for focal loss.
    :param lambda_focal: Weight for focal loss term.
    :param lambda_dice: Weight for dice loss term.
    """

    def __init__(self, focal_gamma: float = 2.0, lambda_focal: float = 1.0, lambda_dice: float = 1.0) -> None:
        super().__init__()
        self.focal_gamma = focal_gamma
        self.lambda_focal = lambda_focal
        self.lambda_dice = lambda_dice

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        batch_alpha: Optional[torch.Tensor] = None,
        region_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute combined loss.

        :param logits: Raw logits (N,1,H,W).
        :param targets: Binary masks (N,1,H,W).
        :param batch_alpha: Optional Tensor of shape (N,) or (N,1,1,1) with per-sample alpha.
        :param region_mask: Optional binary mask (N,1,H,W) restricting loss computation.
        :return: Combined loss scalar.
        """
        if batch_alpha is None:
            alpha_for_loss: Union[float, torch.Tensor] = 0.25
        else:
            if batch_alpha.dim() == 1:
                alpha_for_loss = batch_alpha.view(-1, 1, 1, 1)
            elif batch_alpha.dim() == 2 and batch_alpha.shape[1] == 1:
                alpha_for_loss = batch_alpha.view(-1, 1, 1, 1)
            else:
                alpha_for_loss = batch_alpha

        focal = binary_focal_loss_with_logits(
            logits,
            targets,
            alpha=alpha_for_loss,
            gamma=self.focal_gamma,
            region_mask=region_mask,
            reduction="mean",
        )

        probs = torch.sigmoid(logits)

        dice = soft_dice_loss(probs, targets, region_mask=region_mask, smooth=1e-6, reduction="mean")

        return self.lambda_focal * focal + self.lambda_dice * dice


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
):
    """
    Create a LambdaLR scheduler for linear warmup followed by cosine decay.

    :param optimizer: Optimizer.
    :param num_warmup_steps: Number of warmup steps.
    :param num_training_steps: Total number of training steps.
    :param num_cycles: Number of cosine cycles.
    :return: LambdaLR scheduler.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))

        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * 2.0 * num_cycles * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    epoch: int,
    writer: SummaryWriter,
    scaler: torch.amp.GradScaler,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    max_grad_norm: Optional[float] = None,
) -> float:
    """
    Train for one epoch.

    :return: Average loss.
    """
    model.train()

    running_loss = 0.0
    global_step_base = epoch * len(dataloader)

    for step, batch in enumerate(dataloader):

        img1 = batch["img1"].to(device)
        img2 = batch["img2"].to(device)
        img3 = batch["img3"].to(device)

        mask = batch["mask"].to(device)
        region = batch["region_mask"].to(device)

        optimizer.zero_grad()

        with torch.amp.autocast(device.type):
            logits = model(img1, img2, img3)

            eps = 1e-6
            N = mask.shape[0]

            mask_flat = mask.view(N, -1)
            region_flat = region.view(N, -1)

            pos = (mask_flat * region_flat).sum(dim=1)
            total_region = region_flat.sum(dim=1)
            neg = total_region - pos

            batch_alpha = torch.where(
                total_region > 0,
                neg / (pos + neg + eps),
                torch.tensor(0.25, device=neg.device, dtype=neg.dtype),
            )

            batch_alpha_tensor = batch_alpha.view(N, 1, 1, 1)

            loss = criterion(logits, mask, batch_alpha=batch_alpha_tensor, region_mask=region)

        scaler.scale(loss).backward()

        if max_grad_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item()

        if step % 10 == 0:
            global_step = global_step_base + step
            writer.add_scalar("train/loss_step", loss.item(), global_step)
            writer.add_scalar("train/mean_batch_alpha", batch_alpha.mean().item(), global_step)

            b = 0
            try:
                writer.add_image("train/gt", mask[b].cpu().repeat(3, 1, 1), global_step)
                pred_prob = torch.sigmoid(logits)[b].detach().cpu()
                writer.add_image("train/pred", pred_prob.repeat(3, 1, 1), global_step)
            except Exception:
                pass

    avg_loss = running_loss / max(1, len(dataloader))
    writer.add_scalar("train/loss_epoch", avg_loss, epoch)

    return avg_loss


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    epoch: int,
    writer: SummaryWriter,
    n_image_log: int = 4,
) -> Tuple[float, float, float]:
    """
    Validate model.

    :return: (avg_loss, mean_dice, mean_f1, mean_iou).
    """
    model.eval()

    val_loss = 0.0
    dices = []
    f1s = []
    ious = []

    with torch.no_grad():
        for step, batch in enumerate(dataloader):

            img1 = batch["img1"].to(device)
            img2 = batch["img2"].to(device)
            img3 = batch["img3"].to(device)

            mask = batch["mask"].to(device)
            region = batch["region_mask"].to(device)

            with torch.amp.autocast(device.type):
                logits = model(img1, img2, img3)

                N = mask.shape[0]
                mask_flat = mask.view(N, -1)
                region_flat = region.view(N, -1)

                pos = (mask_flat * region_flat).sum(dim=1)
                total_region = region_flat.sum(dim=1)
                neg = total_region - pos

                batch_alpha = torch.where(
                    total_region > 0,
                    neg / (pos + neg + 1e-6),
                    torch.tensor(0.25, device=neg.device, dtype=neg.dtype),
                )

                batch_alpha_tensor = batch_alpha.view(N, 1, 1, 1)

                loss = criterion(logits, mask, batch_alpha=batch_alpha_tensor, region_mask=region)

            val_loss += loss.item()

            probs = torch.sigmoid(logits)

            per_sample_dice = 1.0 - soft_dice_loss(probs, mask, region_mask=region, reduction="none")

            if isinstance(per_sample_dice, torch.Tensor):
                dices.extend(per_sample_dice.detach().cpu().numpy().tolist())

            # compute F1 (binary) inside region mask using threshold 0.5
            eps = 1e-6
            preds_bin = (probs > 0.5).float()  # (N,1,H,W)

            tp = (preds_bin * mask * region).view(N, -1).sum(dim=1)
            fp = (preds_bin * (1.0 - mask) * region).view(N, -1).sum(dim=1)
            fn = ((1.0 - preds_bin) * mask * region).view(N, -1).sum(dim=1)

            f1_per_sample = torch.where(
                total_region > 0,
                2.0 * tp / (2.0 * tp + fp + fn + eps),
                torch.zeros_like(tp),
            )

            f1s.extend(f1_per_sample.detach().cpu().numpy().tolist())

            # --- IoU (Intersection over Union) per-sample inside region ---
            iou_per_sample = torch.where(
                total_region > 0,
                tp / (tp + fp + fn + eps),
                torch.zeros_like(tp),
            )
            ious.extend(iou_per_sample.detach().cpu().numpy().tolist())
            # -------------------------------------------------------------

            if step < int(n_image_log):
                b = 0
                try:
                    writer.add_image(f"val/gt_{step}", mask[b].cpu().repeat(3, 1, 1), epoch)
                    writer.add_image(f"val/pred_{step}", probs[b].cpu().repeat(3, 1, 1), epoch)
                except Exception:
                    pass

    avg_loss = val_loss / max(1, len(dataloader))
    mean_dice = float(np.mean(dices)) if len(dices) > 0 else 0.0
    mean_f1 = float(np.mean(f1s)) if len(f1s) > 0 else 0.0
    mean_iou = float(np.mean(ious)) if len(ious) > 0 else 0.0

    writer.add_scalar("val/loss", avg_loss, epoch)
    writer.add_scalar("val/dice", mean_dice, epoch)
    writer.add_scalar("val/f1", mean_f1, epoch)
    writer.add_scalar("val/iou", mean_iou, epoch)

    return avg_loss, mean_dice, mean_f1, mean_iou


def train_loop(
    train_root: str,
    val_root: str,
    save_dir: str = "runs/exp_convnext_boundary",
    epochs: int = 30,
    batch_size: int = 8,
    lr: float = 1e-4,
    pretrained: bool = True,
    device: Optional[str] = None,
    weight_decay: float = 1e-4,
    num_warmup_epochs: int = 3,
    num_cycles: float = 0.5,
    num_workers: int = 4,
    decoder_blocks_per_stage: int = 2,
) -> nn.Module:
    """
    Full training loop.
    """
    device_obj = torch.device(device)

    writer = SummaryWriter(log_dir=save_dir)

    crop_size = (512, 512)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    train_transform = SegmentationTransformNoResize(
        crop=crop_size,
        hflip_prob=0.5,
        vflip_prob=0.1,
        rotation_deg=15.0,
        mean=mean,
        std=std,
        is_train=True,
    )

    val_transform = SegmentationTransformNoResize(
        crop=crop_size,
        hflip_prob=0.0,
        vflip_prob=0.0,
        rotation_deg=0.0,
        mean=mean,
        std=std,
        is_train=False,
    )

    train_ds = TripletImageMaskDatasetFromRoot(train_root, transforms=train_transform)
    val_ds = TripletImageMaskDatasetFromRoot(val_root, transforms=val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=max(1, batch_size // 2), shuffle=False, num_workers=max(1, num_workers // 2), pin_memory=True)

    model = TripletConvNeXtUNet(pretrained=pretrained, in_ch=3, num_classes=1, decoder_blocks_per_stage=decoder_blocks_per_stage).to(device_obj)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    total_steps = epochs * len(train_loader)
    warmup_steps = num_warmup_epochs * len(train_loader)

    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps, num_cycles=num_cycles)

    criterion = CombinedLoss(focal_gamma=2.0, lambda_focal=1.0, lambda_dice=1.0)

    scaler = torch.amp.GradScaler(enabled=(device_obj.type == "cuda"))

    best_val = float("inf")
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, optimizer, train_loader, device_obj, criterion, epoch, writer, scaler, scheduler, max_grad_norm=1.0)

        val_loss, val_dice, val_f1, val_iou = validate(model, val_loader, device_obj, criterion, epoch, writer)

        print(f"[Epoch {epoch:03d}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_dice={val_dice:.4f}, val_f1={val_f1:.4f}, val_iou={val_iou:.4f}")

        writer.add_scalar("meta/lr", optimizer.param_groups[0]["lr"], epoch)

        if epoch % 10 == 0:
            ckpt_path = Path(save_dir) / f"checkpoint_epoch{epoch:03d}.pt"
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "optim_state": optimizer.state_dict(), "val_loss": val_loss, "val_dice": val_dice}, ckpt_path)

        if val_loss < best_val:
            best_val = val_loss
            best_path = Path(save_dir) / "best_model.pt"
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "optim_state": optimizer.state_dict(), "val_loss": val_loss, "val_dice": val_dice}, best_path)

    writer.close()

    return model


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    :return: Parsed arguments namespace.
    """
    p = argparse.ArgumentParser(description="Train Triplet ConvNeXt segmentation with boundary-region masked losses.")
    p.add_argument("--train-root", type=str, required=True, help="Train root folder (contains item_* subfolders)")
    p.add_argument("--val-root", type=str, required=True, help="Val root folder (contains item_* subfolders)")
    p.add_argument("--save-dir", type=str, default="runs/exp_convnext_boundary", help="Where to write logs and checkpoints")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--pretrained", action="store_true", help="Use pretrained encoder")
    p.add_argument("--device", type=str, default="cuda", help="Device string e.g. 'cuda' or 'cpu'")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-epochs", type=int, default=20)
    p.add_argument("--num-cycles", type=float, default=0.5)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--decoder-blocks", type=int, default=2, help="Number of ConvNeXt blocks per decoder stage")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    train_loop(
        train_root=args.train_root,
        val_root=args.val_root,
        save_dir=args.save_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        pretrained=args.pretrained,
        device=args.device,
        weight_decay=args.weight_decay,
        num_warmup_epochs=args.warmup_epochs,
        num_cycles=args.num_cycles,
        num_workers=args.num_workers,
        decoder_blocks_per_stage=args.decoder_blocks,
    )
