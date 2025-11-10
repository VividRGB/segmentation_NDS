from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt-style residual block used in decoder.

    :param dim: Number of channels in/out.
    :param drop_path_prob: Simple stochastic depth probability.
    :param layer_scale_init_value: LayerScale initialization scalar.
    """

    def __init__(self, dim: int, drop_path_prob: float = 0.0, layer_scale_init_value: float = 1e-6):
        super().__init__()

        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        # LayerNorm applied on channel-last, consistent with ConvNeXt
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pw1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pw2 = nn.Linear(4 * dim, dim)

        if layer_scale_init_value > 0:
            self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
        else:
            self.gamma = None

        self.drop_path_prob = drop_path_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        :param x: Tensor (N, C, H, W)
        :return: Tensor (N, C, H, W)
        """
        residual = x
        x = self.dwconv(x)
        # N,H,W,C for LayerNorm
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)

        if self.gamma is not None:
            x = x * self.gamma
        x = x.permute(0, 3, 1, 2)

        if self.drop_path_prob > 0.0 and self.training:
            if torch.rand(1).item() < self.drop_path_prob:
                return residual

        return residual + x


class ConvNeXtFuse(nn.Module):
    """
    ConvNeXt-style 1x1 fusion block:
      Conv2d(1x1) -> LayerNorm (channel-last) -> GELU

    Uses channel-last LayerNorm to match ConvNeXt block style.
    """
    def __init__(self, in_ch: int, out_ch: int, eps: float = 1e-6):
        super().__init__()

        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.norm = nn.LayerNorm(out_ch, eps=eps)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, H, W)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.act(x)
        x = x.permute(0, 3, 1, 2)
        return x


class ConvNeXtDecoderBlock(nn.Module):
    """
    Decoder block that upsamples and applies ConvNeXt blocks.

    :param in_ch: Channels from deeper stage.
    :param skip_ch: Channels from skip connection.
    :param out_ch: Output channels.
    :param num_blocks: Number of ConvNeXt blocks to apply.
    :param drop_path_prob: Stochastic depth probability passed to blocks.
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, num_blocks: int = 2, drop_path_prob: float = 0.0):
        super().__init__()

        self.deconv = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2)
        # keep bias=False to be consistent when followed by ConvNeXtBlock (which does LayerNorm)
        self.proj = nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size=1, bias=False)

        blocks = []
        for _ in range(num_blocks):
            blocks.append(ConvNeXtBlock(out_ch, drop_path_prob=drop_path_prob))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for decoder block.

        :param x: Deep feature (N, in_ch, H, W).
        :param skip: Skip feature (N, skip_ch, H*2, W*2).
        :return: Output (N, out_ch, H*2, W*2).
        """
        x = self.deconv(x)

        if x.size(-2) != skip.size(-2) or x.size(-1) != skip.size(-1):
            x = F.interpolate(x, size=(skip.size(-2), skip.size(-1)), mode="bilinear", align_corners=False)

        x = torch.cat([x, skip], dim=1)
        x = self.proj(x)
        x = self.blocks(x)

        return x


class TripletConvNeXtUNet(nn.Module):
    """
    UNet-like model that fuses three image encodings and decodes with ConvNeXt blocks.

    :param pretrained: Whether to use pretrained encoder weights.
    :param in_ch: Number of input channels per image.
    :param num_classes: Number of output channels (1 for binary mask logits).
    :param encoder_name: timm encoder model name.
    :param decoder_blocks_per_stage: Number of ConvNeXt blocks per decoder stage.
    """

    def __init__(
        self,
        pretrained: bool = True,
        in_ch: int = 3,
        num_classes: int = 1,
        encoder_name: str = "convnext_tiny",
        decoder_blocks_per_stage: int = 2,
    ) -> None:
        super().__init__()

        self.encoder = timm.create_model(encoder_name, pretrained=pretrained, features_only=True, in_chans=in_ch)

        feat_channels: List[int] = self.encoder.feature_info.channels()

        assert len(feat_channels) >= 4

        self.fuse_convs = nn.ModuleList()
        for ch in feat_channels[:4]:
            self.fuse_convs.append(ConvNeXtFuse(in_ch=ch * 3, out_ch=ch))

        rev_channels = feat_channels[:4][::-1]

        self.decoder_blocks = nn.ModuleList()

        for i in range(len(rev_channels) - 1):
            in_ch_dec = rev_channels[i]
            skip_ch = rev_channels[i + 1]
            out_ch_dec = rev_channels[i + 1]

            self.decoder_blocks.append(
                ConvNeXtDecoderBlock(in_ch=in_ch_dec, skip_ch=skip_ch, out_ch=out_ch_dec, num_blocks=decoder_blocks_per_stage)
            )

        self.final_conv = nn.Sequential(ConvNeXtBlock(rev_channels[-1]), nn.Conv2d(rev_channels[-1], num_classes, kernel_size=1))


    def forward_single(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward a single image through the encoder.

        :param x: Single image tensor (N, C, H, W).
        :return: List of feature maps from encoder (shallow -> deep).
        """
        return self.encoder(x)

    def forward(self, img1: torch.Tensor, img2: torch.Tensor, img3: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for three images.

        :param img1: First image (N, C, H, W).
        :param img2: Second image (N, C, H, W).
        :param img3: Third image (N, C, H, W).
        :return: Logits (N, 1, H, W).
        """
        feats1 = self.forward_single(img1)
        feats2 = self.forward_single(img2)
        feats3 = self.forward_single(img3)

        fused: List[torch.Tensor] = []

        for i, fuse_conv in enumerate(self.fuse_convs):
            a, b, c = feats1[i], feats2[i], feats3[i]
            x = torch.cat([a, b, c], dim=1)
            x = fuse_conv(x)
            fused.append(x)

        x = fused[-1]

        fused_rev = fused[::-1]

        for i, block in enumerate(self.decoder_blocks):
            skip = fused_rev[i + 1]
            x = block(x, skip)

        logits = self.final_conv(x)

        target_h, target_w = img1.shape[-2], img1.shape[-1]

        if logits.shape[-2:] != (target_h, target_w):
            logits = F.interpolate(logits, size=(target_h, target_w), mode="bilinear", align_corners=False)

        return logits
