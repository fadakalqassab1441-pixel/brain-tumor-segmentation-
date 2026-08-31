"""A small Unet-like zoo"""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint_sequential

from src.models.layers import ConvBnRelu, UBlock, conv1x1, UBlockCbam, CBAM, UBlockSE, SELayer3D


class Unet(nn.Module):
    """Almost the most basic U-net.
    """
    name = "Unet"

    def __init__(self, inplanes, num_classes, width, norm_layer=None, deep_supervision=False, dropout=0,
                 **kwargs):
        super(Unet, self).__init__()
        features = [width * 2 ** i for i in range(4)]
        print(features)

        self.deep_supervision = deep_supervision

        self.encoder1 = UBlock(inplanes, features[0] // 2, features[0], norm_layer, dropout=dropout)
        self.encoder2 = UBlock(features[0], features[1] // 2, features[1], norm_layer, dropout=dropout)
        self.encoder3 = UBlock(features[1], features[2] // 2, features[2], norm_layer, dropout=dropout)
        self.encoder4 = UBlock(features[2], features[3] // 2, features[3], norm_layer, dropout=dropout)

        self.bottom = UBlock(features[3], features[3], features[3], norm_layer, (2, 2), dropout=dropout)

        self.bottom_2 = ConvBnRelu(features[3] * 2, features[2], norm_layer, dropout=dropout)

        self.downsample = nn.MaxPool3d(2, 2)

        self.decoder3 = UBlock(features[2] * 2, features[2], features[1], norm_layer, dropout=dropout)
        self.decoder2 = UBlock(features[1] * 2, features[1], features[0], norm_layer, dropout=dropout)
        self.decoder1 = UBlock(features[0] * 2, features[0], features[0] // 2, norm_layer, dropout=dropout)

        self.upsample = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)

        self.outconv = conv1x1(features[0] // 2, num_classes)

        if self.deep_supervision:
            self.deep_bottom = nn.Sequential(
                conv1x1(features[3], num_classes),
                nn.Upsample(scale_factor=8, mode="trilinear", align_corners=True))

            self.deep_bottom2 = nn.Sequential(
                conv1x1(features[2], num_classes),
                nn.Upsample(scale_factor=8, mode="trilinear", align_corners=True))

            self.deep3 = nn.Sequential(
                conv1x1(features[1], num_classes),
                nn.Upsample(scale_factor=4, mode="trilinear", align_corners=True))

            self.deep2 = nn.Sequential(
                conv1x1(features[0], num_classes),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm, nn.InstanceNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        down1 = self.encoder1(x)
        down2 = self.downsample(down1)
        down2 = self.encoder2(down2)
        down3 = self.downsample(down2)
        down3 = self.encoder3(down3)
        down4 = self.downsample(down3)
        down4 = self.encoder4(down4)

        bottom = self.bottom(down4)
        bottom_2 = self.bottom_2(torch.cat([down4, bottom], dim=1))

        # Decoder

        up3 = self.upsample(bottom_2)
        up3 = self.decoder3(torch.cat([down3, up3], dim=1))
        up2 = self.upsample(up3)
        up2 = self.decoder2(torch.cat([down2, up2], dim=1))
        up1 = self.upsample(up2)
        up1 = self.decoder1(torch.cat([down1, up1], dim=1))

        out = self.outconv(up1)

        if self.deep_supervision:
            deeps = []
            for seg, deep in zip(
                    [bottom, bottom_2, up3, up2],
                    [self.deep_bottom, self.deep_bottom2, self.deep3, self.deep2]):
                deeps.append(deep(seg))
            return out, deeps

        return out


class EquiUnet(Unet):
    """Almost the most basic U-net: all Block have the same size if they are at the same level.
    """
    name = "EquiUnet"

    def __init__(self, inplanes, num_classes, width, norm_layer=None, deep_supervision=False, dropout=0,
                 **kwargs):
        super(Unet, self).__init__()
        features = [width * 2 ** i for i in range(4)]
        print(features)

        self.deep_supervision = deep_supervision

        self.encoder1 = UBlock(inplanes, features[0], features[0], norm_layer, dropout=dropout)
        self.encoder2 = UBlock(features[0], features[1], features[1], norm_layer, dropout=dropout)
        self.encoder3 = UBlock(features[1], features[2], features[2], norm_layer, dropout=dropout)
        self.encoder4 = UBlock(features[2], features[3], features[3], norm_layer, dropout=dropout)

        self.bottom = UBlock(features[3], features[3], features[3], norm_layer, (2, 2), dropout=dropout)

        self.bottom_2 = ConvBnRelu(features[3] * 2, features[2], norm_layer, dropout=dropout)

        self.downsample = nn.MaxPool3d(2, 2)

        self.decoder3 = UBlock(features[2] * 2, features[2], features[1], norm_layer, dropout=dropout)
        self.decoder2 = UBlock(features[1] * 2, features[1], features[0], norm_layer, dropout=dropout)
        self.decoder1 = UBlock(features[0] * 2, features[0], features[0], norm_layer, dropout=dropout)

        self.upsample = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)

        self.outconv = conv1x1(features[0], num_classes)

        if self.deep_supervision:
            self.deep_bottom = nn.Sequential(
                conv1x1(features[3], num_classes),
                nn.Upsample(scale_factor=8, mode="trilinear", align_corners=True))

            self.deep_bottom2 = nn.Sequential(
                conv1x1(features[2], num_classes),
                nn.Upsample(scale_factor=8, mode="trilinear", align_corners=True))

            self.deep3 = nn.Sequential(
                conv1x1(features[1], num_classes),
                nn.Upsample(scale_factor=4, mode="trilinear", align_corners=True))

            self.deep2 = nn.Sequential(
                conv1x1(features[0], num_classes),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True))

        self._init_weights()


class Att_EquiUnet(Unet):
    def __init__(self, inplanes, num_classes, width, norm_layer=None, deep_supervision=False,  dropout=0,
                 **kwargs):
        super(Unet, self).__init__()
        features = [width * 2 ** i for i in range(4)]
        print(features)

        self.deep_supervision = deep_supervision

        self.encoder1 = UBlockCbam(inplanes, features[0], features[0], norm_layer, dropout=dropout)
        self.encoder2 = UBlockCbam(features[0], features[1], features[1], norm_layer, dropout=dropout)
        self.encoder3 = UBlockCbam(features[1], features[2], features[2], norm_layer, dropout=dropout)
        self.encoder4 = UBlockCbam(features[2], features[3], features[3], norm_layer, dropout=dropout)

        self.bottom = UBlockCbam(features[3], features[3], features[3], norm_layer, (2, 2), dropout=dropout)

        self.bottom_2 = nn.Sequential(
            ConvBnRelu(features[3] * 2, features[2], norm_layer, dropout=dropout),
            CBAM(features[2], norm_layer=norm_layer)
        )

        self.downsample = nn.MaxPool3d(2, 2)

        self.decoder3 = UBlock(features[2] * 2, features[2], features[1], norm_layer, dropout=dropout)
        self.decoder2 = UBlock(features[1] * 2, features[1], features[0], norm_layer, dropout=dropout)
        self.decoder1 = UBlock(features[0] * 2, features[0], features[0], norm_layer, dropout=dropout)

        self.upsample = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)

        self.outconv = conv1x1(features[0], num_classes)

        if self.deep_supervision:
            self.deep_bottom = nn.Sequential(
                conv1x1(features[3], num_classes),
                nn.Upsample(scale_factor=8, mode="trilinear", align_corners=True))

            self.deep_bottom2 = nn.Sequential(
                conv1x1(features[2], num_classes),
                nn.Upsample(scale_factor=8, mode="trilinear", align_corners=True))

            self.deep3 = nn.Sequential(
                conv1x1(features[1], num_classes),
                nn.Upsample(scale_factor=4, mode="trilinear", align_corners=True))

            self.deep2 = nn.Sequential(
                conv1x1(features[0], num_classes),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True))

        self._init_weights()


class EquiUnetSE(Unet):
    """EquiUnet with Squeeze-and-Excitation channel attention in the decoder.

    Encoder and bottleneck are untouched plain UBlocks (identical to
    EquiUnet). Every decoder UBlock (decoder3/2/1) and the bottom_2 fusion
    conv are followed by an SE gate, so after each skip connection is merged
    back in, the network gets to reweight *which* encoder/decoder channels
    actually matter for that resolution before upsampling further and before
    the final 1x1 output conv.

    This is the standalone SE-decoder building block: useful on its own for
    an ablation against CascadedSEUnet below (SE alone vs. SE + cascade).
    """
    name = "EquiUnetSE"

    def __init__(self, inplanes, num_classes, width, norm_layer=None, deep_supervision=False, dropout=0,
                 se_reduction=8, **kwargs):
        super(Unet, self).__init__()
        features = [width * 2 ** i for i in range(4)]
        print(features)

        self.deep_supervision = deep_supervision

        # Encoder / bottleneck: unchanged from EquiUnet
        self.encoder1 = UBlock(inplanes, features[0], features[0], norm_layer, dropout=dropout)
        self.encoder2 = UBlock(features[0], features[1], features[1], norm_layer, dropout=dropout)
        self.encoder3 = UBlock(features[1], features[2], features[2], norm_layer, dropout=dropout)
        self.encoder4 = UBlock(features[2], features[3], features[3], norm_layer, dropout=dropout)

        self.bottom = UBlock(features[3], features[3], features[3], norm_layer, (2, 2), dropout=dropout)

        self.bottom_2 = nn.Sequential(
            ConvBnRelu(features[3] * 2, features[2], norm_layer, dropout=dropout),
            SELayer3D(features[2], reduction_ratio=se_reduction),
        )

        self.downsample = nn.MaxPool3d(2, 2)

        # Decoder: every UBlock gets an SE gate
        self.decoder3 = UBlockSE(features[2] * 2, features[2], features[1], norm_layer, dropout=dropout,
                                  se_reduction=se_reduction)
        self.decoder2 = UBlockSE(features[1] * 2, features[1], features[0], norm_layer, dropout=dropout,
                                  se_reduction=se_reduction)
        self.decoder1 = UBlockSE(features[0] * 2, features[0], features[0], norm_layer, dropout=dropout,
                                  se_reduction=se_reduction)

        self.upsample = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)

        self.outconv = conv1x1(features[0], num_classes)

        if self.deep_supervision:
            self.deep_bottom = nn.Sequential(
                conv1x1(features[3], num_classes),
                nn.Upsample(scale_factor=8, mode="trilinear", align_corners=True))

            self.deep_bottom2 = nn.Sequential(
                conv1x1(features[2], num_classes),
                nn.Upsample(scale_factor=8, mode="trilinear", align_corners=True))

            self.deep3 = nn.Sequential(
                conv1x1(features[1], num_classes),
                nn.Upsample(scale_factor=4, mode="trilinear", align_corners=True))

            self.deep2 = nn.Sequential(
                conv1x1(features[0], num_classes),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True))

        self._init_weights()


class _CoarseSubNet(nn.Module):
    """Small, half-resolution stage-1 network for the coarse-to-fine cascade.

    Deliberately shallow and plain (no SE/CBAM): its only job is a fast,
    rough localisation of the tumour, not the final boundary, so it's kept
    cheap. Input is average-pooled by 2x first (global, smoothed context —
    this also structurally suppresses the kind of thin, high-elongation
    structures, e.g. contrast-enhanced vessels, that showed up as ET false
    positives in the post-processing pass, since they get blurred out at
    half resolution), then a 2-level U-Net produces ET/TC/WT logits, which
    are upsampled back to the input's own full resolution so they line up
    with `x` and with the training targets.
    """

    def __init__(self, inplanes, num_classes, width, norm_layer=None, dropout=0):
        super(_CoarseSubNet, self).__init__()
        f0 = width
        f1 = width * 2

        self.pool_in = nn.AvgPool3d(2, 2)  # x -> x/2 (cheap, smoothed global context)
        self.downsample = nn.MaxPool3d(2, 2)  # x/2 -> x/4
        self.upsample = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)

        self.encoder1 = UBlock(inplanes, f0, f0, norm_layer, dropout=dropout)
        self.encoder2 = UBlock(f0, f1, f1, norm_layer, dropout=dropout)
        self.bottom = UBlock(f1, f1, f1, norm_layer, dropout=dropout)
        self.decoder1 = UBlock(f0 + f1, f0, f0, norm_layer, dropout=dropout)

        self.outconv = conv1x1(f0, num_classes)
        self.out_upsample = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)

    def forward(self, x):
        x_small = self.pool_in(x)  # D/2

        enc1 = self.encoder1(x_small)  # D/2
        enc2 = self.downsample(enc1)  # D/4
        enc2 = self.encoder2(enc2)  # D/4

        bottom = self.bottom(enc2)  # D/4

        up1 = self.upsample(bottom)  # D/2
        dec1 = self.decoder1(torch.cat([enc1, up1], dim=1))  # D/2

        out_small = self.outconv(dec1)  # D/2, num_classes channels
        coarse_out = self.out_upsample(out_small)  # D, num_classes channels (back to x's resolution)
        return coarse_out


class CascadedSEUnet(nn.Module):
    """Coarse-to-fine cascade with a Squeeze-and-Excitation decoder.

    Stage 1 (coarse, `_CoarseSubNet`): a cheap, half-resolution look at the
    whole volume that produces a rough ET/TC/WT probability map — a first
    guess of *where* the tumour is before stage 2 has to work out its exact
    boundary.

    Stage 2 (fine, EquiUnet-style body): full-resolution encoder/decoder.
      - Input = the original modalities concatenated with the coarse stage's
        sigmoid probability map, so stage 2 is *conditioned* on stage 1
        rather than cropped to it. No bounding-box/ROI-crop logic is needed
        anywhere, so shapes stay static and batching is unaffected — the
        rest of train.py/inference.py needs no changes.
      - Every decoder UBlock (and bottom_2) is SE-gated, exactly like
        EquiUnetSE above.
      - Just before the output conv, the last decoder feature map is
        *softly* re-weighted by the coarse whole-tumour probability:
        `feat * (1 + coarse_wt_prob)`. This is additive/residual on purpose,
        not a hard mask: during the last post-processing pass we saw a real
        recall miss (BraTS20_Training_141 — real ET present, model predicted
        none). A hard crop/mask driven by a wrong or empty coarse prediction
        would guarantee stage 2 also predicts nothing there; the residual
        gate can only amplify stage 2's own signal, and degrades gracefully
        to plain EquiUnetSE behaviour if stage 1 is wrong.

    Deep supervision, when enabled, supervises the usual EquiUnet decoder
    heads *and* the upsampled coarse-stage output (added as an extra entry
    in `deeps`), so stage 1 gets its own ET/TC/WT Dice gradient instead of
    learning only indirectly through stage 2.

    Signature-compatible with the other architectures in this file
    (`model_maker(4, 3, width=..., deep_supervision=..., norm_layer=...,
    dropout=...)` in train.py), so it's selectable as-is via `--arch
    CascadedSEUnet` with no other code changes.
    """
    name = "CascadedSEUnet"
    # Overridden by CascadedSEUnetAtt below to get CBAM-attention encoder stages
    # (mirroring how Att_EquiUnet swaps UBlockCbam in for UBlock) without
    # duplicating the whole class body.
    _encoder_block = UBlock

    def __init__(self, inplanes, num_classes, width, norm_layer=None, deep_supervision=False, dropout=0,
                 coarse_width=None, se_reduction=8, **kwargs):
        super(CascadedSEUnet, self).__init__()
        self.deep_supervision = deep_supervision
        self.num_classes = num_classes

        # ---- Stage 1: coarse ----
        coarse_width = coarse_width or max(width // 2, 8)
        self.coarse_net = _CoarseSubNet(inplanes, num_classes, coarse_width, norm_layer, dropout=dropout)

        # ---- Stage 2: fine, EquiUnet-style body with an SE-gated decoder ----
        features = [width * 2 ** i for i in range(4)]
        fine_inplanes = inplanes + num_classes  # original modalities + coarse guidance map
        print(features)

        eb = self._encoder_block
        self.encoder1 = eb(fine_inplanes, features[0], features[0], norm_layer, dropout=dropout)
        self.encoder2 = eb(features[0], features[1], features[1], norm_layer, dropout=dropout)
        self.encoder3 = eb(features[1], features[2], features[2], norm_layer, dropout=dropout)
        self.encoder4 = eb(features[2], features[3], features[3], norm_layer, dropout=dropout)

        self.bottom = eb(features[3], features[3], features[3], norm_layer, (2, 2), dropout=dropout)

        self.bottom_2 = nn.Sequential(
            ConvBnRelu(features[3] * 2, features[2], norm_layer, dropout=dropout),
            SELayer3D(features[2], reduction_ratio=se_reduction),
        )

        self.downsample = nn.MaxPool3d(2, 2)

        self.decoder3 = UBlockSE(features[2] * 2, features[2], features[1], norm_layer, dropout=dropout,
                                  se_reduction=se_reduction)
        self.decoder2 = UBlockSE(features[1] * 2, features[1], features[0], norm_layer, dropout=dropout,
                                  se_reduction=se_reduction)
        self.decoder1 = UBlockSE(features[0] * 2, features[0], features[0], norm_layer, dropout=dropout,
                                  se_reduction=se_reduction)

        self.upsample = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)

        self.outconv = conv1x1(features[0], num_classes)

        if self.deep_supervision:
            self.deep_bottom = nn.Sequential(
                conv1x1(features[3], num_classes),
                nn.Upsample(scale_factor=8, mode="trilinear", align_corners=True))

            self.deep_bottom2 = nn.Sequential(
                conv1x1(features[2], num_classes),
                nn.Upsample(scale_factor=8, mode="trilinear", align_corners=True))

            self.deep3 = nn.Sequential(
                conv1x1(features[1], num_classes),
                nn.Upsample(scale_factor=4, mode="trilinear", align_corners=True))

            self.deep2 = nn.Sequential(
                conv1x1(features[0], num_classes),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm, nn.InstanceNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # ---- Stage 1: coarse localisation (full-res logits, upsampled back) ----
        coarse_out = self.coarse_net(x)  # (B, num_classes, D, H, W)
        coarse_probs = torch.sigmoid(coarse_out)
        coarse_wt_prob = coarse_probs[:, self.num_classes - 1:self.num_classes]  # WT channel (last label)

        # ---- Stage 2: fine segmentation, conditioned on stage 1's guess ----
        fine_in = torch.cat([x, coarse_probs], dim=1)

        down1 = self.encoder1(fine_in)
        down2 = self.downsample(down1)
        down2 = self.encoder2(down2)
        down3 = self.downsample(down2)
        down3 = self.encoder3(down3)
        down4 = self.downsample(down3)
        down4 = self.encoder4(down4)

        bottom = self.bottom(down4)
        bottom_2 = self.bottom_2(torch.cat([down4, bottom], dim=1))

        up3 = self.upsample(bottom_2)
        up3 = self.decoder3(torch.cat([down3, up3], dim=1))
        up2 = self.upsample(up3)
        up2 = self.decoder2(torch.cat([down2, up2], dim=1))
        up1 = self.upsample(up2)
        up1 = self.decoder1(torch.cat([down1, up1], dim=1))

        # Soft coarse-to-fine gate: amplify where stage 1 believes there is
        # whole tumour, never suppress to zero (see class docstring).
        up1 = up1 * (1 + coarse_wt_prob)

        out = self.outconv(up1)

        if self.deep_supervision:
            deeps = []
            for seg, deep in zip(
                    [bottom, bottom_2, up3, up2],
                    [self.deep_bottom, self.deep_bottom2, self.deep3, self.deep2]):
                deeps.append(deep(seg))
            deeps.append(coarse_out)  # supervise stage 1 directly too
            return out, deeps

        return out


class CascadedSEUnetAtt(CascadedSEUnet):
    """CascadedSEUnet with a CBAM-attention encoder.

    Mirrors how Att_EquiUnet swaps UBlockCbam in for UBlock on top of the plain
    EquiUnet body: same coarse-to-fine cascade and SE-gated decoder as
    CascadedSEUnet, but encoder1..4 and the dilated `bottom` stage use
    UBlockCbam (channel + spatial attention) instead of plain UBlock, exactly
    like the reference paper's "3D attention U-net version" of its own U-Net.
    This is the third of the three per-fold Pipeline-A variants: (1) this
    attention encoder, (2) plain CascadedSEUnet on the unfiltered dataset, and
    (3) plain CascadedSEUnet on a filtered/cleaned dataset (see
    dataset/brats.py's `exclude_ids` and the patient-filtering helper script).
    """
    name = "CascadedSEUnetAtt"
    _encoder_block = UBlockCbam