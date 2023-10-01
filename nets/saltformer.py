from typing import Tuple, List, Union,Iterable,Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import modules as md
from segmentation_models_pytorch.base import initialization as init
from segmentation_models_pytorch.base import (
    SegmentationModel,
    SegmentationHead,
    ClassificationHead,
)

class RegressionHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        conv2d = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=kernel_size // 2
        )
        bn=nn.BatchNorm2d(out_channels)
        activation = md.Activation("tanh")
        super().__init__(conv2d, bn, activation)

class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        use_batchnorm=True,
        attention_type=None,
    ):
        super().__init__()
        self.conv1 = md.Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = md.Attention(attention_type, in_channels=in_channels + skip_channels)
        self.conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = md.Attention(attention_type, in_channels=out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x

class CenterBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        conv1 = md.Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        super().__init__(conv1, conv2)

class UnetDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        decoder_channels,
        n_blocks=5,
        use_batchnorm=True,
        attention_type=None,
        center=False,
    ):
        super().__init__()
        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[1:]
        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        if center:
            self.center = CenterBlock(head_channels, head_channels, use_batchnorm=use_batchnorm)
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, *features):

        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        x = self.center(head)
        res = [x]
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            res.append(decoder_block(res[i], skip))
        return res

def upsample(size=None, scale_factor=None):
    return nn.Upsample(size=size, scale_factor=scale_factor, mode='bilinear', align_corners=False)

class SaltFormer(SegmentationModel):
    def __init__(
        self,
        encoder_name: str = "mit_b5",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 2,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = {"classes":2},
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            attention_type=decoder_attention_type,
        )
        self.fuse_image = nn.Sequential(
            nn.Linear(self.encoder.out_channels[-1], 32),
            nn.ReLU(inplace=True)
        )
        self.logit_image = nn.Sequential(
            nn.Linear(32, 1)
        )
        #deep supervision
        num_filters=32
        self.logit_pixel5 = nn.Sequential(
            nn.Dropout2d(p=0.5, inplace=True),
            nn.Conv2d(decoder_channels[0], num_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.Conv2d(num_filters, 1, kernel_size=1),
        )
        self.logit_pixel4 = nn.Sequential(
            nn.Dropout2d(p=0.5, inplace=True),
            nn.Conv2d(decoder_channels[1], num_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.Conv2d(num_filters, 1, kernel_size=1),
        )
        self.logit_pixel3 = nn.Sequential(
            nn.Dropout2d(p=0.5, inplace=True),
            nn.Conv2d(decoder_channels[2], num_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.Conv2d(num_filters, 1, kernel_size=1),
        )
        self.logit_pixel2 = nn.Sequential(
            nn.Dropout2d(p=0.5, inplace=True),
            nn.Conv2d(decoder_channels[3], num_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.Conv2d(num_filters, 1, kernel_size=1),
        )
        self.logit_pixel1 = nn.Sequential(
            nn.Dropout2d(p=0.5, inplace=True),
            nn.Conv2d(decoder_channels[4], num_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.Conv2d(num_filters, 1, kernel_size=1),
        )
        
        #salt dome
        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )
        #edge
        self.classification_head =  SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )
        #sdm
        self.regression_head = RegressionHead(
            in_channels=decoder_channels[-1],
            out_channels=1
        )

        self.name = "SaltFormer-{}".format(encoder_name)
        self.initialize()
    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)
        init.initialize_head(self.classification_head)
        init.initialize_head(self.regression_head)

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        batch_size, _, _, _ = x.shape
        self.check_input_shape(x)
        features = self.encoder(x)
        decoder_output = self.decoder(*features)
        #deep supervision
        ds_features = decoder_output[::-1]
        d1 = ds_features[0]
        d1_size = d1.size()[2:]
        upsampler = nn.Upsample(size=d1_size, mode='bilinear', align_corners=False)
        u5 = upsampler(ds_features[4])
        u4 = upsampler(ds_features[3])
        u3 = upsampler(ds_features[2])
        u2 = upsampler(ds_features[1])
        d = torch.cat((d1, u2, u3, u4, u5), 1)

        e = F.adaptive_avg_pool2d(features[-1], output_size=1).view(batch_size, -1)  # image pool
        e = F.dropout(e, p=0.50, training=self.training)
        fuse_image = self.fuse_image(e)
        logit_image = self.logit_image(fuse_image).view(-1)

        logit_pixel = (
            self.logit_pixel1(d1), self.logit_pixel2(u2), self.logit_pixel3(u3), self.logit_pixel4(u4),
            self.logit_pixel5(u5),
        )
        #mtl
        masks = self.segmentation_head(decoder_output[-1])
        edge = self.classification_head(decoder_output[-1])
        sdm = self.regression_head(decoder_output[-1])

        return masks, edge, sdm, logit_pixel, logit_image

    @torch.no_grad()
    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`
        """
        if self.training:
            self.eval()
        x = self.forward(x)
        return x

if __name__ == '__main__':
    saltFormer = SaltFormer('resnet34')

    input = torch.randn((32,3,128,128))
    output = saltFormer(input)
    for i in output:
        print(type(i))