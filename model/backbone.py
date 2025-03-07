from torch import nn, Tensor
from torchvision.models import \
    efficientnet_v2_l, EfficientNet_V2_L_Weights, \
    efficientnet_v2_m, EfficientNet_V2_M_Weights, \
    efficientnet_v2_s, EfficientNet_V2_S_Weights
from torchvision.ops import FeaturePyramidNetwork
from torchvision.models.detection.backbone_utils import IntermediateLayerGetter, LastLevelMaxPool


class EfficientNetV2FPNBackbone(nn.Module):

    def __init__(self, variant: str, pretrained: bool, last_level_max_pool: bool):
        super().__init__()

        if variant == 'large':
            weights = EfficientNet_V2_L_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = efficientnet_v2_l(weights=weights).features
        elif variant == 'medium':
            weights = EfficientNet_V2_M_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = efficientnet_v2_m(weights=weights).features
        else:
            weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = efficientnet_v2_s(weights=weights).features
        backbone[0][0] = nn.Conv2d(1, 32 if variant == 'large' else 24, 3, 2, 1, bias=False)

        stage_indices = [i for i, layer in enumerate(backbone) if isinstance(layer, nn.Sequential)]

        returned_layers = sorted([stage_indices[i] for i in range(len(stage_indices))
                                  if i % 2 == (1 if variant == 'small' else 0)])[3:]
        return_layers = {str(stage_indices[k]): str(v) for v, k in enumerate(returned_layers)}

        in_channels_list = [backbone[1][0].out_channels] \
            + [backbone[stage_indices[i]][-1].out_channels for i in stage_indices[1:-1]] \
            + [backbone[-1][0].out_channels]
        in_channels_list = [in_channels_list[i] for i in returned_layers]

        if last_level_max_pool:
            extra_blocks = LastLevelMaxPool()
            self.channel_sizes = in_channels_list + [in_channels_list[-1]]  # max pooling leaves channels as is
        else:
            extra_blocks = None
            self.channel_sizes = in_channels_list

        self.featmap_names = list(return_layers.values())
        self.out_channels = in_channels_list[-1]

        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=in_channels_list[-1],
            extra_blocks=extra_blocks,
            norm_layer=nn.BatchNorm2d
        )

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """
        Forward pass through the backbone. Retrieve feature maps from all convolutional blocks through
        IntermediateLayerGetter. Pass them through FPN to generate feature maps pyramid for all blocks (and extra one if
        LastLevelMaxPool is used).
        :param x: input tensor
        :return: pyramid as dictionary of named feature maps
        """
        x = self.body(x)
        x = self.fpn(x)
        return x


class EfficientNetV2Backbone(nn.Module):

    def __init__(self, variant: str = 'large', pretrained: bool = True):
        super().__init__()

        if variant == 'large':
            weights = EfficientNet_V2_L_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = efficientnet_v2_l(weights=weights).features
        elif variant == 'medium':
            weights = EfficientNet_V2_M_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = efficientnet_v2_m(weights=weights).features
        else:
            weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = efficientnet_v2_s(weights=weights).features
        self.backbone[0][0] = nn.Conv2d(1, 32 if variant == 'large' else 24, 3, 2, 1, bias=False)

        self.featmap_names = ['0']  # default value for single output feature map
        self.out_channels = self.backbone[-1][0].out_channels

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the backbone. Retrieve feature maps from final convolutional block only.
        :param x: input tensor
        :return: feature map
        """
        return self.backbone(x)
