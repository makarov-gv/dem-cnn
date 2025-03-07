from torch import nn, Tensor
from torchvision.models.detection.faster_rcnn import FasterRCNN, FastRCNNConvFCHead
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import MultiScaleRoIAlign

from model.backbone import EfficientNetV2FPNBackbone, EfficientNetV2Backbone


class FasterRCNNHead(nn.Module):
    DEFAULT_ANCHOR_SIZES = ((8, 16, 32, 64, 128, 256, 512),)
    DEFAULT_ASPECT_RATIOS = ((0.25, 0.5, 1., 2., 4.),)

    def __init__(self, backbone: EfficientNetV2FPNBackbone | EfficientNetV2Backbone,
                 num_classes: int, fpn: bool, normalize: bool):
        super().__init__()

        anchor_sizes = self.DEFAULT_ANCHOR_SIZES
        aspect_ratios = self.DEFAULT_ASPECT_RATIOS
        if fpn:
            anchor_sizes *= len(backbone.channel_sizes)
            aspect_ratios *= len(anchor_sizes)

        rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
        rpn_head = RPNHead(backbone.out_channels, rpn_anchor_generator.num_anchors_per_location()[0], 2)

        box_roi_pooler = MultiScaleRoIAlign(backbone.featmap_names, 7, 2)
        box_head = FastRCNNConvFCHead((backbone.out_channels, 7, 7), [256, 256, 256, 256], [1024], nn.BatchNorm2d)
        self.model = FasterRCNN(
            backbone,
            num_classes,
            rpn_anchor_generator=rpn_anchor_generator,
            rpn_head=rpn_head,
            box_roi_pool=box_roi_pooler,
            box_head=box_head
        )

        if not normalize:
            grcnn = GeneralizedRCNNTransform(int(0.9 * 640), int(1.1 * 1280), [0.], [1.])  # size is from 640 to 1280 px
            self.model.transform = grcnn  # to avoid tensor normalization

    def forward(self, *args) -> list[dict[str, Tensor]]:
        """
        Forward pass through the model to get detections based on DEM image. Expects input batch of images and,
        optionally, targets (for validation/testing).
        :param x: input batch of images and, optionally, targets
        :return: predictions
        """
        return self.model(*args)
