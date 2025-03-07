from torch import Tensor
from torchvision.models.detection import FasterRCNN
import numpy as np

from model.backbone import EfficientNetV2FPNBackbone, EfficientNetV2Backbone
from model.head import FasterRCNNHead


def create_model(num_classes: int, fpn: bool, variant: str, pretrained: bool,
                 last_level_max_pool: bool, normalize: bool) -> FasterRCNN:
    """
    Create a backbone, either with FPN or without it depending on the arguments, and then pass it to the head.
    The result is a customized Faster R-CNN model (reconfigured RPN, generalized R-CNN transform and else).
    :param num_classes: amount of classes to predict
    :param fpn: whether to use Feature Pyramid Network or not
    :param variant: variant of the EfficientNet model (large, medium or small)
    :param pretrained: whether to use pre-trained weights for backbone or not
    :param last_level_max_pool: whether to use max pooling on the last level of FPN or not
    :param normalize: whether to normalize the input tensor or not
    :return: customized Faster R-CNN model
    """
    backbone = (EfficientNetV2FPNBackbone(variant, pretrained, last_level_max_pool)
                if fpn else EfficientNetV2Backbone(variant, pretrained))

    model = FasterRCNNHead(backbone, num_classes, fpn, normalize)

    return model


class LossAverager:
    def __init__(self):
        self._sum = 0.
        self._count = 0

    def update(self, loss: Tensor):
        """
        Update the loss value to be averaged later on.
        :param loss: loss value
        """
        self._sum += loss
        self._count += 1

    def compute(self) -> float:
        """
        Compute the average loss value. If no updates were made, return NaN.
        :return: average loss value
        """
        return self._sum / self._count if self._count > 0 else np.nan
