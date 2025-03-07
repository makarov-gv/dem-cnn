from torch import Tensor
from torchvision.tv_tensors import Image
from matplotlib import pyplot as plt


def plot(image: Image | Tensor, target: dict = None, mapping: dict = None,
         color: str = 'red', line_width: int = 2, text_size: int = 12):
    """
    Plot an *image* with given *target* dict, draw found bounding boxes as well as their labels on top of it. Use this
    function for visualizing inference results. If *target* is not provided, or it is empty, only plot the image.
    :param image: depth map as **torchvision** tensor
    :param target: optional dictionary with objects' information
    :param mapping: optional dictionary with classes mapping
    :param color: color of bounding boxes and text
    :param line_width: line width in pixels
    :param text_size: text size (height, for default font) in pixels
    """
    image_np = image.numpy()[0]

    fig, ax = plt.subplots()
    map_ = ax.imshow(image_np, cmap='viridis')  # yellow-purple colormap for better visualization
    fig.colorbar(map_)

    if target and target['boxes'].shape[1] != 0:
        for bbox, label, score in zip(target['boxes'], target['labels'], target['scores']):
            x_min, y_min, x_max, y_max = map(int, bbox)
            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(x_max, image_np.shape[1] - 1), min(y_max, image_np.shape[0] - 1)
            ax.add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                       fill=False, edgecolor=color, linewidth=line_width))

            text = f'{mapping[int(label)] if mapping else label} ({float(score):.4f})'
            text_pos = (x_min + line_width, max(0, y_min - text_size - line_width))
            ax.text(text_pos[0], text_pos[1], text, color=color, size=text_size)

    plt.show()
