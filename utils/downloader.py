import os
from tempfile import TemporaryDirectory
from urllib.parse import urlencode
import requests

import torch
from torch import Tensor
from torchvision.tv_tensors import Image, BoundingBoxes
import rasterio
import numpy as np

base_url = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?'  # Yandex Disk API URL


def get_demo(image_url: str, annotation_url: str, image_size: int) -> tuple[Image | Tensor, dict]:
    """
    Get demo data sample that consist of single image and annotation in the format as if it was given by *DEMDataset*.
    :param image_url: link to demo image from Yandex Disk
    :param annotation_url: link to demo annotation from Yandex Disk
    :param image_size: image size for resizing
    :return: demo data sample
    """
    with TemporaryDirectory() as tmp_dir:
        url = base_url + urlencode({'public_key': image_url})
        response = requests.get(url, timeout=10)
        download_url = response.json()['href']
        download_response = requests.get(download_url, timeout=10)

        image_path = os.path.join(tmp_dir, 'image.tif')
        with open(image_path, 'wb') as f:
            f.write(download_response.content)

        with rasterio.open(image_path) as src:
            image = src.read(out_shape=(image_size, image_size))
            image_nan = image.copy()
            image_nan[image_nan == src.nodata] = np.nan

            image = image - np.nanmin(image_nan)
            image[np.isnan(image_nan)] = np.random.uniform(low=0., high=image.max(), size=np.isnan(image_nan).sum())
            image = Image(image)

        url = base_url + urlencode({'public_key': annotation_url})
        response = requests.get(url, timeout=10)
        download_url = response.json()['href']
        download_response = requests.get(download_url, timeout=10)

        annotation_path = os.path.join(tmp_dir, 'annotation.txt')
        with open(annotation_path, 'wb') as f:
            f.write(download_response.content)

        with open(annotation_path, encoding='utf-8') as f:
            annotations = f.readlines()

        boxes = torch.empty((len(annotations), 4), dtype=torch.float32)
        labels = torch.empty(len(annotations), dtype=torch.int64)

        for i, annotation in enumerate(annotations):
            label, x_min, y_min, x_max, y_max = map(float, annotation[:-2].split('\t'))
            x_min, x_max = map(lambda x: x * image_size / src.width, (x_min, x_max))
            y_min, y_max = map(lambda y: y * image_size / src.height, (y_min, y_max))

            boxes[i] = torch.as_tensor([x_min, y_min, x_max, y_max], dtype=torch.float32)
            labels[i] = torch.as_tensor(int(label))

    target = {'boxes': BoundingBoxes(boxes, format='XYXY', canvas_size=(image_size, image_size)), 'labels': labels}

    return image, target
