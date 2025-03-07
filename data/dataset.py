import os

import torch
from torch.utils.data import Dataset
from torchvision.tv_tensors import Image, BoundingBoxes
from torchvision.transforms import v2
import rasterio
import numpy as np


class DEMDataset(Dataset):
    def __init__(self, images_dir: os.path, annotations_dir: os.path, size: int, transforms: v2.Compose | None):
        self.images = os.listdir(images_dir)
        self.images = sorted([os.path.join(images_dir, image) for image in self.images])

        self.annotations = []
        for image in self.images:
            annotation = os.path.join(annotations_dir, os.path.splitext(os.path.basename(image))[0] + '.txt')
            self.annotations.append(annotation if os.path.exists(annotation) else '')

        self.size = size
        self.transforms = transforms

    def __getitem__(self, idx) -> tuple[Image, dict]:
        with rasterio.open(self.images[idx]) as src:
            image = src.read(out_shape=(self.size, self.size))
            image_nan = image.copy()
            image_nan[image == src.nodata] = np.nan

            image = image - np.nanmin(image_nan)
            image[np.isnan(image_nan)] = np.random.uniform(low=0., high=image.max(), size=np.isnan(image_nan).sum())
            image = Image(image)

        boxes = torch.empty((0, 4), dtype=torch.float32)
        labels = torch.empty(0, dtype=torch.int64)
        if self.annotations[idx]:
            with open(self.annotations[idx], encoding='utf-8') as f:
                annotations = f.readlines()

            boxes = torch.empty((len(annotations), 4), dtype=torch.float32)
            labels = torch.empty(len(annotations), dtype=torch.int64)

            for i, annotation in enumerate(annotations):
                label, x_min, y_min, x_max, y_max = map(float, annotation[:-2].split('\t'))
                x_min, x_max = map(lambda x: x * self.size / src.width, (x_min, x_max))
                y_min, y_max = map(lambda y: y * self.size / src.height, (y_min, y_max))

                boxes[i] = torch.as_tensor([x_min, y_min, x_max, y_max], dtype=torch.float32)
                labels[i] = torch.as_tensor(int(label))

        target = {'boxes': BoundingBoxes(boxes, format='XYXY', canvas_size=(self.size, self.size)), 'labels': labels}

        if self.transforms:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        return len(self.images)
