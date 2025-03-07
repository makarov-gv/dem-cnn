from torchvision.transforms import v2


def get_transforms(image_size: int) -> v2.Compose:
    """
    Get **torchvision** transformations that will augment the dataset during its loading. This composure of different
    transformations may be adjusted in the future. There is no need to change dtype or transform image to tensor as it
    is done in the *DEMDataset* by default.
    :param image_size: image size for resizing
    :return: composed transformations
    """
    return v2.Compose([v2.RandomVerticalFlip(p=0.5), v2.RandomHorizontalFlip(p=0.5),
                       v2.RandomResize(int(0.9 * image_size), int(1.1 * image_size))])
