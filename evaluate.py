import logging

import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN
from torchmetrics.detection import MeanAveragePrecision
from tqdm import tqdm
import mlflow
from mlflow.utils import logging_utils

from config import MLFLOW_URI, DATASET_DIR, MLFLOW_MODEL, NUM_WORKERS, LOG_STEP, DEVICE, BATCH_SIZE, IMAGE_SIZE
from data import get_dataloader

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%d.%m %H:%M:%S')
log = logging.getLogger()
logging_utils.disable_logging()  # MLflow throws some excessive logging warnings

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment('dem_cnn')


@torch.inference_mode()
def test(model: FasterRCNN, dataloader: DataLoader):
    """
    Run one loop of testing, iterating through every image in the testing dataloader. No model adjustments are performed
    during the testing. Calculate global and @50 Mean Average Precision for all images in the testing subset and log it.
    :param model: Faster R-CNN model
    :param dataloader: testing dataloader
    """
    map_metric = MeanAveragePrecision().to(DEVICE)

    progress_bar = tqdm(dataloader, total=len(dataloader), desc='mAP: ...%, mAP@50: ...%')
    for i, data in enumerate(progress_bar):
        images, targets = data
        images = [image.to(DEVICE) for image in images]
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        with torch.inference_mode():  # making sure all weights are unaffected
            preds = model(images)
            preds = [{k: v.to(DEVICE) for k, v in p.items()} for p in preds]

        map_metric.update(preds, targets)

        if i % LOG_STEP == 0:
            mlflow.log_metrics({
                'map': map_metric.compute()['map'].cpu().item() * 100.,
                'map_50': map_metric.compute()['map_50'].cpu().item() * 100.
            }, step=i)

        progress_bar.set_description(f'mAP: {map_metric.compute()["map"].cpu().item() * 100.:.2f}%, '
                                     f'mAP@50: {map_metric.compute()["map_50"].cpu().item() * 100.:.2f}%')


def main():
    log.info('Loading model...')

    model = mlflow.pytorch.load_model(MLFLOW_MODEL, map_location=DEVICE)
    model = model.to(DEVICE)
    model.eval()

    test_dataloader = get_dataloader('test', DATASET_DIR, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS)

    log.info('Evaluating...')

    test(model, test_dataloader)

    log.info('Finished')

    if DEVICE == torch.device('cuda'):
        torch.cuda.empty_cache()
        log.info('Emptied CUDA cache')


if __name__ == '__main__':
    main()
