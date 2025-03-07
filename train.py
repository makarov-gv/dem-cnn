import os
import logging
from tempfile import TemporaryDirectory

import torch
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torchinfo import summary
from torchvision.models.detection import FasterRCNN
from tqdm import tqdm
import mlflow
from mlflow.utils import logging_utils
from mlflow.models import infer_signature, ModelSignature

from config import (MLFLOW_URI, DATASET_DIR, CLASSES_MAPPING, NUM_WORKERS, NUM_EPOCHS, LOG_STEP, DEVICE, BATCH_SIZE,
                    IMAGE_SIZE, LEARNING_RATE, MOMENTUM, NESTEROV, WEIGHT_DECAY, STEP_SIZE, GAMMA, BEST_MODEL, BACKBONE,
                    OPTIMIZER, LR_SCHEDULER, FPN, VARIANT, PRETRAINED, LAST_LEVEL_MAX_POOL, NORMALIZE)
from model import create_model, LossAverager
from data import get_dataloader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%d.%m %H:%M:%S'
)
log = logging.getLogger()
logging_utils.disable_logging()  # MLflow throws some excessive logging warnings

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment('dem_cnn')
mlflow.enable_system_metrics_logging()

code_paths = ['inference_demo.ipynb', 'config.py', 'utils', 'model']


def train(model: FasterRCNN, optimizer: SGD, scheduler: StepLR,
          dataloader: DataLoader, epoch: int, signature: ModelSignature) -> float:
    """
    Run one loop of training, iterating through every image in the training dataloader and adjusting the model
    according to the losses. Both optimizer and learning rate scheduler are active during the training. Save the model
    as **last** in MLflow artifacts, replacing the one from previous epoch.
    :param model: Faster R-CNN model
    :param optimizer: SGD optimizer object
    :param scheduler: learning rate scheduler (StepLR) object
    :param dataloader: training dataloader
    :param epoch: number of epoch (for logging)
    :param signature: MLflow model signature
    :return: average training loss
    """
    loss_averager = LossAverager()

    progress_bar = tqdm(dataloader, total=len(dataloader), desc='Training loss: ...')
    for i, data in enumerate(progress_bar):
        images, targets = data
        images = [image.to(DEVICE) for image in images]
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())
        loss_averager.update(float(loss.detach().cpu()))

        loss.backward()
        optimizer.step()

        optimizer.zero_grad()

        step = epoch * len(dataloader) + i
        if step % LOG_STEP == 0:
            mlflow.log_metrics({'train/' + k: v for k, v in loss_dict.items()}, step=step)
            mlflow.log_metric('train/loss', loss, step=step)  # sum of losses for ease of monitoring

        progress_bar.set_description(f'Training loss: {loss:.4f}')

    scheduler.step()

    mlflow.pytorch.log_model(model, 'last', signature=signature, code_paths=code_paths)

    return loss_averager.compute()


@torch.inference_mode()  # making sure all weights are unaffected
def validate(model: FasterRCNN, best_loss: torch.Tensor | None,
             dataloader: DataLoader, epoch: int, signature: ModelSignature) -> float:
    """
    Run one loop of validation, iterating through every image in the validation dataloader and logging the losses.
    No model adjustments are performed during the validation. If *BEST_MODEL* in config is set to True, check
    validation loss each *LOG_STEP* steps and save the model as **best** in MLflow artifacts, replacing worse one.
    :param model: Faster R-CNN model
    :param best_loss: unsqueezed best validation loss
    :param dataloader: validation dataloader
    :param epoch: number of epoch (for logging)
    :param signature: MLflow model signature
    :return: average validation loss
    """
    loss_averager = LossAverager()

    progress_bar = tqdm(dataloader, total=len(dataloader), desc='Validation loss: ...')
    for i, data in enumerate(progress_bar):
        images, targets = data
        images = [image.to(DEVICE) for image in images]
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())
        loss_averager.update(float(loss.detach().cpu()))

        step = epoch * len(dataloader) + i
        if step % LOG_STEP == 0:
            mlflow.log_metrics({'val/' + k: v for k, v in loss_dict.items()}, step=step)
            mlflow.log_metric('val/loss', loss, step=step)  # sum of losses for ease of monitoring

        progress_bar.set_description(f'Validation loss: {loss:.4f}')

    if best_loss and loss_averager.compute() < best_loss[0]:
        best_loss[0] = loss_averager.compute()
        mlflow.pytorch.log_model(model, 'best', signature=signature, code_paths=code_paths)

    return loss_averager.compute()


def main():
    log.info('Creating model...')

    model = create_model(len(CLASSES_MAPPING), FPN, VARIANT, PRETRAINED, LAST_LEVEL_MAX_POOL, NORMALIZE)
    model = model.to(DEVICE)
    model.eval()

    hyperparams = {
        'num_epochs': NUM_EPOCHS,
        'backbone': f'{BACKBONE} {VARIANT.capitalize()}',
        'optimizer': OPTIMIZER,
        'lr_scheduler': LR_SCHEDULER,
        'batch_size': BATCH_SIZE,
        'image_size': IMAGE_SIZE,
        'learning_rate': LEARNING_RATE,
        'momentum': MOMENTUM,
        'weight_decay': WEIGHT_DECAY,
        'nesterov': NESTEROV,
        'step_size': STEP_SIZE,
        'gamma': GAMMA,
        'fpn': FPN,
        'pretrained': PRETRAINED,
        'last_level_max_pool': LAST_LEVEL_MAX_POOL,
        'normalize': NORMALIZE,
        'num_classes': len(CLASSES_MAPPING)
    }
    mlflow.log_params(hyperparams)

    log.info(f'Using hyperparameters: {hyperparams}')

    with TemporaryDirectory() as tmp_dir:
        model_summary = os.path.join(tmp_dir, 'summary.txt')
        with open(model_summary, 'w', encoding='utf-8') as f:
            f.write(str(summary(model, verbose=0)))
        mlflow.log_artifact(model_summary)

    with torch.inference_mode():
        model_input = torch.rand(1, 1, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)
        model_output = model(model_input)[0]

        signature = infer_signature(
            model_input={'image': model_input.cpu().numpy()},
            model_output={key: model_output[key].cpu().numpy() for key in model_output.keys()}
        )
    model.train()

    best_loss = torch.tensor(torch.inf).unsqueeze(0).to(DEVICE) if BEST_MODEL else None  # mutable tensor

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = SGD(params, lr=LEARNING_RATE, momentum=MOMENTUM, nesterov=NESTEROV, weight_decay=WEIGHT_DECAY)
    scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    train_dataloader = get_dataloader('train', DATASET_DIR, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS)
    val_dataloader = get_dataloader('val', DATASET_DIR, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS)

    log.info('Training...')

    for epoch in range(NUM_EPOCHS):
        log.info(f'Epoch {epoch + 1}:')

        train_loss = train(model, optimizer, scheduler, train_dataloader, epoch, signature)
        val_loss = validate(model, best_loss, val_dataloader, epoch, signature)

        log.info(f'Average training loss: {train_loss:.4f}, validation loss: {val_loss:.4f}')

    log.info('Finished')

    if DEVICE == torch.device('cuda'):
        torch.cuda.empty_cache()
        log.info('Emptied CUDA cache')


if __name__ == '__main__':
    main()
