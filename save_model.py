import os
import logging

import torch
import mlflow
from mlflow.utils import logging_utils

from config import MLFLOW_URI, MLFLOW_MODEL, STATE_DICT, DEVICE

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%d.%m %H:%M:%S')
log = logging.getLogger()
logging_utils.disable_logging()  # MLflow throws some excessive logging warnings

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment('dem_cnn')
mlflow.enable_system_metrics_logging()


def main():
    log.info('Loading model...')

    os.makedirs('weights', exist_ok=True)
    model = mlflow.pytorch.load_model(MLFLOW_MODEL, map_location=DEVICE)
    if STATE_DICT:
        torch.save(model.state_dict(), 'weights/dem_cnn_sd.pt')
        log.info(f'Model state dictionary saved to {os.path.abspath("weights/dem_cnn_sd.pt")}')
    else:
        torch.save(model, 'weights/dem_cnn.pt')
        log.info(f'Model saved to {os.path.abspath("weights/dem_cnn.pt")}')


if __name__ == '__main__':
    main()
