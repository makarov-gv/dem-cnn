import os

import torch

MLFLOW_URI = 'http://localhost:5000'  # URI for MLflow tracking server
MLFLOW_MODEL = 'models:/dem_cnn@test'  # URI for MLflow latest test-suitable DEM CNN model

STATE_DICT = True  # save/load model's state dictionary (preferred), pickle/unpickle it otherwise
SCORE_THRESHOLD = 0.8  # minimum score (confidence) threshold value
IOU_THRESHOLD = 0.2  # intersection over union (IoU) threshold value

CLASSES_MAPPING = {
    0: '__background__',  # index 0 is reserved for background
    1: 'stone',  # single stone separate form others
    2: 'stones',  # group of stones close together
    3: 'convexity',  # small pile of sand, gravel, ore or else
    4: 'cavity',  # counterpart of pile - hole, well or crack
    5: 'special vehicle',  # excavators, bulldozers, tractors and else
    6: 'vehicle',  # conventional trucks, buses and cars
    7: 'toroid',  # toroidal objects like tyres, wells
    8: 'lying post',  # lying post-like objects like fallen tree log, pipes
    9: 'standing post',  # standing post-like objects like power lines, road signs
    10: 'box'  # box-like objects like containers, shacks
}

DATASET_DIR = os.path.abspath('dataset/')  # to load dataset generated by extract_data.ipynb
NUM_EPOCHS = 1000  # number of training epochs
LOG_STEP = 10  # number of steps between logging metrics
BATCH_SIZE = 2  # number of images and annotations per batch
IMAGE_SIZE = 640  # to resize input image so it fits input tensor shape (preferably from 640 to 1280)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = os.cpu_count() if os.cpu_count() else 0  # number of workers for dataloader
BEST_MODEL = True  # save best model according to sum of validation losses or not

BACKBONE = 'EfficientNet V2'  # model backbone for hyperparameter logging
FPN = True  # use FPN for backbone or not
VARIANT = 'large'  # 'large', 'medium' or 'small'
PRETRAINED = True  # use pretrained backbone weights for (ImageNet-1K) or not
LAST_LEVEL_MAX_POOL = True  # use extra max pooling layer or not (only with FPN)
NORMALIZE = False  # normalize input tensor or not

OPTIMIZER = 'SGD'  # optimizer for hyperparameter logging
MOMENTUM = 0.9  # momentum for optimizer
WEIGHT_DECAY = 5e-4  # weight decay for optimizer
NESTEROV = True  # enable Nesterov momentum or not
LEARNING_RATE = 1e-3  # learning rate for optimizer

LR_SCHEDULER = 'StepLR'  # learning rate scheduler for hyperparameter logging
STEP_SIZE = 25  # number of epochs between applying learning rate scheduler
GAMMA = 0.66  # decay rate for learning rate scheduler
