# DEM CNN

## Overview

DEM CNN is a modified Faster R-CNN model from conference proceedings: *Road Obstacles Detection on Digital Elevation Model*. This project includes dataset preparation utilities, model training/evaluation/saving scripts and an inference example.

## Installation

* Set up conda environment:

    ```bash
    conda env create -n dem_cnn -f environment.yml
    conda activate dem_cnn
    ```

* Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

Be aware that preinstalled CUDA is required to run the project on GPU
(instructions [here](https://developer.nvidia.com/cuda-downloads)).

## Usage

### Inference

Refer to `inference.ipynb` for an example of how to use DEM CNN for inference. It implements image+annotation preprocessing, inference, postprocessing and visualization partially using this repository's tools. Your model can be loaded from MLflow registry or from local file.

### Training and evaluation

Due to proprietary nature of original work's dataset, it cannot be provided for training the model. However, this repository holds `extract_data.ipynb` notebook that can be used to generate a dataset of your own using DEMs and segmentation masks with roads and annotated obstacles. Use `dataset_viewer.ipynb` to view generated files.

After obtaining a dataset, MLflow tracking server has to be started. If you do not have it in your server already, you can start it locally:

```bash
mlflow ui
```

MLflow will be used to store all artifacts and metrics, and it is only implemented option for experimenting.

To train and evaluate DEM CNN, follow these steps:

* Customize `config.py` to fit your needs, leave unknown parameters to you as is.

* Run training:

    ```bash
    python3 train.py
    ```

* After successful training, register model in MLflow manually. Set *test* tag to registered model in registry. This signifies that model is ready for evaluation and saving.

* Evaluate model:

    ```bash
    python3 evaluate.py
    ```

* If evaluation has proven that model is ready for deployment, save it:

    ```bash
    python3 save_model.py
    ```

It is generally preferred to save model's state dictionary so it can be loaded later in your project without preserving
original project structure. If you saved model's state dictionary, getting `weights/dem_cnn_sd.pt` file as a result, use
**create_model** function from `model/tools.py` and **load_state_dict** method from PyTorch to load it, similar to how it is done in `inference.ipynb`.

## Results

Original pretrained DEM CNN model weights will not be distributed as it is used in proprietary software. The results, however, can be replicated and surpassed using this repository specifically for your data.

This model achieves global mAP: 42.02% and mAP@50: 67.52% on test dataset with various data of different quality. However, when provided with a dataset of DEMs with resolution of 2 cm/pixel, evaluation metrics are better - global mAP: 52.03%, mAP@50: 77.57%.

Parameters for training original model and predicting with it are by default in `config.py` (classes are also in there).

It is worth noting that DEM CNN usage is not limited to detecting only road obstacles. Model can be used to detect any objects on DEMs, as long as they are located on flat surfaces and are not noisy. Just rasterize regions of interest during inference with such flat surfaces and make predictions on them to get similar results.

## Citation

```bibtex
@inproceedings{makarov2025demcnn,
  author = {Georgy Makarov and Denis Gontar},
  title = {Road Obstacles Detection on Digital Elevation Model},
  year = {2025},
  # organization = {NOT PUBLISHED YET},
  # booktitle = {NOT PUBLISHED YET},
  # pages = {NOT PUBLISHED YET}
}
```
