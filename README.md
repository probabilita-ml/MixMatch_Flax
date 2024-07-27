[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

This repository contains a JAX-based implementation of the semi-supervised learning method *MixMatch* presented in the 2019 NeuRIPS paper: ["MixMatch: A Holistic Approach to Semi-Supervised Learning"](https://papers.nips.cc/paper_files/paper/2019/hash/1cd138d0499a68f4bb72bee04bbec2d7-Abstract.html).

The implementation in this repository is inspired from many PyTorch-based implementations available on Github and a Jax-based implementation [Missel](https://github.com/h-terao/Missel). The reason to re-implement in Jax although there is a Jax-based implementation is because [Missel](https://github.com/h-terao/Missel) have the following constraints:
 - it depends on a third-party libraries written by the same author of [Missel](https://github.com/h-terao/Missel), and
 - the current implementation is not jit-table due to the usage of Python `list` instead of Jax's `Array`.

The implementation in this repository tries to be as close to standard practice, such as data-loading and common models (e.g., ResNet) as possible. Thus, some common libraries are required as follows:
 - `mlx-data` from Apple to load data. Why is `mlx-data`? Because it is light-weight and easy to use.
 - `transformers[flax]` from HuggingFace to load models (e.g., ResNets). One advantage of this library is that `bfloat16` can be used with ease.
 - `mlflow` to track and manage experiments. It is free and easy to setup compared to other tools. In addition, it does not require to obtain a unique id to use.

> NOTE: Jax is installed in a specific way. Please follow the instructions at [jax.readthedocs.io](https://jax.readthedocs.io/en/latest/installation.html) to install Jax correctly. Other packages can be referred in `requirements.txt`.

## Data

The current implementation is for vision datasets. If another type of data is used, please modified the corresponding data loading in `utils.py`.

To make the data-loading modular, it is designed not to follow "folder structure of data", but rely on `json` files. The input files used to run the implementation has the following format:
```{json}
[
    {
        "file": "train_images/train_image_1.png",
        "label": 1
    },
    {
        "file": "train_images/train_image_2.png",
        "label": 5
    },
    {
        "file": "train_images/train_image_3.png",
        "label": 1
    }
]
```

> NOTE: the above file uses *relative* file path. In that case, it is important to set the flag `--ds-root` in the bash-shell file to point to the directory storing the dataset. For example: `--ds-root "/sda2/datasets/stl_10"` for the STL-10 dataset.

## Up and running

The implementation uses `mlflow` to track and manage experiments. To run the implementation, please do the following two steps:

 1. Open a terminal and run `mlflow_server.sh`:
```{bash}
bash mlflow_server.sh
```
 2. Open another terminal and run the experiment of interest:
```{bash}
bash run.sh
```

## An example of an experiment on STL-10 dataset

Belows are the training loss and the testing accuracy on STL-10 dataset.

![training loss](/img/loss.png)

![accuracy](/img/accuracy.png)