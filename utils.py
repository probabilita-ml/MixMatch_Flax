from ResNet import ResNet

import json
import random
import os
import logging
import numpy as np


import mlx.data as dx

import jax
import optax
import flax.linen as nn
from flax.training import train_state
from flax.core import FrozenDict
from chex import Array, PRNGKey


class TrainState(train_state.TrainState):
    batch_stats: FrozenDict


def get_dataset(dataset_file: str, root: str = None, resize: tuple[int, int] = None) -> dx._c.Buffer:
    """load a dataset from a JSON file

    Args:
        dataset_file: path to the JSON file

    Returns:
        dset: the dataset of interest
    """
    # load information from the JSON file
    with open(file=dataset_file, mode='r') as f:
        # load a list of dictionaries
        json_data = json.load(fp=f)

    data_dicts = []
    for sample in json_data:
        file_path = os.path.join(root, sample['file']) if root is not None else sample['file']

        data_dicts.append(dict(file=file_path.encode('ascii'), label=sample['label']))

    # load image dataset without batching nor shuffling
    dset = (
        dx.buffer_from_vector(data=data_dicts)
        .load_image(key='file', output_key='image')
    )

    if resize is not None:
        dset = dset.image_resize(key='image', w=resize[0], h=resize[1])

    return dset


def prepare_dataset(
    dataset: dx._c.Buffer,
    shuffle: bool,
    batch_size: int,
    prefetch_size: int,
    num_threads: int,
    mean: tuple[int, int, int] = None,
    std: tuple[int, int, int] = None,
    random_crop_size: tuple[int, int] = None,
    prob_random_h_flip: float = None
) -> dx._c.Buffer:
    """batch, shuffle and convert from uint8 to float32 to train

    Args:
        dataset:
        shuffle:
        batch_size:
        prefetch_size:
        num_threads:
        mean: the mean to normalised input samples (translation)
        std: the standard deviation to normalised input samples (inverse scaling)
    """
    if shuffle:
        dset = dataset.shuffle()
    else:
        dset = dataset

    # region DATA AUGMENTATION
    # randomly crop
    if random_crop_size is not None:
        dset = dset.image_random_crop(
            key='image',
            w=random_crop_size[0],
            h=random_crop_size[1]
        )
    
    # randomly horizontal-flip
    if prob_random_h_flip is not None:
        if prob_random_h_flip < 0 or prob_random_h_flip > 1:
            raise ValueError('Probability to randomly horizontal-flip must be in [0, 1]'
                             ', but provided with {:f}'.format(prob_random_h_flip))

        dset = dset.image_random_h_flip(key='image', prob=prob_random_h_flip)
    
    # normalisation
    if (mean is None) or (std is None):
        logging.info(
            msg='mean and std must not be None. Found one or both of them are None.'
        )

        mean = 0.
        std = 1.
    
    mean = np.array(object=mean, dtype=np.float32)
    std = np.array(object=std, dtype=np.float32)
        
    dset = dset.key_transform(
        key='image',
        func=lambda x: (x.astype('float32') / 255 + mean) / std
    )
    # endregion

    # batching, converting to stream and return
    dset = (
        dset
        .to_stream()
        .batch(batch_size=batch_size)
        .prefetch(prefetch_size=prefetch_size, num_threads=num_threads)
    )

    return dset


def initialise_model(
    model: nn.Module,
    sample: Array,
    num_training_samples: int,
    lr: float,
    batch_size: int,
    num_epochs: int,
    key: PRNGKey
) -> TrainState:
    """initialise the parameters and optimiser of a model

    Args:
        sample: a sample from the dataset

    Returns:
        state:
    """
    lr_schedule_fn = optax.cosine_decay_schedule(
        init_value=lr,
        decay_steps=max(num_epochs, 1_000) * (num_training_samples // batch_size + 1)
    )

    # pass dummy data to initialise model's parameters
    params = model.init(rngs=key, x=sample, train=False)

    # add L2 regularisation(aka weight decay)
    weight_decay = optax.masked(
        inner=optax.add_decayed_weights(
            weight_decay=0.0005,
            mask=None
        ),
        mask=lambda p: jax.tree_util.tree_map(lambda x: x.ndim != 1, p)
    )


    # define an optimizer
    tx = optax.chain(
        weight_decay,
        optax.add_noise(eta=0.01, gamma=0.55, seed=random.randint(a=0, b=1_000)),
        optax.sgd(learning_rate=lr_schedule_fn, momentum=0.9)
    )

    state = TrainState.create(
        apply_fn=model.apply,
        params=params['params'],
        batch_stats=params['batch_stats'],
        tx=tx
    )

    return state


def initialise_huggingface_resnet(
    model: ResNet,
    sample: Array,
    num_training_samples: int,
    lr: float,
    batch_size: int,
    num_epochs: int,
    key: PRNGKey
) -> TrainState:
    """initialise the parameters and optimiser of a model

    Args:
        sample: a sample from the dataset

    Returns:
        state:
    """
    lr_schedule_fn = optax.cosine_decay_schedule(
        init_value=lr,
        decay_steps=(num_epochs + 50) * (num_training_samples // batch_size)
    )

    # pass dummy data to initialise model's parameters
    # params = model.init(rngs=key, x=sample, train=False)
    params = model.init_weights(rng=key, input_shape=sample.shape)

    # add L2 regularisation(aka weight decay)
    weight_decay = optax.masked(
        inner=optax.add_decayed_weights(
            weight_decay=5e-4,
            mask=None
        ),
        mask=lambda p: jax.tree_util.tree_map(lambda x: x.ndim != 1, p)
    )


    # define an optimizer
    tx = optax.chain(
        weight_decay,
        optax.add_noise(eta=0.01, gamma=0.55, seed=random.randint(a=0, b=1_000)),
        optax.sgd(learning_rate=lr_schedule_fn, momentum=0.9)
        # optax.adam(learning_rate=lr),
        # optax.ema(decay=0.999)
    )

    state = TrainState.create(
        apply_fn=model.apply,
        params=params['params'],
        batch_stats=params['batch_stats'],
        tx=tx
    )

    return state


# Define a custom argument type for a list of integers
def list_of_ints(inputs: str) -> list[int]:
    # remove parentheses
    inputs = inputs.replace('(', '')
    inputs = inputs.replace(')', '')

    return tuple(map(int, inputs.split(',')))


def list_of_floats(inputs: str) -> list[float]:
    # remove parentheses
    inputs = inputs.replace('(', '')
    inputs = inputs.replace(')', '')

    return tuple(map(float, inputs.split(',')))


def list_of_strings(inputs: str) -> list[str]:
    # remove parentheses
    inputs = inputs.replace('(', '')
    inputs = inputs.replace(')', '')

    return tuple(inputs.split(','))