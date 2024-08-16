from ResNet import ResNet

import json
import random
import os
import logging
import numpy as np

from tqdm import tqdm

import mlx.data as dx

import jax
import optax
import flax.linen as nn
from flax.training import train_state
from flax.core import FrozenDict
from chex import Array, PRNGKey


class TrainState(train_state.TrainState):
    batch_stats: FrozenDict


def make_dataset(
    annotation_files: list[str],
    ground_truth_file: str,
    root: str = None,
    shape: tuple[int, int] = None
) -> dx._c.Buffer:
    """make the dataset from multiple annotation files.

    Each file may contain only a subset of the whole dataset.
    If one annotator does not label a sample, the label will be set to -1.

    Args:
        annotation_files: list of pathes to the json files of annotators
        ground_truth_file: path to the json file of ground truth
        root: the directory to the dataset folder
        shape: the new shape (width and height) of the resized image

    Returns:
        dataset:
    """
    if len(annotation_files) < 1:
        raise ValueError('len(annotation_files) <= 1, expecting len(annotation_files) > 1')

    # region DEFAULT VALUE OF MISSING ANNOTATION
    _missing_value: int = -1
    _is_one_hot: bool = False
    _num_class: int
    _missing_label: list[int] | int
    # endregion

    all_filenames = []
    annotators = []

    # load json data
    for annotation_file in tqdm(
        iterable=annotation_files,
        desc='Load annotations',
        ncols=80,
        colour='green',
        leave=False
    ):
        with open(file=annotation_file, mode='r') as f:
            # load a list of dictionaries
            annotations = json.load(fp=f)
        
        # initialise a dict: key=file, value=label for each annotator
        annotator = {}
        
        for annotation in annotations:
            file_path = annotation['file']

            if file_path not in all_filenames:
                all_filenames.append(file_path)
            
            annotator[file_path] = annotation['label']

            # check if one_hot or not
            if isinstance(annotation['label'], int):
                continue
            elif isinstance(annotation['label'], list):
                _is_one_hot = True
                _num_class = len(annotation['label'])
            else:
                raise ValueError(
                    'Unknown label format. Expect either integer or list of floats, '
                    f'but found {annotation["label"].__class__}'
                )
        
        annotators.append(annotator)
    
    # set missing labels
    if _is_one_hot:
        _missing_label = [_missing_value] * _num_class
    else:
        _missing_label = _missing_value

    with open(file=ground_truth_file, mode='r') as f:
        ground_truth_annotations = json.load(fp=f)

    ground_truth_annotations = {
        gt_annotation['file']: gt_annotation['label']
            for gt_annotation in ground_truth_annotations
    }

    # initialise a list to store all the json data from all the provided datasets
    consolidation = []

    for filename in tqdm(iterable=ground_truth_annotations, desc='make dataset', ncols=80, leave=False):
        file_path = os.path.join(root, filename) if root is not None else filename
        record = dict(file=file_path.encode('ascii'), label=[], ground_truth=_missing_label)

        for annotator in annotators:
            record['label'].append(annotator.get(filename, _missing_label))
        
        # set ground truth
        record['ground_truth'] = ground_truth_annotations.get(filename)

        consolidation.append(record)

    # load image dataset without batching nor shuffling
    dset = (
        dx.buffer_from_vector(data=consolidation)
        .load_image(key='file', output_key='image')
    )

    if shape is not None:
        dset = dset.image_resize(key='image', w=shape[0], h=shape[1])

    return dset


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
        optax.clip_by_global_norm(max_norm=10),
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
