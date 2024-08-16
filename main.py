# python standard libraries
import argparse
import os
from pathlib import Path
import random
from functools import partial

# third-party libraries
from tqdm import tqdm

import hydra
from omegaconf import DictConfig, OmegaConf
import flatdict

import jax
import jax.numpy as jnp

from flax.nnx import metrics
from flax.core import FrozenDict

import orbax.checkpoint as ocp

import optax

import dm_pix

from chex import Array, Scalar

import mlflow

import mlx.data as dx

# locally-imported
from utils import (
    TrainState,
    get_dataset,
    prepare_dataset,
    initialise_huggingface_resnet
)

# from ResNet import resnet18
from PreActResNet import ResNet18 as resnet18


def interleave(xy: list[Array]):
    nu = len(xy) - 1
    xy = [[x[::-1] for x in reversed(jnp.array_split(v[::-1], nu + 1))] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return jnp.array(object=[jnp.concatenate(v) for v in xy])


@partial(jax.jit, static_argnames=('crop_size', 'prob_random_h_flip'))
def augment_an_image(
    key1: jax.random.PRNGKey,
    key2: jax.random.PRNGKey,
    image: Array,
    crop_size: tuple[int, int, int],
    prob_random_h_flip: float
) -> Array:
    """perform data augmentation on a single image. For a batch of images,
    please apply jax.vmap

    Args:
        key1 and key2: random PRNG keys
        image: the image of interest

    Returns:
        out: the augmented image
    """
    x = dm_pix.pad_to_size(
        image=image,
        target_height=image.shape[0] + 8,
        target_width=image.shape[1] + 8
    )
    x = dm_pix.gaussian_blur(image=x, sigma=0.005, kernel_size=5)
    x = dm_pix.random_crop(key=key1, image=x, crop_sizes=crop_size)
    x = dm_pix.random_flip_left_right(key=key2, image=x, probability=prob_random_h_flip)

    return x


@partial(jax.jit, static_argnames=('crop_size', 'prob_random_h_flip'))
def augment_batch_images(
    keys1: jax.random.PRNGKey,
    keys2: jax.random.PRNGKey,
    images: Array,
    crop_size: tuple[int, int, int],
    prob_random_h_flip: float
) -> Array:
    """
    """
    augment_images_fn = jax.vmap(fun=augment_an_image, in_axes=(0, 0, 0, None, None))

    return augment_images_fn(keys1, keys2, images, crop_size, prob_random_h_flip)


@partial(jax.jit, static_argnames=('crop_size', 'prob_random_h_flip', 'num'))
def augment_images_function(
    images: Array,
    crop_size: tuple[int, int, int],
    prob_random_h_flip: float,
    prng_key: jax.random.PRNGKey,
    num: int = 1,
) -> Array:
    """perform data augmentation on a batch of images multiple times

    Args:
        images: a batch of images
        crop_size:
        prob_random_h_flip:
        num: the number of times to perform the data augmentation
        prng_key:

    Returns:
        x: the augmented images
    """
    num_images = len(images)
    keys = jax.random.split(key=prng_key, num=num)

    x = images + 0.
    for i in range(num):
        random_keys = jax.random.split(key=keys[i], num=2 * num_images)
        x = augment_batch_images(
            random_keys[:num_images],
            random_keys[num_images:],
            x,
            crop_size,
            prob_random_h_flip
        )
    
    return x


@partial(jax.jit, static_argnames=('lambda_u',), donate_argnames=('state',))
def train_step(
    mixed_inputs: Array,
    mixed_labels: Array,
    state: TrainState,
    lambda_u: float,
) -> tuple[TrainState, Scalar]:
    """
    """
    apply_batch = jax.vmap(fun=partial(state.apply_fn, mutable=['batch_stats']), in_axes=(None, 0, None))

    def loss_function(params: FrozenDict) -> tuple[Scalar, FrozenDict]:
        """
        """
        logits, batch_stats = apply_batch(
            {'params': params, 'batch_stats': state.batch_stats},
            mixed_inputs,
            True
        )
        batch_stats = jax.tree.map(f=lambda x: jnp.mean(a=x, axis=0), tree=batch_stats)
        logits = interleave(logits)

        logits_x, logits_y = logits[0], jnp.concatenate(arrays=logits[1:], axis=0)
        labels_x, labels_y = mixed_labels[:len(logits_x)], mixed_labels[len(logits_x):]

        sup_loss = optax.softmax_cross_entropy(
            logits=logits_x,
            labels=labels_x
        ).mean()
        unsup_loss = optax.squared_error(
            predictions=jax.nn.softmax(x=logits_y, axis=-1),
            targets=labels_y
        ).mean()

        loss = sup_loss + lambda_u * unsup_loss

        return loss, batch_stats

    grad_value_fn = jax.value_and_grad(fun=loss_function, argnums=0, has_aux=True)
    (loss, batch_stats), grads = grad_value_fn(state.params)

    # update parameters from gradients
    state = state.apply_gradients(grads=grads)

    # update batch statistics
    state = state.replace(batch_stats=batch_stats['batch_stats'])

    return state, loss


@jax.jit
def forward_step(x: Array, state: TrainState) -> tuple[Array, FrozenDict]:
    """perform the forward pass
    """
    logits, batch_stats = state.apply_fn(
        variables={'params': state.params, 'batch_stats': state.batch_stats},
        x=x,
        train=True,
        mutable=['batch_stats']
    )

    return logits, batch_stats


def train(
    dataset_labelled: dx._c.Buffer,
    dataset_unlabelled: dx._c.Buffer,
    state: TrainState,
    cfg: DictConfig
) -> tuple[TrainState, Scalar]:
    """the main training procedure
    """
    # batching and shuffling the dataset
    stream_labelled = prepare_dataset(
        dataset=dataset_labelled,
        shuffle=True,
        batch_size=cfg.training.batch_size,
        prefetch_size=cfg.training.prefetch_size,
        num_threads=cfg.training.num_threads,
        mean=cfg.dataset.mean,
        std=cfg.dataset.std,
        random_crop_size=None,
        prob_random_h_flip=None
    )
    stream_unlabelled = prepare_dataset(
        dataset=dataset_unlabelled,
        shuffle=True,
        batch_size=cfg.training.batch_size,
        prefetch_size=cfg.training.prefetch_size,
        num_threads=cfg.training.num_threads,
        mean=cfg.dataset.mean,
        std=cfg.dataset.std,
        random_crop_size=None,
        prob_random_h_flip=None
    )

    # metric to track the training loss
    loss_accum = metrics.Average()

    fold_in_batch = jax.vmap(fun=jax.random.fold_in, in_axes=(0, 0))
    crop_size = (*cfg.dataset.crop_size, 3)

    for unlabelled_samples in tqdm(
        iterable=stream_unlabelled,
        desc='train',
        total=len(dataset_unlabelled) // cfg.training.batch_size,
        ncols=80,
        leave=False,
        position=2,
        colour='blue',
        disable=not cfg.training.progress_bar
    ):
        if (len(unlabelled_samples['image']) < cfg.training.batch_size):
            break

        try:
            labelled_samples = next(stream_labelled)
            if (len(labelled_samples['image']) < cfg.training.batch_size):
                raise StopIteration
        except StopIteration:
            # reset the labelled data subset
            stream_labelled = prepare_dataset(
                dataset=dataset_labelled,
                shuffle=True,
                batch_size=cfg.training.batch_size,
                prefetch_size=cfg.training.prefetch_size,
                num_threads=cfg.training.num_threads,
                mean=cfg.dataset.mean,
                std=cfg.dataset.std,
                random_crop_size=None,
                prob_random_h_flip=None
            )
            labelled_samples = next(stream_labelled)
        
        labelled_inputs = jnp.asarray(a=labelled_samples['image'], dtype=jnp.float32)
        y = jnp.asarray(a=labelled_samples['label'], dtype=jnp.int32)
        unlabelled_inputs = jnp.asarray(a=unlabelled_samples['image'], dtype=jnp.float32)

        # region RANDOM KEYS for batch data augmentation
        key0, key1, key2 = jax.random.split(
            key=jax.random.key(seed=random.randint(a=0, b=100_000)),
            num=3
        )
        keys0 = jax.random.split(key=key0, num=len(y))
        keys1 = jax.random.split(key=key1, num=len(y))
        keys2 = jax.random.split(key=key2, num=len(y))

        # fold data into the key to further randomise
        keys01 = fold_in_batch(keys0, y)
        keys11 = fold_in_batch(keys1, y)
        keys21 = fold_in_batch(keys2, y)
        # endregion

        x = augment_batch_images(
            keys1=keys0,
            keys2=keys01,
            images=labelled_inputs,
            crop_size=crop_size,
            prob_random_h_flip=cfg.training.prob_random_h_flip
        )
        u1 = augment_batch_images(
            keys1=keys1,
            keys2=keys11,
            images=unlabelled_inputs,
            crop_size=crop_size,
            prob_random_h_flip=cfg.training.prob_random_h_flip
        )
        u1 = augment_batch_images(
            keys1=keys11,
            keys2=keys1,
            images=u1,
            crop_size=crop_size,
            prob_random_h_flip=cfg.training.prob_random_h_flip
        )
        u2 = augment_batch_images(
            keys1=keys2,
            keys2=keys21,
            images=unlabelled_inputs,
            crop_size=crop_size,
            prob_random_h_flip=cfg.training.prob_random_h_flip
        )
        u2 = augment_batch_images(
            keys1=keys21,
            keys2=keys2,
            images=u2,
            crop_size=crop_size,
            prob_random_h_flip=cfg.training.prob_random_h_flip
        )
        lx = optax.smooth_labels(
            labels=jax.nn.one_hot(x=y, num_classes=cfg.dataset.num_classes),
            alpha=0.01
        )

        # guess labels
        logits_u1, _ = forward_step(x=u1, state=state)
        logits_u2, _ = forward_step(x=u2, state=state)

        # average
        lu1 = jax.nn.softmax(x=logits_u1, axis=-1)
        lu2 = jax.nn.softmax(x=logits_u2, axis=-1)
        lu = 0.5 * (lu1 + lu2)

        # sharpening
        lu = lu ** (1 / cfg.training.sharpen_factor)
        lu = lu / jnp.sum(a=lu, axis=-1, keepdims=True)  # normalised

        # region MIXUP
        inputs = jnp.concatenate(arrays=[x, u1, u2], axis=0)
        labels = jnp.concatenate(arrays=[lx, lu, lu], axis=0)

        index_rng, mixup_ratio_rng = jax.random.split(
            key=jax.random.key(seed=random.randint(a=0, b=100_000))
        )
        index = jax.random.permutation(key=index_rng, x=len(inputs))
        lam = jax.random.beta(
            key=mixup_ratio_rng,
            a=cfg.training.alpha,
            b=cfg.training.alpha
        )
        lam = jnp.maximum(lam, 1 - lam)
        mixed_inputs = lam * inputs + (1 - lam) * inputs[index]
        mixed_labels = lam * labels + (1 - lam) * labels[index]

        mixed_inputs = jnp.array_split(mixed_inputs, len(inputs) // len(x))
        mixed_inputs = interleave(mixed_inputs)
        # mixed_inputs = jnp.array(object=mixed_inputs, dtype=jnp.float32)

        state, loss = train_step(
            mixed_inputs=mixed_inputs,
            mixed_labels=mixed_labels,
            state=state,
            lambda_u=cfg.training.lambda_u
        )

        if jnp.isnan(loss):
            raise ValueError('Training loss is NaN.')

        # tracking
        loss_accum.update(values=loss)

    return state, loss_accum.compute()


@jax.jit
def prediction_step(x: Array, state: TrainState) -> Array:
    logits, _ = state.apply_fn(
        variables={'params': state.params, 'batch_stats': state.batch_stats},
        x=x,
        train=False,
        mutable=['batch_stats']
    )
    
    return logits


def evaluate(dataset: dx._c.Buffer, state: TrainState, cfg: DictConfig) -> Scalar:
    """calculate the average cluster probability vector

    Args:
        dataset:
        state:
    """
    # prepare dataset for training
    dset = prepare_dataset(
        dataset=dataset,
        shuffle=True,
        batch_size=cfg.training.batch_size,
        prefetch_size=cfg.training.prefetch_size,
        num_threads=cfg.training.num_threads,
        mean=cfg.dataset.mean,
        std=cfg.dataset.std,
        random_crop_size=None,
        prob_random_h_flip=None
    )

    accuracy_accum = metrics.Accuracy()

    for samples in tqdm(
        iterable=dset,
        desc='evaluate',
        total=len(dataset) // cfg.training.batch_size,
        ncols=80,
        leave=False,
        position=2,
        colour='blue',
        disable=not cfg.training.progress_bar
    ):
        x = jnp.asarray(a=samples['image'], dtype=jnp.float32)  # input samples
        y = jnp.asarray(a=samples['label'], dtype=jnp.int32)  # annotated labels (batch_size, num_experts)

        logits = prediction_step(x=x, state=state)
        accuracy_accum.update(logits=logits, labels=y)

    return accuracy_accum.compute()


@hydra.main(version_base=None, config_path='.', config_name='conf')
def main(cfg: DictConfig) -> None:
    jax.config.update('jax_disable_jit', cfg.jax.disable_jit)
    jax.config.update('jax_platforms', cfg.jax.platform)
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(cfg.jax.mem)

    # region DATA
    dset_train_labelled = get_dataset(
        dataset_file=cfg.dataset.train_labelled_file,
        root=cfg.dataset.root,
        resize=cfg.dataset.resized_shape
    )
    dset_train_unlabelled = get_dataset(
        dataset_file=cfg.dataset.train_unlabelled_file,
        root=cfg.dataset.root,
        resize=cfg.dataset.resized_shape
    )
    dset_test = get_dataset(
        dataset_file=cfg.dataset.test_file,
        root=cfg.dataset.root,
        resize=cfg.dataset.crop_size if cfg.dataset.crop_size is not None else cfg.dataset.resized_shape
    )
    # endregion
    
    # region MODELS
    state = initialise_huggingface_resnet(
        model=resnet18(num_classes=cfg.dataset.num_classes, dtype=jnp.bfloat16),
        sample=jnp.expand_dims(a=dset_train_labelled[0]['image'] / 255, axis=0),
        num_training_samples=len(dset_train_unlabelled),
        lr=cfg.training.lr,
        batch_size=cfg.training.batch_size,
        num_epochs=cfg.training.num_epochs,
        key=jax.random.key(seed=random.randint(a=0, b=1_000))
    )

    # options to store models
    ckpt_options = ocp.CheckpointManagerOptions(
        save_interval_steps=1,
        max_to_keep=1,
        step_format_fixed_length=3,
        enable_async_checkpointing=True
    )
    # endregion

    # region Mlflow
    mlflow.set_tracking_uri(uri=cfg.experiment.tracking_uri)
    mlflow.set_experiment(experiment_name=cfg.experiment.name)
    mlflow.disable_system_metrics_logging()

    # create a directory for storage (if not existed)
    if not os.path.exists(path=cfg.experiment.logdir):
        Path(cfg.experiment.logdir).mkdir(parents=True, exist_ok=True)
    # endregion

    # enable mlflow tracking
    with mlflow.start_run(run_id=cfg.experiment.run_id, log_system_metrics=False) as mlflow_run:
        # append run id into the artifact path
        ckpt_dir = os.path.join(
            os.getcwd(),
            cfg.experiment.logdir,
            cfg.experiment.name,
            mlflow_run.info.run_id
        )

        # enable an orbax checkpoint manager to save model's parameters
        with ocp.CheckpointManager(directory=ckpt_dir, options=ckpt_options) as ckpt_mngr:

            if cfg.experiment.run_id is None:  # new run
                # log hyper-parameters
                mlflow.log_params(
                    params=flatdict.FlatDict(
                        value=OmegaConf.to_container(cfg=cfg),
                        delimiter='.'
                    )
                )

                # log source code
                mlflow.log_artifact(
                    local_path=os.path.abspath(path=__file__),
                    artifact_path='source_code'
                )

                start_epoch_id = 0
            else:  # load parameters from an existing run
                start_epoch_id = ckpt_mngr.latest_step()

                checkpoint = ckpt_mngr.restore(
                    step=start_epoch_id,
                    args=ocp.args.StandardRestore(item=state)
                )

                state = checkpoint

                del checkpoint

            # training
            for epoch_id in tqdm(
                iterable=range(start_epoch_id, cfg.training.num_epochs, 1),
                desc='progress',
                ncols=80,
                leave=True,
                position=1,
                colour='green',
                disable=not cfg.training.progress_bar
            ):
                state, loss = train(
                    dataset_labelled=dset_train_labelled,
                    dataset_unlabelled=dset_train_unlabelled,
                    state=state,
                    cfg=cfg
                )

                accuracy = evaluate(dataset=dset_test, state=state, cfg=cfg)

                # save parameters asynchronously
                ckpt_mngr.save(
                    step=epoch_id + 1,
                    args=ocp.args.StandardSave(state)
                )

                # log metrics NOTE: batch logging has issue with jax.Array
                for key, value in zip(('loss', 'accuracy'), (loss, accuracy)):
                    mlflow.log_metric(key=key, value=value, step=epoch_id + 1)

                # # wait for checkpoint manager completing the asynchronous saving
                # ckpt_mngr.wait_until_finished()

    return None


if __name__ == '__main__':
    # region CACHING to reduce compilation time
    jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 120)
    # endregion

    os.environ['XLA_FLAGS'] = (
        '--xla_gpu_enable_triton_softmax_fusion=true '
        '--xla_gpu_triton_gemm_any=True '
    )

    main()
