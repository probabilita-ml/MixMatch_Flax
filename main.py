# python standard libraries
import argparse
import os
from pathlib import Path
import random
from functools import partial

# third-party libraries
from tqdm import tqdm

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
    initialise_huggingface_resnet,
    list_of_ints,
    list_of_floats
)

from ResNet import resnet18


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Parse input arguments')

    parser.add_argument('--experiment-name', type=str, default='MixMatch')
    parser.add_argument('--dataset-name', type=str, help='Name of dataset')
    parser.add_argument('--ds-root', type=str, help='Root folder of dataset')

    parser.add_argument('--train-labelled-file', type=str, help='Path to json file of labelled training samples')
    parser.add_argument('--train-unlabelled-file', type=str)
    parser.add_argument('--test-file', type=str)

    parser.add_argument('--num-classes', type=int, help='Number of classes')

    parser.add_argument('--lr', type=float, help='Learning rate for gating model')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--num-epochs', type=int, default=100, help='Number of epochs')

    parser.add_argument('--sharpen-factor', type=float, default=0.5)
    parser.add_argument('--alpha', type=float, default=0.75)
    parser.add_argument('--lambda-u', type=float, default=75.)

    # region DATA AUGMENTATION
    parser.add_argument(
        '--resized-shape',
        type=list_of_ints,
        default=None,
        help='Tuple of width and height, e.g., (230, 230)'
    )
    parser.add_argument(
        '--crop-size',
        type=list_of_ints,
        default=None,
        help='Tuple of width and height, e.g., (224, 224)'
    )
    parser.add_argument(
        '--prob-random-h-flip',
        type=float,
        default=None,
        help='Probability to horizontal-flip'
    )
    parser.add_argument(
        '--mean',
        type=list_of_floats,
        default=None,
        help='The mean for sample normalisation'
    )
    parser.add_argument(
        '--std',
        type=list_of_floats,
        default=None,
        help='The standard deviation for sample normalisation'
    )
    # endregion

    parser.add_argument('--run-id', type=str, default=None, help='Run ID in MLFlow')

    parser.add_argument(
        '--jax-platform',
        type=str,
        default='cpu',
        help='cpu, cuda or tpu'
    )
    parser.add_argument(
        '--mem-frac',
        type=float,
        default=0.9,
        help='Percentage of GPU memory allocated for Jax'
    )

    parser.add_argument('--prefetch-size', type=int, default=8)
    parser.add_argument('--num-threads', type=int, default=2)

    parser.add_argument('--progress-bar', dest='progress_bar', action='store_true')
    parser.add_argument('--no-progress-bar', dest='progress_bar', action='store_false')
    parser.set_defaults(progress_bar=True)

    parser.add_argument('--train', dest='train', action='store_true')
    parser.add_argument('--debug', dest='train', action='store_false')
    parser.set_defaults(train=True)

    parser.add_argument(
        '--tracking-uri',
        type=str,
        default='http://127.0.0.1:5000',
        help='MLFlow server'
    )
    parser.add_argument('--logdir', type=str, default='logdir', help='Path to save model')

    args = parser.parse_args()

    return args


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
    apply_batch = jax.vmap(fun=state.apply_fn, in_axes=(None, 0, None))

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


def train(
    dataset_labelled: dx._c.Buffer,
    dataset_unlabelled: dx._c.Buffer,
    state: TrainState
) -> tuple[TrainState, Scalar]:
    """the main training procedure
    """
    # batching and shuffling the dataset
    stream_labelled = prepare_dataset(
        dataset=dataset_labelled,
        shuffle=True,
        batch_size=args.batch_size,
        prefetch_size=args.prefetch_size,
        num_threads=args.num_threads,
        mean=args.mean,
        std=args.std,
        random_crop_size=None,
        prob_random_h_flip=None
    )
    stream_unlabelled = prepare_dataset(
        dataset=dataset_unlabelled,
        shuffle=True,
        batch_size=args.batch_size,
        prefetch_size=args.prefetch_size,
        num_threads=args.num_threads,
        mean=args.mean,
        std=args.std,
        random_crop_size=None,
        prob_random_h_flip=None
    )

    # metric to track the training loss
    loss_accum = metrics.Average()

    fold_in_batch = jax.vmap(fun=jax.random.fold_in, in_axes=(0, 0))
    crop_size = (*args.crop_size, 3)

    for unlabelled_samples in tqdm(
        iterable=stream_unlabelled,
        desc='train',
        total=len(dataset_unlabelled)//args.batch_size,
        ncols=80,
        leave=False,
        position=2,
        colour='blue',
        disable=not args.progress_bar
    ):
        if (len(unlabelled_samples['image']) < args.batch_size):
            break

        try:
            labelled_samples = next(stream_labelled)
            if (len(labelled_samples['image']) < args.batch_size):
                raise StopIteration
        except StopIteration:
            # reset the labelled data subset
            stream_labelled = prepare_dataset(
                dataset=dataset_labelled,
                shuffle=True,
                batch_size=args.batch_size,
                prefetch_size=args.prefetch_size,
                num_threads=args.num_threads,
                mean=args.mean,
                std=args.std,
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
            prob_random_h_flip=args.prob_random_h_flip
        )
        u1 = augment_batch_images(
            keys1=keys1,
            keys2=keys11,
            images=unlabelled_inputs,
            crop_size=crop_size,
            prob_random_h_flip=args.prob_random_h_flip
        )
        u1 = augment_batch_images(
            keys1=keys11,
            keys2=keys1,
            images=u1,
            crop_size=crop_size,
            prob_random_h_flip=args.prob_random_h_flip
        )
        u2 = augment_batch_images(
            keys1=keys2,
            keys2=keys21,
            images=unlabelled_inputs,
            crop_size=crop_size,
            prob_random_h_flip=args.prob_random_h_flip
        )
        u2 = augment_batch_images(
            keys1=keys21,
            keys2=keys2,
            images=u2,
            crop_size=crop_size,
            prob_random_h_flip=args.prob_random_h_flip
        )
        lx = optax.smooth_labels(
            labels=jax.nn.one_hot(x=y, num_classes=args.num_classes),
            alpha=0.01
        )

        # guess labels
        logits_u1, _ = state.apply_fn(
            variables={'params': state.params, 'batch_stats': state.batch_stats},
            x=u1,
            train=True
        )
        logits_u2, _ = state.apply_fn(
            variables={'params': state.params, 'batch_stats': state.batch_stats},
            x=u2,
            train=True
        )

        # average
        lu1 = jax.nn.softmax(x=logits_u1, axis=-1)
        lu2 = jax.nn.softmax(x=logits_u2, axis=-1)
        lu = 0.5 * (lu1 + lu2)

        # sharpening
        lu = lu ** (1 / args.sharpen_factor)
        lu = lu / jnp.sum(a=lu, axis=-1, keepdims=True)  # normalised

        # region MIXUP
        inputs = jnp.concatenate(arrays=[x, u1, u2], axis=0)
        labels = jnp.concatenate(arrays=[lx, lu, lu], axis=0)

        index_rng, mixup_ratio_rng = jax.random.split(
            key=jax.random.key(seed=random.randint(a=0, b=100_000))
        )
        index = jax.random.permutation(key=index_rng, x=len(inputs))
        lam = jax.random.beta(key=mixup_ratio_rng, a=args.alpha, b=args.alpha)
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
            lambda_u=args.lambda_u
        )

        if jnp.isnan(loss):
            raise ValueError('Training loss is NaN.')

        # tracking
        loss_accum.update(values=loss)

    return state, loss_accum.compute()


def evaluate(dataset: dx._c.Buffer,state: TrainState) -> Scalar:
    """calculate the average cluster probability vector

    Args:
        dataset:
        state:
    """
    # prepare dataset for training
    dset = prepare_dataset(
        dataset=dataset,
        shuffle=True,
        batch_size=args.batch_size,
        prefetch_size=args.prefetch_size,
        num_threads=args.num_threads,
        mean=args.mean,
        std=args.std,
        random_crop_size=None,
        prob_random_h_flip=None
    )

    accuracy_accum = metrics.Accuracy()

    for samples in tqdm(
        iterable=dset,
        desc='evaluate',
        total=len(dataset)//args.batch_size,
        ncols=80,
        leave=False,
        position=2,
        colour='blue',
        disable=not args.progress_bar
    ):
        x = jnp.asarray(a=samples['image'], dtype=jnp.float32)  # input samples
        y = jnp.asarray(a=samples['label'], dtype=jnp.int32)  # annotated labels (batch_size, num_experts)

        logits, _ = state.apply_fn(
            variables={'params': state.params, 'batch_stats': state.batch_stats},
            x=x,
            train=False
        )
        accuracy_accum.update(logits=logits, labels=y)

    return accuracy_accum.compute()


def main() -> None:
    # region DATA
    dset_train_labelled = get_dataset(
        dataset_file=args.train_labelled_file,
        root=args.ds_root,
        resize=args.resized_shape
    )
    dset_train_unlabelled = get_dataset(
        dataset_file=args.train_unlabelled_file,
        root=args.ds_root,
        resize=args.resized_shape
    )
    dset_test = get_dataset(
        dataset_file=args.test_file,
        root=args.ds_root,
        resize=args.crop_size if args.crop_size is not None else args.resized_shape
    )
    # endregion
    
    # region MODELS
    state = initialise_huggingface_resnet(
        model=resnet18(num_classes=args.num_classes, dtype=jnp.bfloat16),
        sample=jnp.expand_dims(a=dset_train_labelled[0]['image'] / 255, axis=0),
        num_training_samples=len(dset_train_unlabelled),
        lr=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
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
    mlflow.set_tracking_uri(uri=args.tracking_uri)
    mlflow.set_experiment(experiment_name=args.experiment_name)
    mlflow.set_system_metrics_sampling_interval(interval=600)
    mlflow.set_system_metrics_samples_before_logging(samples=1)

    # create a directory for storage (if not existed)
    if not os.path.exists(path=args.logdir):
        Path(args.logdir).mkdir(parents=True, exist_ok=True)
    # endregion

    # enable mlflow tracking
    with mlflow.start_run(run_id=args.run_id, log_system_metrics=True) as mlflow_run:
        # append run id into the artifact path
        ckpt_dir = os.path.join(args.logdir, args.experiment_name, mlflow_run.info.run_id)

        # enable an orbax checkpoint manager to save model's parameters
        with ocp.CheckpointManager(directory=ckpt_dir, options=ckpt_options) as ckpt_mngr:

            if args.run_id is None:  # new run

                # log hyper-parameters
                mlflow.log_params(params=args.__dict__)

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
                iterable=range(start_epoch_id, args.num_epochs, 1),
                desc='progress',
                ncols=80,
                leave=True,
                position=1,
                colour='green',
                disable=not args.progress_bar
            ):
                state, loss = train(
                    dataset_labelled=dset_train_labelled,
                    dataset_unlabelled=dset_train_unlabelled,
                    state=state
                )

                accuracy = evaluate(dataset=dset_test, state=state)

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
    # get the global configuration
    args = parse_arguments()

    jax.config.update('jax_disable_jit', not args.train)
    jax.config.update('jax_platforms', args.jax_platform)

    # region CACHING to reduce compilation time
    # jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
    # jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
    # jax.config.update("jax_persistent_cache_min_compile_time_secs", 120)
    # endregion

    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(args.mem_frac)

    os.environ['XLA_FLAGS'] = (
        '--xla_gpu_enable_triton_softmax_fusion=true '
        '--xla_gpu_triton_gemm_any=True '
    )

    main()
