from argparse import Namespace
from datetime import datetime
from functools import partial
from pathlib import Path
import time
import jaxlib.xla_extension
import jax
import jax.numpy as jnp
import numpy as onp
import optax
from typing import Any
from flax.training import (train_state, checkpoints)
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
from model.autoencoder import AutoEncoder
import util.losses as losses
import dataset_loader

from absl import app, flags

FLAGS = flags.FLAGS


#! Training hyperparameter
flags.DEFINE_float("AE_learning_rate", 1e-4, "The autoencoders learning rate")
flags.DEFINE_float("AE_weight_decay", 1e-4, "The autoencoders weight decay")
flags.DEFINE_float("AE_ema_momentum", 0.990, "The autoencoders EMA momentum")
flags.DEFINE_integer("AE_epochs", 100, "The autoencoders training epochs")

#! Others
flags.DEFINE_integer("AE_batch_size", 64, "The autoencoders batch size")
flags.DEFINE_integer("AE_run_seed", 0, "The seed used to generate JAX prng")
flags.DEFINE_integer("AE_ecg_length", 30_720, "The length of a single ECG in samples")

#! Logging flags
now = datetime.now().strftime("%Y%m%d-%H%M%S")
flags.DEFINE_string("AE_output_dir", f"./outputs/autoencoder/{now}", "The output root directory where all the models output will be saved")
flags.DEFINE_string("AE_img_dir", f"./outputs/autoencoder/{now}/images", "The directory where evaluation images will be stored")
flags.DEFINE_string("AE_log_dir", f"./outputs/autoencoder/{now}/logs", "The directory where logs will be stored")
flags.DEFINE_string("AE_ckpt_dir", f"./outputs/autoencoder/{now}/checkpoints", "The directory where model checkpoints will be stored")



def main(argv):
    Path(FLAGS.AE_output_dir).mkdir(parents=True, exist_ok=True)
    Path(f"{FLAGS.AE_img_dir}").mkdir(parents=True, exist_ok=True)
    Path(f"{FLAGS.AE_log_dir}").mkdir(parents=True, exist_ok=True)
    Path(f"{FLAGS.AE_ckpt_dir}").mkdir(parents=True, exist_ok=True)
    
    train()


# SERIES_LENGTH = 30_720  # Length of a single ECG in samples
# BATCH_SIZE = 64


class TrainState(train_state.TrainState):
    batch_stats: Any
    epoch: int = None
    ema_params: Any = None
    ema_momentum: float = None





def evaluate(ecgs, state, epoch, img_dir):
    variables = {"params": state.ema_params, "batch_stats": state.batch_stats}

    model_outputs, _ = state.apply_fn(variables, ecgs, train=False)

    plot_ecg = ecgs[0]
    plot_output = model_outputs[0]
    plt.plot(plot_ecg)
    plt.savefig(f"{img_dir}/epoch_{epoch}_ecg.png")
    plt.close()
    plt.figure()
    plt.plot(plot_output)
    plt.savefig(f"{img_dir}/epoch_{epoch}_ecg_output.png")
    plt.close()
    plt.figure()
    plt.plot(plot_output[0:5000])
    plt.savefig(f"{img_dir}/epoch_{epoch}_ecg_output_zoom.png")
    plt.close()

    plt.figure()
    plt.plot(plot_output[0:5000])
    plt.plot(plot_ecg[0:5000], alpha=0.5)
    plt.savefig(f"{img_dir}/epoch_{epoch}_ecg_output_zoom_both.png")
    plt.close()


def create_train_state(rng):
    model = AutoEncoder(
        block_depths=1,
        sample_rng = rng
    )
    rng_params, rng = jax.random.split(rng)
    dummy_ecg = jnp.ones((1, FLAGS.AE_ecg_length), dtype=jnp.float32)
    variables = model.init(rng_params, dummy_ecg, train=True)

    tx = optax.adamw(learning_rate=FLAGS.AE_learning_rate,
                     weight_decay=FLAGS.AE_weight_decay)

    return TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=tx,
        epoch = 0,
        batch_stats=variables["batch_stats"],
        ema_params=None,
        ema_momentum=FLAGS.AE_ema_momentum
    )


def compute_ema_params(ema_params, new_params):
    ema_momentum = FLAGS.AE_ema_momentum
    return ema_momentum * ema_params + (1-ema_momentum)*new_params
    

def copy_params_to_ema(state):
    return state.replace(params_ema = state.params)

@jax.jit
def train_step(state, batch):
    def compute_loss(params):
        outputs, mutated_vars = state.apply_fn(
            {
                "params": params,
                "batch_stats": state.batch_stats
            },
            batch, train=True, mutable=["batch_stats"]
        )
        predicted_ecg, latent_space = outputs
        reconstruction_loss = losses.L1(predicted_ecg, batch).mean()
        deterministic, mean, log_var = latent_space
        regularisation_loss = losses.KLD(mean, log_var).mean()
        total_loss = losses.vae_loss(reconstruction_loss, regularisation_loss, state.epoch)
        return total_loss, (reconstruction_loss, regularisation_loss, mutated_vars)
    grad_fn = jax.value_and_grad(compute_loss, has_aux=True)
    (loss, aux), grads = grad_fn(state.params)
    reconstruction_loss, regularisation_loss, mutated_vars = aux
    new_state = state.apply_gradients(
        grads=grads, batch_stats=mutated_vars['batch_stats'])

    # new_ema_params = jax.tree_map(
    #     compute_ema_params, new_state.ema_params, new_state.params, new_state.ema_momentum
    # )
    # new_state = new_state.replace(ema_params=new_ema_params)
    
    return new_state, loss, reconstruction_loss, regularisation_loss


def train() -> TrainState:
    tf.config.experimental.set_visible_devices([], 'GPU')

    rng = jax.random.PRNGKey(FLAGS.AE_run_seed)
    dataset_rng, rng = jax.random.split(rng)

    series_iter, label_iter = dataset_loader.load_ecg_dataset(dataset_rng, FLAGS.AE_ecg_length, FLAGS.AE_batch_size)
    state_rng, rng = jax.random.split(rng)
    state = create_train_state(state_rng)
    
    ema_params = state.params.copy(add_or_replace={})
    state = state.replace(ema_params=ema_params)

    for epoch in range(FLAGS.AE_epochs):
        pbar = tqdm(range(len(series_iter)), desc=f"Epoch {epoch}")

        for i in pbar:
            series_batch = series_iter[i]
            #label_batch = label_iter[i]

            rng, train_step_rng = jax.random.split(rng)
            state, loss, recon_loss, regu_loss = train_step(
                state=state,
                batch=series_batch)
            new_ema_params = jax.tree_map(compute_ema_params, state.ema_params, state.params)
            state = state.replace(ema_params = new_ema_params)
            pbar.set_postfix({"Loss": f"{loss:.5f}", "L2": f"{recon_loss:.5f}", "KLD": f"{regu_loss:.5f}"})

        state = state.replace(epoch=epoch)
        evaluate(series_batch, state, epoch, FLAGS.AE_img_dir)
        checkpoints.save_checkpoint(ckpt_dir=FLAGS.AE_ckpt_dir, target=state, step=epoch)


if __name__ == '__main__':
    app.run(main)