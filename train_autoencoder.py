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
from util.learning_rate_scheduler import create_learning_rate_fn
from absl import app, flags


FLAGS = flags.FLAGS


#! Training hyperparameter
flags.DEFINE_float("AE_learning_rate", 1e-4, "The autoencoders learning rate")
flags.DEFINE_float("AE_weight_decay", 0.1, "The autoencoders weight decay")
flags.DEFINE_float("AE_ema_momentum", 0.990, "The autoencoders EMA momentum")
flags.DEFINE_integer("AE_epochs", 30, "The autoencoders training epochs")


#! Others
flags.DEFINE_integer("AE_batch_size", 64, "The autoencoders batch size")
flags.DEFINE_integer("AE_run_seed", 0, "The seed used to generate JAX prng")
flags.DEFINE_integer("AE_ecg_length", 30_720, "The length of a single ECG in samples")
flags.DEFINE_bool("AE_normalise_data", True, "If true, normalises all ecg's to be between 0 and 1")

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

    model_outputs, latent_space, _ = state.apply_fn(variables, ecgs, train=False)

    plot_ecg = ecgs[0]
    plot_latent_space = latent_space[0]
    plot_latent_space = plot_latent_space.reshape((-1))
    plot_output = model_outputs[0]
    plt.plot(plot_ecg)
    plt.savefig(f"{img_dir}/epoch_{epoch}_ecg.png")
    plt.close()
    plt.figure()
    plt.plot(plot_output)
    plt.savefig(f"{img_dir}/epoch_{epoch}_ecg_output.png")
    plt.close()
    plt.figure()
    plt.plot(plot_latent_space)
    plt.savefig(f"{img_dir}/epoch_{epoch}_ecg_latent_space.png")
    plt.close()

    plt.figure()
    plt.plot(plot_output)
    plt.plot(plot_ecg, alpha=0.5)
    plt.savefig(f"{img_dir}/epoch_{epoch}_ecg_output_both.png")
    plt.close()


def create_train_state(rng, learning_rate_fn):
    model = AutoEncoder(
        block_depths=1,
        sample_rng = rng
    )
    rng_params, rng = jax.random.split(rng)
    dummy_ecg = jnp.ones((1, FLAGS.AE_ecg_length), dtype=jnp.float32)
    variables = model.init(rng_params, dummy_ecg, train=True)
    tx = optax.adamw(learning_rate=FLAGS.AE_learning_rate, weight_decay=FLAGS.AE_weight_decay)
    param_count = sum(x.size for x in jax.tree_leaves(variables))
    print(f"Autoencoder parameter count: f{param_count}")
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

@partial(jax.jit, static_argnums=2)
def train_step(state, batch, learning_rate_fn):
    def compute_loss(params):
        outputs, mutated_vars = state.apply_fn(
            {
                "params": params,
                "batch_stats": state.batch_stats
            },
            batch, train=True, mutable=["batch_stats"]
        )
        predicted_ecg, latent_space, embedding_space_loss = outputs
        reconstruction_loss = (losses.L2(predicted_ecg, batch)).mean()

        total_loss = reconstruction_loss + embedding_space_loss

        return total_loss, (reconstruction_loss,  embedding_space_loss, mutated_vars)
    grad_fn = jax.value_and_grad(compute_loss, has_aux=True)
    (loss, aux), grads = grad_fn(state.params)
    reconstruction_loss, embedding_space_loss, mutated_vars = aux
    new_state = state.apply_gradients(
        grads=grads, batch_stats=mutated_vars['batch_stats'])

    # new_ema_params = jax.tree_map(
    #     compute_ema_params, new_state.ema_params, new_state.params, new_state.ema_momentum
    # )
    # new_state = new_state.replace(ema_params=new_ema_params)
    #lr = learning_rate_fn(state.step)
    return new_state, loss, reconstruction_loss, embedding_space_loss


def train() -> TrainState:
    tf.config.experimental.set_visible_devices([], 'GPU')

    rng = jax.random.PRNGKey(FLAGS.AE_run_seed)
    dataset_rng, rng = jax.random.split(rng)

    series_iter, label_iter = dataset_loader.load_ecg_dataset(dataset_rng, FLAGS.AE_ecg_length, FLAGS.AE_batch_size, normalise=FLAGS.AE_normalise_data)
    state_rng, rng = jax.random.split(rng)
    
    learning_rate_fn = create_learning_rate_fn(2160)
    state = create_train_state(state_rng, learning_rate_fn)
    
    ema_params = state.params.copy(add_or_replace={})
    state = state.replace(ema_params=ema_params)
    
    

    for epoch in range(FLAGS.AE_epochs):
        pbar = tqdm(range(len(series_iter)), desc=f"Epoch {epoch}")

        for i in pbar:
            series_batch = series_iter[i]
            #label_batch = label_iter[i]

            rng, train_step_rng = jax.random.split(rng)
            state, loss, reconstructin_loss, regularisation_loss = train_step(
                state=state,
                batch=series_batch,
                learning_rate_fn=learning_rate_fn
                )
            new_ema_params = jax.tree_map(compute_ema_params, state.ema_params, state.params)
            state = state.replace(ema_params = new_ema_params)
            pbar.set_postfix({"Loss": f"{loss:.5f}", "REC_L": f"{reconstructin_loss:.5f}", "REG_L": f"{regularisation_loss:.5f}"})

        state = state.replace(epoch=epoch)
        evaluate(series_batch, state, epoch, FLAGS.AE_img_dir)
        checkpoints.save_checkpoint(ckpt_dir=FLAGS.AE_ckpt_dir, target=state, step=epoch)


if __name__ == '__main__':
    app.run(main)