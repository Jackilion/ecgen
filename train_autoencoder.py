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
from util.learning_rate_scheduler import create_learning_rate_fn
import argparse
import yaml
from config.config import Config
from util.data_loader import data_loader
import seaborn as sns; sns.set()

class TrainState(train_state.TrainState):
    #batch_stats: Any
    dropout_rng: Any
    epoch: int = None
    ema_params: Any = None
    ema_momentum: float = None




def evaluate(ecgs, state, epoch, img_dir):
    variables = {"params": state.ema_params}

    model_outputs, latent_space, _ = state.apply_fn(variables, ecgs, train=False)

    plot_ecg = ecgs[0]
    plot_latent_space = latent_space[0]
    plot_output = model_outputs[0]
    #plot 2 by 2 grid of ecgs
    #plt.figure()#
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    for i, ax in enumerate(axs.flat):
        ax.plot(ecgs[i])
        ax.plot(model_outputs[i])
    plt.savefig(f"{img_dir}/epoch_{epoch}_ecg_grid.png")

    plt.close()
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    plt.grid(False)
    for i, ax in enumerate(axs.flat):
        ax.imshow(latent_space[i].T, aspect="auto", cmap="viridis")
    plt.savefig(f"{img_dir}/epoch_{epoch}_ecg_latent_space_grid_tokens.png")
    plt.close()

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    for i, ax in enumerate(axs.flat):
        ax.plot(latent_space[i].reshape((-1)))
    plt.savefig(f"{img_dir}/epoch_{epoch}_ecg_latent_space_grid_flat.png")
    plt.close()


def create_train_state(rng, learning_rate_fn):
    config = Config().settings
    model = AutoEncoder(
        block_depths=config["AE_block_depths"],
        embed_size_K=config["AE_embed_size_K"],
        embed_dim_D=config["AE_embed_dim_D"],
        commitment_loss_beta=config["AE_commitment_loss_beta"]
    )
    rng_params, rng = jax.random.split(rng)
    rng_dropout, rng = jax.random.split(rng)
    dummy_ecg = jnp.ones((16, config["AE_sample_length"]), dtype=jnp.float32)
    variables = model.init(rng_params, dummy_ecg, train=False)
    tx = optax.adamw(learning_rate=config["AE_learning_rate"], weight_decay=config["AE_weight_decay"])
    #tx = optax.adam(learning_rate=config["learning_rate"])
    param_count = sum(x.size for x in jax.tree_leaves(variables))
    print(f"Autoencoder parameter count: {param_count}")
    return TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=tx,
        epoch = 0,
        dropout_rng = rng_dropout,
        #batch_stats=variables["batch_stats"],
        ema_params=None,
        ema_momentum=config["AE_ema_momentum"]
    )


def compute_ema_params(ema_params, new_params):
    config = Config().settings
    ema_momentum = config["AE_ema_momentum"]
    return ema_momentum * ema_params + (1-ema_momentum)*new_params
    

def copy_params_to_ema(state):
    return state.replace(params_ema = state.params)

@jax.vmap
def L2(prediction, targets):
    return jnp.square(jnp.subtract(prediction, targets))

@jax.vmap
def L1(prediction, targets):
    return jnp.abs(jnp.subtract(prediction, targets))

@partial(jax.jit, static_argnums=2)
def train_step(state, batch, learning_rate_fn, dropout_key):
    dropout_train_key = jax.random.fold_in(key=dropout_key, data=state.step)
    def compute_loss(params):
        predicted_ecg, latent_space, embedding_space_loss = state.apply_fn(
            {
                "params": params,
                #"batch_stats": state.batch_stats
            },
            batch, train=True, rngs={'dropout': dropout_train_key} #, mutable=["batch_stats"],
            
        
        )
        #predicted_ecg, latent_space, embedding_space_loss = outputs
        reconstruction_loss = (L2(predicted_ecg, batch)).mean()

        total_loss = reconstruction_loss + embedding_space_loss

        return total_loss, (reconstruction_loss,  embedding_space_loss)
    grad_fn = jax.value_and_grad(compute_loss, has_aux=True)
    (loss, aux), grads = grad_fn(state.params)
    reconstruction_loss, embedding_space_loss = aux
    # new_state = state.apply_gradients(
    #     grads=grads, batch_stats=mutated_vars['batch_stats'])
    new_state = state.apply_gradients(
        grads=grads
    )

    # new_ema_params = jax.tree_map(
    #     compute_ema_params, new_state.ema_params, new_state.params, new_state.ema_momentum
    # )
    # new_state = new_state.replace(ema_params=new_ema_params)
    #lr = learning_rate_fn(state.step)
    return new_state, loss, reconstruction_loss, embedding_space_loss


def train() -> TrainState:
    tf.config.experimental.set_visible_devices([], 'GPU')
    #FLAGS = Config().instance
    config = Config().settings
    

    #print(config)
    rng = jax.random.PRNGKey(config["AE_jax_seed"])
    #dataset_rng, rng = jax.random.split(rng)

    #series_iter, label_iter = dataset_loader.load_ecg_dataset(dataset_rng, FLAGS.AE_signal_length, FLAGS.AE_batch_size, normalise=FLAGS.AE_normalise_data)

    
    state_rng, rng = jax.random.split(rng)
    dropout_rng, rng = jax.random.split(rng)
    
    learning_rate_fn = create_learning_rate_fn(epochs=config["AE_epochs"], steps_per_epoch = 3500, base_learning_rate= config["AE_learning_rate"], max_learning_rate=config["AE_max_learning_rate"], warmup_epochs = config["AE_warmup_epochs"] )
    state = create_train_state(state_rng, learning_rate_fn)
    
    ema_params = state.params.copy()
    state = state.replace(ema_params=ema_params)
    
    
    losses = []


    for epoch in range(config["AE_epochs"]):
        pbar = tqdm(data_loader(config["AE_dataset_chunks"], config["AE_dataset_root"]) , desc=f"Epoch {epoch}")
        for signal_batch in pbar:
            #signal_batch = series_iter[i]
            #label_batch = label_iter[i]
            # for i in range(len(signal_batch)):
            #     plt.plot(signal_batch[i])
            #     plt.savefig(f"log/images/lrelu/ecg_{i}.png")
            #     plt.figure()
            # quit()
            rng, train_step_rng = jax.random.split(rng)
            state, loss, reconstructin_loss, regularisation_loss = train_step(
                state=state,
                batch=signal_batch,
                learning_rate_fn=learning_rate_fn,
                dropout_key=dropout_rng
                )
            new_ema_params = jax.tree_map(compute_ema_params, state.ema_params, state.params)
            state = state.replace(ema_params = new_ema_params)
            pbar.set_postfix({"Loss": f"{loss:.5f}", "REC_L": f"{reconstructin_loss:.5f}", "REG_L": f"{regularisation_loss:.5f}"})
            losses.append(loss)
        state = state.replace(epoch=epoch)
        evaluate(signal_batch, state, epoch, config["output_root_dir"] + config["image_dir"])
        checkpoints.save_checkpoint(ckpt_dir=config["output_root_dir"] + config["checkpoint_dir"], target=state, step=epoch)

    plt.plot(losses)
    plt.savefig(config["output_root_dir"] + "loss.png")
    plt.close()


    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VQ-VAE')
    parser.add_argument('--config', type=str, default='config/ae_ecg_300ep_512hz.yml', help='Path to the config file')
    args = parser.parse_args()
    config = Config(f"{args.config}")
    train()