import argparse
import flax
import jax
import jax.numpy as jnp
import numpy as onp
import optax
import multiprocessing

from datetime import datetime
from tqdm import tqdm
from pathlib import Path
from typing import Any

import neurokit2 as nk
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from config.config import Config
from model.autoencoder import AutoEncoder
from model.ddim_medium import DiffusionModelMedium
from util.learning_rate_scheduler import create_learning_rate_fn
import util.losses as losses
import dataset_loader

from flax.training import train_state, checkpoints
from train_autoencoder import TrainState as AutoEncoderTrainState
from train_ddim_medium import TrainState as DDIMTrainState

from functools import partial

# ------------------------------------------------------------------------------
# 1) Load AutoEncoder & DDIM from checkpoints
# ------------------------------------------------------------------------------
def get_autoencoder(rng: jax.random.PRNGKey) -> AutoEncoderTrainState:
    config = {
        "AE_block_depths": 1,
        "AE_embed_size_K": 1024,
        "AE_embed_dim_D": 16,
        "AE_commitment_loss_beta": .8,
        "AE_learning_rate": 0.001,
        "AE_weight_decay": 0.01,
        "AE_convolution_filters": [32, 32, 32, 32, 32],
        "AE_kernel_sizes": [32, 16, 8, 4, 4],
        "AE_dropout": 0.1,
        "AE_ema_momentum": 0.999,
        "output_root_dir": "/home/dominik.kranz/ecgen/output/VQ-VAE/",
        "checkpoint_dir": "checkpoints/",
    }
    model = AutoEncoder(
        block_depths=config["AE_block_depths"],
        embed_size_K=config["AE_embed_size_K"],
        embed_dim_D=config["AE_embed_dim_D"],
        commitment_loss_beta=config["AE_commitment_loss_beta"],
        convolution_filters=config["AE_convolution_filters"],
        kernel_sizes=config["AE_kernel_sizes"],
        dropout=config["AE_dropout"]
    )

    rng_params, rng = jax.random.split(rng)
    dummy_ecg = jnp.ones((64, 5*512), dtype=jnp.float32)
    variables = model.init(rng_params, dummy_ecg, train=True)

    tx = optax.adamw(
        learning_rate=config["AE_learning_rate"], 
        weight_decay=config["AE_weight_decay"]
    )
    init_state = AutoEncoderTrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=tx,
        dropout_rng=rng,
        ema_params=None,
        ema_momentum=config["AE_ema_momentum"]
    )

    checkpoint_path = config["output_root_dir"] + config["checkpoint_dir"] + "25"
    ae_state = flax.training.checkpoints.restore_checkpoint(
        ckpt_dir=checkpoint_path, target=init_state, step=25
    )
    return ae_state

def get_ddim(rng: jax.random.PRNGKey) -> DDIMTrainState:
    config = {
        "DDIM_convolution_filters": [128, 128, 128, 128, 128],
        "DDIM_batch_dims": [1440, 16],  # shape of each sample
        "DDIM_block_depths": 2,
        "DDIM_learning_rate": 0.0001,
        "DDIM_weight_decay": 0.001,
        "DDIM_ema_momentum": 0.99,
        "checkpoint_dir": "/home/dominik.kranz/data/checkpoints/ecgen/medium/5"
    }
    model = DiffusionModelMedium(
        feature_sizes=config["DDIM_convolution_filters"],
        block_depths=config["DDIM_block_depths"],
    )

    rng_init, rng_params = jax.random.split(rng)
    dummy_batch = jnp.ones((1, config["DDIM_batch_dims"][0], config["DDIM_batch_dims"][1]), 
                           dtype=jnp.float32)
    variables = model.init(rng_init, dummy_batch, rng_params, train=False)

    tx = optax.adamw(
        learning_rate=config["DDIM_learning_rate"], 
        weight_decay=config["DDIM_weight_decay"]
    )
    ddim_state = DDIMTrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=tx,
        dropout_rng=rng,
        ema_params=variables["params"],
        ema_momentum=config["DDIM_ema_momentum"]
    )

    checkpoint_path = config["checkpoint_dir"]
    ddim_state = flax.training.checkpoints.restore_checkpoint(
        ckpt_dir=checkpoint_path, target=ddim_state, step=5
    )
    return ddim_state


# ------------------------------------------------------------------------------
# 2) JIT-compile your model calls
# ------------------------------------------------------------------------------
@partial(jax.jit, static_argnums=(2,))
def generate_ddim(ddim_params, rng, sample_count, ddim_state):
    """
    Generate a batch of shapes (sample_count, 1440, 16) using the DiffusionModel.
    """
    return ddim_state.apply_fn(
        {"params": ddim_params}, rng, sample_count, 
        method=DiffusionModelMedium.generate
    )

@jax.jit
def autoencode(ae_params, x, ae_state):
    """
    Do a batched embed->decode in one call for the entire batch x.
    """
    #input shape is (100, 1440, 16)
    #reshape to (-1, 80, 16)
    x = jnp.reshape(x, (-1, 80, 16))
    embedded = ae_state.apply_fn({"params": ae_params}, x, method=AutoEncoder.embed)
    decoded = ae_state.apply_fn({"params": ae_params}, embedded, method=AutoEncoder.decode)
    #reshape back
    return jnp.reshape(decoded, (100, -1))


# ------------------------------------------------------------------------------
# 3) Main: generate data in batches, autoencode in batches, then postprocess
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # If you really need to spawn, do so, but usually for JAX CPU/GPU usage
    # you can skip or carefully set:
    multiprocessing.set_start_method("spawn", force=True)

    rng = jax.random.PRNGKey(2)


    # 1) Load models
    ae_state = get_autoencoder(rng)
    ddim_state = get_ddim(rng)

    # Number of total loops you want, each loop generating 'batch_size' samples
    N_GENERATION_LOOPS = 2000
    BATCH_SIZE = 100  # each call to generate_ddim will produce 500 samples

    results = []
    for _ in tqdm(range(N_GENERATION_LOOPS)):
        rng, run_rng = jax.random.split(rng)

        # 2) Generate entire batch at once
        generated_batch = generate_ddim(ddim_state.params, run_rng, BATCH_SIZE, ddim_state)
        # generated_batch shape: (BATCH_SIZE, 1440, 16)

        # 3) AE embed & decode in one shot
        reconstructed_batch = autoencode(ae_state.params, generated_batch, ae_state)
        # shape: (BATCH_SIZE, 1440, 16)
        print(reconstructed_batch.shape)
        plt.figure(figsize=(100, 10))
        plt.plot(reconstructed_batch[0].flatten())
        plt.savefig("test.png")
        quit()

        # 4) Move to NumPy, do post-processing in Python
        reconstructed_np = onp.array(reconstructed_batch)  # CPU memory

        for i in range(BATCH_SIZE):
            # Flatten, clean, detect peaks
            signal_1d = reconstructed_np[i].flatten()
            cleaned = nk.ecg_clean(signal_1d, sampling_rate=512)
            rpeaks = nk.ecg_findpeaks(cleaned, sampling_rate=512)["ECG_R_Peaks"]
            # Convert from samples to ms:
            rpeaks_ms = (rpeaks / 512) * 1000

            # Compute meanNN & sdNN
            nn_intervals = onp.diff(rpeaks_ms)
            mean_nn = onp.mean(nn_intervals) if len(nn_intervals) > 0 else onp.nan
            sdnn = onp.std(nn_intervals) if len(nn_intervals) > 0 else onp.nan

            results.append((mean_nn, sdnn, rpeaks_ms))

    # 5) Save
    df = pd.DataFrame(results, columns=["meanNN", "sdNN", "rpeaks"])
    df.to_parquet("/home/dominik.kranz/data/generated_ecgen_large_bd_2_200k.parquet")
    print("Done! Saved results to parquet.")
