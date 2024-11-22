import argparse
from functools import partial
import flax
import numpy as onp
import jax.numpy as jnp
import matplotlib.pyplot as plt

from datetime import datetime
from config.config import Config
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
from model.ddim import DiffusionModel
# from model_loader import get_autoencoder
from util.learning_rate_scheduler import create_learning_rate_fn
import util.losses as losses
import dataset_loader
from pathlib import Path
from train_autoencoder import TrainState as AutoEncoderTrainState
from train_ddim import TrainState as DDIMTrainState



def get_autoencoder(rng: jax.random.PRNGKey) -> AutoEncoder:
    config = {
        "AE_block_depths": 2,
        "AE_embed_size_K": 1024,
        "AE_embed_dim_D": 16,
        "AE_commitment_loss_beta": 1.0,
        "AE_learning_rate": 0.001,
        "AE_weight_decay": 0.01,
        "AE_ema_momentum": 0.999,
        "output_root_dir": "/home/dominik.kranz/ecgen/output/VQ-VAE/",
        "checkpoint_dir": "checkpoints/",

    }
    model = AutoEncoder(
        block_depths=config["AE_block_depths"],
        embed_size_K=config["AE_embed_size_K"],
        embed_dim_D=config["AE_embed_dim_D"],
        commitment_loss_beta=config["AE_commitment_loss_beta"]
        )
    rng_params, rng = jax.random.split(rng)
    dummy_ecg = jnp.ones((64, 2048), dtype=jnp.float32)
    variables = model.init(rng_params, dummy_ecg, train=True)

    tx = optax.adamw(learning_rate=config["AE_learning_rate"],
                        weight_decay=config["AE_weight_decay"])
    
    init_state = AutoEncoderTrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=tx,
        dropout_rng = rng,
        # batch_stats=variables["batch_stats"],
        ema_params=None,
        ema_momentum=config["AE_ema_momentum"]
    )
    path = config["output_root_dir"] + config["checkpoint_dir"]
    return flax.training.checkpoints.restore_checkpoint(ckpt_dir=path, target=init_state, step=47)



def get_ddim(rng: jax.random.PRNGKey) -> DiffusionModel:
    config = {
        "DDIM_convolution_filters": [128, 128, 128, 128, 128],
        "DDIM_batch_dims": [3200, 16],
        "DDIM_block_depths": 2,
        "DDIM_learning_rate": 0.0001,
        "DDIM_weight_decay": 0.001,
        "DDIM_ema_momentum": 0.99,
        "checkpoint_dir": "/home/dominik.kranz/ecgen/output/ddim/checkpoints/101/"
    }
    model = DiffusionModel(
        feature_sizes=config["DDIM_convolution_filters"],
        block_depths=config["DDIM_block_depths"],
    )

    rng_init, rng_params = jax.random.split(rng)

    dummy_batch = jnp.ones((1, config["DDIM_batch_dims"][0], config["DDIM_batch_dims"][1]), dtype=jnp.float32)
    variables = model.init(rng_init, dummy_batch, rng_params, train=False)

    tx = optax.adamw(learning_rate=config["DDIM_learning_rate"], weight_decay=config["DDIM_weight_decay"])


    ddim_state = DDIMTrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=tx,
        dropout_rng = rng,
        ema_params=variables["params"],
        ema_momentum=config["DDIM_ema_momentum"]
    )

    path = config["checkpoint_dir"]
    return flax.training.checkpoints.restore_checkpoint(ckpt_dir=path, target=ddim_state, step=101)




if __name__ == "__main__":
    rng = jax.random.PRNGKey(2)
    parser = argparse.ArgumentParser(description="Produce samples from a trained model")
    parser.add_argument("--config", type=str, default="config/inference_config.yml", help="Path to the config file")

    ae_state = get_autoencoder(rng)
    ddim_state = get_ddim(rng)


    generated_batch = ddim_state.apply_fn({"params": ddim_state.params}, rng, 1, method=DiffusionModel.generate) #(1, 3200, 16)
    data_size, data_length, _ = generated_batch.shape
    batched = jnp.array_split(generated_batch[0], data_length // (32))
    batched = jnp.array(batched)
    batched = jnp.squeeze(batched)

    embedded = ae_state.apply_fn({"params": ae_state.params}, batched, method=AutoEncoder.embed)
    generated_ecg = ae_state.apply_fn({"params": ae_state.params}, embedded, method=AutoEncoder.decode) #(16, ?)

    flat = generated_ecg.flatten()
    #save to csv
    onp.savetxt("test.csv", flat, delimiter=",")
    plt.plot(generated_ecg.flatten()[0:2048])
    plt.savefig("test.png")
    print(generated_ecg.shape)

    quit()