import argparse
from functools import partial
import flax
import numpy as onp
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
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
from ecgen.model.ddim_small import DiffusionModelSmall as DiffusionModel
# from model_loader import get_autoencoder
from util.learning_rate_scheduler import create_learning_rate_fn
import util.losses as losses
import dataset_loader
from pathlib import Path
from train_autoencoder import TrainState as AutoEncoderTrainState
from ecgen.train_ddim_small import TrainState as DDIMTrainState



def get_autoencoder(rng: jax.random.PRNGKey) -> AutoEncoder:
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
    path = config["output_root_dir"] + config["checkpoint_dir"] + "25"
    return flax.training.checkpoints.restore_checkpoint(ckpt_dir=path, target=init_state, step=25)



def get_ddim(rng: jax.random.PRNGKey) -> DiffusionModel:
    config = {
        "DDIM_convolution_filters": [128, 128, 128, 128, 128],
        "DDIM_batch_dims": [480, 16],
        "DDIM_block_depths": 2,
        "DDIM_learning_rate": 0.0001,
        "DDIM_weight_decay": 0.001,
        "DDIM_ema_momentum": 0.99,
        "checkpoint_dir":"/home/dominik.kranz/data/checkpoints/ecgen/small/"
    }
    model = DiffusionModel(
        feature_sizes=config["DDIM_convolution_filters"],
        block_depths=config["DDIM_block_depths"],
    )

    rng_init, rng_params = jax.random.split(rng)

    dummy_batch = jnp.ones((1, config["DDIM_batch_dims"][0], config["DDIM_batch_dims"][1]), dtype=jnp.float32)
    dummy_labels = jnp.ones((1,), dtype=jnp.int32)
    variables = model.init(rng_init, dummy_batch, dummy_labels, rng_params, train=False)

    tx = optax.adamw(learning_rate=config["DDIM_learning_rate"], weight_decay=config["DDIM_weight_decay"])


    ddim_state = DDIMTrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=tx,
        dropout_rng = rng,
        ema_params=variables["params"],
        ema_momentum=config["DDIM_ema_momentum"]
    )

    path = config["checkpoint_dir"] + "50/"
    return flax.training.checkpoints.restore_checkpoint(ckpt_dir=path, target=ddim_state, step=50)




def get_batch(rhythm="sinus"):
    """
    returns 100 ECGs
    """
    rng = jax.random.PRNGKey(69)
    parser = argparse.ArgumentParser(description="Produce samples from a trained model")
    parser.add_argument("--config", type=str, default="config/inference_config.yml", help="Path to the config file")

    ae_state = get_autoencoder(rng)
    ddim_state = get_ddim(rng)

    labels = jnp.ones((100,), dtype=jnp.int32) if rhythm == "sinus" else jnp.zeros((100,), dtype=jnp.int32)
    # labels = jnp.array([0, 0, 0, 1, 1, 1])
    generated_batch = ddim_state.apply_fn({"params": ddim_state.params}, rng, labels.shape[0], labels, method=DiffusionModel.generate) #(1, 3200, 16)
    data_size, data_length, _ = generated_batch.shape # (100, 480, 16)
    
    # batched = jnp.reshape(generated_batch, (data_size, 6, 80, 16))
    batched = jnp.reshape(generated_batch, (-1, 80, 16))
    embedded = ae_state.apply_fn({"params": ae_state.params}, batched, method=AutoEncoder.embed)
    generated_ecg = ae_state.apply_fn({"params": ae_state.params}, embedded, method=AutoEncoder.decode) #(16, ?)

    generated_ecg = jnp.reshape(generated_ecg, (data_size, 30 * 512))
    print(generated_ecg.shape)
    # quit()
    return generated_ecg.squeeze()
    #batched = jnp.array_split(generated_batch[0], data_length // (80))
    # batched = jnp.array(batched)
    # batched = jnp.squeeze(batched)
    # print(batched.shape)
    # quit()
    
    
    # for i in range(data_size):
    #     codes = batched[i]
        

    #     embedded = ae_state.apply_fn({"params": ae_state.params}, codes, method=AutoEncoder.embed)
    #     generated_ecg = ae_state.apply_fn({"params": ae_state.params}, embedded, method=AutoEncoder.decode) #(16, ?)

    #     flat = generated_ecg.flatten()
    #     ecgs.append(flat)
        
    #save to csv
    #onp.savetxt("test.csv", flat, delimiter=",")
    
    
    # fig = plt.figure(figsize=(30, 36))
    # fig.suptitle("Conditionally Generated ECGs", fontsize=32, fontweight="bold")
    # for i, ecg in enumerate(ecgs):
    #     # print(ecg.shape)
    #     plt.subplot(6, 1, i+1)
    #     plt.plot(ecg)
    #     # print(generated_ecg.shape)
    # fig.text(0.5, 0.04, "Samples", ha="center", fontsize=24)
    # fig.text(0.04, 0.5, "AU", va="center", fontsize=24, rotation="vertical")
    # # fig.tight_layout()
    # plt.savefig("test.png")
    
    
if __name__ == "__main__":
    sinus_ecgs = []
    for i in tqdm(range(100)):
        sinus_ecgs.append(get_batch("sinus"))
    #save as npy
    sinus_ecgs = jnp.array(sinus_ecgs)
    sinus_ecgs = sinus_ecgs.reshape(-1, 30 * 512)
    print(sinus_ecgs.shape)
    onp.save("/home/dominik.kranz/data/ecg/inhouse_afib_dataset/generated/sinus_ecgs.npy", sinus_ecgs)
    #("/home/dominik.kranz/data/ecg/inhouse_afib_dataset/generated/sinus_ecgs.parquet")
    
    
    afib_ecgs = []
    for i in tqdm(range(100)):
        afib_ecgs.append(get_batch("afib"))
    #save as npy
    afib_ecgs = jnp.array(afib_ecgs)
    afib_ecgs = afib_ecgs.reshape(-1, 30 * 512)
    print(afib_ecgs.shape)
    onp.save("/home/dominik.kranz/data/ecg/inhouse_afib_dataset/generated/afib_ecgs.npy", afib_ecgs)
    # get_batch("afib")
