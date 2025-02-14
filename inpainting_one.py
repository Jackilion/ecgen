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
# from model.ddim_small import DiffusionModelSmall as DiffusionModel
from model.ddim_medium import DiffusionModelMedium as DiffusionModel
# from model_loader import get_autoencoder
from util.learning_rate_scheduler import create_learning_rate_fn
import util.losses as losses
import dataset_loader
from pathlib import Path
from train_autoencoder import TrainState as AutoEncoderTrainState
# from train_ddim_small import TrainState as DDIMTrainState
from train_ddim_medium import TrainState as DDIMTrainState
import neurokit2 as nk
from scipy.signal import butter, filtfilt


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
        "DDIM_convolution_filters": [128, 128, 128, 128, 128, 128],
        "DDIM_batch_dims": [1440, 16],
        "DDIM_block_depths": 2,
        "DDIM_learning_rate": 0.0001,
        "DDIM_weight_decay": 0.001,
        "DDIM_ema_momentum": 0.99,
        "checkpoint_dir":"/home/dominik.kranz/data/checkpoints/ecgen/medium/"
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

    path = config["checkpoint_dir"] + "40/"
    return flax.training.checkpoints.restore_checkpoint(ckpt_dir=path, target=ddim_state, step=40)



if __name__ == "__main__":
    rng = jax.random.PRNGKey(69)
    autoencoder = get_autoencoder(rng)    
    ddim = get_ddim(rng)
    # print(autoencoder)
    
    df = pd.read_parquet("/home/dominik.kranz/data/ecg/inhouse_afib_dataset/sinus_90s/sinus_segments_22.parquet")
    ecgs= df["ecg_processed"]
    ecg = ecgs[0]
    
    ecg = (ecg - ecg.min()) / (ecg.max() - ecg.min() + 1e-6)
    ecg = jnp.array(ecg)
    ecg = jnp.expand_dims(ecg, axis=0)
    #ecg = onp.expand_dims(ecg, axis=-1)
    #print(ecg.shape)
    #print(ecg.shape)
    #quit()
    # quit()
    z_q, _ = autoencoder.apply_fn({"params": autoencoder.params}, ecg, method=AutoEncoder.encode)
    
    print(z_q.shape)

    #repeat the tokens 100 times
    z_q = jnp.repeat(z_q, 256, axis=0)
    #mask = 1 for 480 samples, 0 for 480 samples, 1 for 480 samples
    mask = jnp.concatenate([jnp.ones((256,480,16)), jnp.zeros((256,480,16)), jnp.ones((256,480,16))], axis=1)
    # mask = mask.reshape(1, -1)
    print(mask.shape)
    
    #inpaint
    inpainted = ddim.apply_fn({"params": ddim.params}, z_q, mask, rng, steps=29, step_offset=0.0, method=DiffusionModel.inpaint)
    print(inpainted.shape)
    #decode
    decoded = autoencoder.apply_fn({"params": autoencoder.params}, inpainted, method=AutoEncoder.decode)
    print(decoded.shape)
    decoded = jnp.squeeze(decoded)
    inpainted_part = decoded[:, 15360: 2*15360]
    #average across batch dimension
    inpainted_part_mean= jnp.mean(inpainted_part, axis=0)
    inpainted_part_std = jnp.std(inpainted_part, axis=0)
    
    #low pass filter the mean 
    #everything over 0.5 Hz is removed
    b, a = butter(3, 5.0, fs=512, btype="low")
    inpainted_part_mean = filtfilt(b, a, inpainted_part_mean)
    inpainted_part_std = filtfilt(b, a, inpainted_part_std)
    
    
    
    print(inpainted_part_mean.shape)
    print(inpainted_part_std.shape)
    
    
    mask_decoded = jnp.concatenate([jnp.ones((15360)), jnp.zeros((15360)), jnp.ones((15360))], axis=0)
    print(mask_decoded.shape)
    inpainted_part_mean_conc = jnp.concatenate([jnp.zeros(15360), inpainted_part_mean, jnp.zeros(15360)], axis=0)
    print(inpainted_part_mean_conc.shape)
    decoded = ecg.squeeze() * mask_decoded + inpainted_part_mean_conc * (1 - mask_decoded)
    print(decoded.shape)
    std = jnp.concatenate([jnp.zeros(15360), inpainted_part_std, jnp.zeros(15360)], axis=0)
    print(std.shape)
    
    decoded = decoded[10360: 35720]
    std = std[10360: 35720]
    #plot with std as shaded area
    plt.figure(figsize=(50, 12))
    plt.title("Visualization of Temporal Coherence of Inpainting", weight="bold", fontsize=55)
    plt.ylabel("Normalized Amplitude[AU]", fontsize=30)
    plt.xlabel("Sample", fontsize=30)
    plt.plot(decoded)
    plt.fill_between(range(25360), decoded - 1*std, decoded + 1*std, color='b', alpha=0.2)
    #vertical lines to indicate inpainted part
    plt.axvline(x=5000, color='red', linestyle='--')
    plt.axvline(x=20360, color='red', linestyle='--')
    plt.tight_layout()
    # plt.fill_between(range(15360, 2*15360), inpainted_part_mean - 1*inpainted_part_std, inpainted_part_mean + 1*inpainted_part_std, color='r', alpha=0.2)
    plt.savefig("output/inpainting/quantitative/shaded.png")
    
    quit()
    
    
    
