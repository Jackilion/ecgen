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
    rng = jax.random.PRNGKey(42)
    autoencoder = get_autoencoder(rng)    
    ddim = get_ddim(rng)
    # print(autoencoder)
    
    df = pd.read_parquet("/home/dominik.kranz/data/ecg/inhouse_afib_dataset/sinus_90s/sinus_segments_22.parquet")
    ecgs= df["ecg_processed"]
    ecgs = ecgs.to_numpy()
    #batch into 100
    ecgs = onp.array_split(ecgs, 100)
    print(type(ecgs))
    print(len(ecgs)) #100
    print(ecgs[0].shape) #(100,)
    print(ecgs[0][0].shape) #(46080,)
    #reshape to (100, 46080)
    heart_rates = []
    sdnns = []
    for ecg_batch in tqdm(ecgs):
        ecg = onp.array([onp.array(ecg) for ecg in ecg_batch])
        #min max normalize along axis 1 but keep dims
        ecg = (ecg - ecg.min(axis=1, keepdims=True)) / (ecg.max(axis=1, keepdims=True) - ecg.min(axis=1, keepdims=True))
        #ecg = onp.expand_dims(ecg, axis=-1)
        #print(ecg.shape)
        # print(ecg.shape)
        # quit()
        z_q, _ = autoencoder.apply_fn({"params": autoencoder.params}, ecg, method=AutoEncoder.encode)
        
        # print(z_q.shape)
    
    
        #mask = 1 for 480 samples, 0 for 480 samples, 1 for 480 samples
        mask = jnp.concatenate([jnp.ones((100,480,16)), jnp.zeros((100,480,16)), jnp.ones((100,480,16))], axis=1)
        # mask = mask.reshape(1, -1)
        # print(mask.shape)
        
        #inpaint
        inpainted = ddim.apply_fn({"params": ddim.params}, z_q, mask, rng, steps=29, step_offset=0.0, method=DiffusionModel.inpaint)
        # print(inpainted.shape)
        #decode
        decoded = autoencoder.apply_fn({"params": autoencoder.params}, inpainted, method=AutoEncoder.decode)
        # print(decoded.shape)
        decoded = jnp.squeeze(decoded)
        
        mask_decoded = jnp.concatenate([jnp.ones((100,15360)), jnp.zeros((100, 15360)), jnp.ones((100, 15360))], axis=1)
        decoded = ecg * mask_decoded + decoded * (1 - mask_decoded)
        
        #!qunatitative analyis
        #!We compute the heart rate on the left and right side of the inpainted region, as well as the inpainted section
        for ecg_decoded in decoded:
            # print(ecg_decoded.shape)
            # quit()
            #!left side
            left_side = ecg_decoded[:15360]
            #rpeaks
            rpeaks_left = nk.ecg_findpeaks(left_side, sampling_rate=512)["ECG_R_Peaks"]
            
            #heart rate
            rri_left = onp.diff(rpeaks_left) / 512 * 1000
            
            meanNN_left = onp.mean(rri_left)
            sdnn_left = onp.std(rri_left)
            heart_rate_left = 60000 / meanNN_left


            
            #!right side
            right_side = ecg_decoded[2*15360:]
            #rpeaks
            rpeaks_right = nk.ecg_findpeaks(right_side, sampling_rate=512)["ECG_R_Peaks"]
            
            #heart rate
            rri_right = onp.diff(rpeaks_right) / 512 * 1000
            
            meanNN_right = onp.mean(rri_right)
            sdnn_right = onp.std(rri_right)
            heart_rate_right = 60000 / meanNN_right
            
            #!inpainted region
            inpainted_region = ecg_decoded[15360:2*15360]
            #rpeaks
            rpeaks_inpainted = nk.ecg_findpeaks(inpainted_region, sampling_rate=512)["ECG_R_Peaks"]
            
            #heart rate
            rri_inpainted = onp.diff(rpeaks_inpainted) / 512 * 1000
            
            meanNN_inpainted = onp.mean(rri_inpainted)
            sdnn_inpainted = onp.std(rri_inpainted)
            heart_rate_inpainted = 60000 / meanNN_inpainted
            
            heart_rates.append((heart_rate_left, heart_rate_inpainted,  heart_rate_right))
            sdnns.append((sdnn_left, sdnn_inpainted, sdnn_right))
        # break
            
    #save the heart rates and sdnns to csv
    heart_rates = onp.array(heart_rates)
    sdnns = onp.array(sdnns)
    df = pd.DataFrame(heart_rates, columns=["heart_rate_left", "heart_rate_inpainted", "heart_rate_right"])
    df.to_csv("output/inpainting/quantitative/heart_rates.csv", index=False)
    df = pd.DataFrame(sdnns, columns=["sdnn_left", "sdnn_inpainted", "sdnn_right"])
    df.to_csv("output/inpainting/quantitative/sdnns.csv", index=False)
    #plot the heart rate differences between left and right as histogram
    
    print(heart_rates.shape)
    print(heart_rates)
    #make same bins for all histograms
    bins = onp.linspace(-40, 40, 100)
    plt.figure(figsize=(11.69,8.27))
    plt.hist(heart_rates[:,0] - heart_rates[:,2], bins=bins, density=True)
    plt.xlabel("Heart Rate Difference [bpm]")
    plt.ylabel("Frequency")
    plt.title("Heart Rate Difference between Left and Right Side")
    plt.xlim(-40, 40)
    plt.savefig("output/inpainting/quantitative/heart_rate_diff.png")
    
    plt.figure(figsize=(11.69,8.27))
    plt.hist(heart_rates[:,0] - heart_rates[:,1], bins=bins, density=True)
    plt.xlabel("Heart Rate Difference [bpm]")
    plt.ylabel("Frequency")
    plt.title("Heart Rate Difference between Left and Inpainted Side")
    plt.xlim(-40, 40)
    plt.savefig("output/inpainting/quantitative/heart_rate_diff_left_inpainted.png")
    
    plt.figure(figsize=(11.69,8.27))
    plt.hist(heart_rates[:,1] - heart_rates[:,2], bins=bins, density=True)
    plt.xlabel("Heart Rate Difference [bpm]")
    plt.ylabel("Frequency")
    plt.xlim(-40, 40)
    plt.title("Heart Rate Difference between Inpainted and Right Side")
    plt.savefig("output/inpainting/quantitative/heart_rate_diff_right_inpainted.png")
        
    #same for sdnn
    bins = onp.linspace(-200, 200, 100)
    plt.figure(figsize=(11.69,8.27))
    plt.hist(sdnns[:,0] - sdnns[:,2], bins=bins, density=True)
    plt.xlabel("SDNN Difference")
    plt.ylabel("Frequency")
    plt.title("SDNN Difference between Left and Right Side")
    # plt.xlim(200, 200)
    plt.savefig("output/inpainting/quantitative/sdnn_diff.png")
    
    plt.figure(figsize=(11.69,8.27))
    plt.hist(sdnns[:,0] - sdnns[:,1], bins=bins, density=True)
    plt.xlabel("SDNN Difference")
    plt.ylabel("Frequency")
    plt.title("SDNN Difference between Left and Inpainted Side")
    # plt.xlim(-200, 200)
    plt.savefig("output/inpainting/quantitative/sdnn_diff_left_inpainted.png")
    
    plt.figure(figsize=(11.69,8.27))
    plt.hist(sdnns[:,1] - sdnns[:,2], bins=bins, density=True)
    plt.xlabel("SDNN Difference")
    plt.ylabel("Frequency")
    plt.title("SDNN Difference between Inpainted and Right Side")
    # plt.xlim(-200, 200)
    plt.savefig("output/inpainting/quantitative/sdnn_diff_right_inpainted.png")
