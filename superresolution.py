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
from model.ddim_small import DiffusionModelSmall as DiffusionModel
# from model_loader import get_autoencoder
from util.learning_rate_scheduler import create_learning_rate_fn
import util.losses as losses
import dataset_loader
from pathlib import Path
from train_autoencoder import TrainState as AutoEncoderTrainState
from train_ddim_small import TrainState as DDIMTrainState
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



def superresolution(ecg, ddim_state, ae_state):
    rng = jax.random.PRNGKey(2)
    ecg = ecg.reshape(1, -1)
    #ecg is in 64 H, sample and hold to 512
    #ecg = jnp.repeat(ecg, 8).reshape(1, -1)
    # print(ecg.shape)
    #encode into latent space
    z_q, _ = ae_state.apply_fn({"params": ae_state.params}, ecg, method=AutoEncoder.encode) # (B, 480, 16)
    # decoded = ae_state.apply_fn({"params": ae_state.params}, z_q, method=AutoEncoder.decode)
    # print(decoded.shape)
    # plt.figure(figsize=(30, 12))
    # plt.plot(ecg[0])
    # plt.plot(decoded[0])
    # plt.savefig("decoded.png")
    # quit()
    # print(z_q.shape)
    #add a bit of noise
    # print(z_q.shape)
    # quit()
    noise = jax.random.normal(rng, z_q.shape, dtype=z_q.dtype)
    # print(noise.shape)
    noise_1 = 0.8 * z_q + 0.2 * noise
    noise_2 = 0.5 * z_q + 0.5 * noise
    noise_3 = 0.2 * z_q + 0.8 * noise
    
    #plot tokens and noisy tokens
    # plt.figure(figsize=(30, 12))
    # plt.imshow(z_q[0].T, aspect="auto")
    # plt.colorbar()
    # plt.savefig("z_q.png")
    # plt.figure(figsize=(30, 12))
    # plt.imshow(noise_1[0].T, aspect="auto")
    # plt.colorbar()
    # plt.savefig("z_q_noise1.png")
    # plt.figure(figsize=(30, 12))
    # plt.imshow(noise_2[0].T, aspect="auto")
    # plt.colorbar()
    # plt.savefig("z_q_noise2.png")
    # plt.figure(figsize=(30, 12))
    # plt.imshow(noise_3[0].T, aspect="auto")
    # plt.colorbar()
    # plt.savefig("z_q_noise3.png")
    
    
    # print(noise.shape)
    
    #generate from noise
    generated = ddim_state.apply_fn({"params": ddim_state.params}, noise_2, step_offset=0.5, method=DiffusionModel.generate_from_noise)
    # print(z_q.shape)
    # quit()
    #generated = ddim_state.apply_fn({"params": ddim_state.params}, rng, 1, None, method=DiffusionModel.generate)
    # generated = ddim_state.apply_fn
    #generated = generated.reshape(-1, 80, 16)
    # print(generated.shape)
    #decode
    generated = ae_state.apply_fn({"params": ae_state.params}, generated, method=AutoEncoder.embed)
    generated = ae_state.apply_fn({"params": ae_state.params}, generated, method=AutoEncoder.decode)
    generated = generated.reshape(1, -1)
    return generated


if __name__ == "__main__":
    #load models
    rng = jax.random.PRNGKey(1)
    ae_state = get_autoencoder(rng)
    ddim_state = get_ddim(rng)
    
    #load ecgs
    df = pd.read_parquet("/home/dominik.kranz/data/ecg/inhouse_afib_dataset/sinus/sinus_segments_22.parquet")
    ecgs = df["ecg_processed"]
    
    
    misdetected_cases_traditional = 0
    misdetected_cases_diffusion = 0
    mean_diffs_traditional = []
    mean_diffs_diffusion = []
    for ecg in tqdm(ecgs):
        try:
            # ecg = ecgs[1]
            #min max normalize
            ecg = (ecg - ecg.min()) / (ecg.max() - ecg.min() + 1e-6)
            #downsample to 64 hz
            ecg64hz = nk.signal_resample(ecg, sampling_rate=512, desired_sampling_rate=64, method="interpolation")
            ecg_upsampled_traditional = nk.signal_resample(ecg64hz, sampling_rate=64, desired_sampling_rate=512, method="interpolation")
            ecg_upsampled_diffusion = superresolution(ecg_upsampled_traditional, ddim_state, ae_state)
            ecg_upsampled_diffusion = jnp.squeeze(ecg_upsampled_diffusion)
            print(ecg_upsampled_diffusion.shape)
            # quit()
            #plot
            # sample_and_hold = jnp.repeat(ecg64hz, 8)
            # plt.figure(figsize=(30, 12))
            # plt.plot(ecg_upsampled_diffusion)
            # plt.plot(ecg, alpha=0.5)
            # plt.plot(ecg_upsampled_traditional, alpha=0.5)
            # plt.savefig("test_superresolution.png")
            
            # #zoomed version of the plot
            # plt.figure(figsize=(30, 12))
            # plt.plot(ecg_upsampled_diffusion[0:500])
            # plt.plot(ecg[0:500], alpha=0.5)
            # plt.plot(ecg_upsampled_traditional[0:500], alpha=0.5)
            # plt.savefig("test_superresolution_zoomed.png")
            # print(ecg.shape)
            
            #!Detect r-peaks
            
            r_peaks_ground_truth = nk.ecg_findpeaks(ecg, sampling_rate=512)["ECG_R_Peaks"] / 512 * 1000
            r_peaks_traditional = nk.ecg_findpeaks(ecg_upsampled_traditional, sampling_rate=512)["ECG_R_Peaks"] / 512 * 1000
            r_peaks_diffusion = nk.ecg_findpeaks(ecg_upsampled_diffusion, sampling_rate=512)["ECG_R_Peaks"] / 512 * 1000
            
            
            #check if lengths differ
            if len(r_peaks_ground_truth) != len(r_peaks_traditional):
                print("Lengths differ")
                misdetected_cases_traditional += 1
                continue
            
            if len(r_peaks_ground_truth) != len(r_peaks_diffusion):
                print("Lengths differ")
                misdetected_cases_diffusion += 1
                continue
            
            diff_traditional = abs(r_peaks_traditional - r_peaks_ground_truth)
            diff_diffusion = abs(r_peaks_diffusion - r_peaks_ground_truth)
            
            #filter out NaNs
            diff_traditional = diff_traditional[~pd.isnull(diff_traditional)]
            diff_diffusion = diff_diffusion[~pd.isnull(diff_diffusion)]
            
            #check if any diff is larger than 20ms, if so we add it to the misdetected cases
            if (diff_traditional > 20).any():
                print("Traditional method misdetected")
                misdetected_cases_traditional += 1
                continue
            if (diff_diffusion > 20).any():
                print("Diffusion method misdetected")
                misdetected_cases_diffusion += 1
                continue
            
            mean_diffs_traditional.append(diff_traditional.mean())
            mean_diffs_diffusion.append(diff_diffusion.mean())
        except:
            print("Error in processing")
            continue
        
    
    
    #save lists to csv
    onp.savetxt("mean_diffs_traditional.csv", mean_diffs_traditional, delimiter=",")
    onp.savetxt("mean_diffs_diffusion.csv", mean_diffs_diffusion, delimiter=",")
    
    #make histograms and print
    plt.figure()
    plt.hist(mean_diffs_traditional, bins=50)
    plt.title("Traditional method")
    plt.savefig("hist_traditional.png")
    plt.figure()
    plt.hist(mean_diffs_diffusion, bins=50)
    plt.title("Diffusion method")
    plt.savefig("hist_diffusion.png")
    
    print(f"Traditional method misdetected: {misdetected_cases_traditional}")
    print(f"Diffusion method misdetected: {misdetected_cases_diffusion}")
    print(f"Traditional method mean diff: {sum(mean_diffs_traditional) / len(mean_diffs_traditional)}")
    print(f"Diffusion method mean diff: {sum(mean_diffs_diffusion) / len(mean_diffs_diffusion)}")
    
        
        
        
        
        
        
        