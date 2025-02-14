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
from model.ddim_small import DiffusionModelSmall as DiffusionModel
# from model_loader import get_autoencoder
from util.learning_rate_scheduler import create_learning_rate_fn
import util.losses as losses
import dataset_loader
from pathlib import Path
from train_autoencoder import TrainState as AutoEncoderTrainState
from util.data_loader import load_afib_tokens, load_tokenised_dataset


def get_autoencoder(rng):
    config = {
        "AE_block_depths": 1,
        "AE_embed_size_K": 1024,
        "AE_embed_dim_D": 16,
        "AE_commitment_loss_beta": 0.8,
        "AE_learning_rate": 0.001,
        "AE_convolution_filters": [32, 32, 32, 32, 32],
        "AE_kernel_sizes": [32, 16, 8, 4, 4],
        "AE_weight_decay": 0.01,
        "AE_ema_momentum": 0.999,
        "AE_dropout": 0.1,
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
    dummy_ecg = jnp.ones((64, 5 * 512), dtype=jnp.float32)
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
    path = config["output_root_dir"] + config["checkpoint_dir"] + "25/"
    return flax.training.checkpoints.restore_checkpoint(ckpt_dir=path, target=init_state, step=25)


def main():
    config = Config().settings

    Path(config["output_root_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["image_dir"]).mkdir(parents=True, exist_ok=True)
    # Path(config["DDIM_log_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
    
    ddim_state = train()
    #Evaluate after training
    # evaluate_finished_model(ddim_state)
    
    
class TrainState(train_state.TrainState):
    #batch_stats: Any
    ema_params: Any = None
    ema_momentum: float = None
    dropout_rng: Any = None
    epoch: int = None


def create_train_state(rng, learning_rate_fn):
    """Creates initial TrainState to hold params"""

    config = Config().settings

    model = DiffusionModel(
        feature_sizes=config["DDIM_convolution_filters"],
        block_depths=config["DDIM_block_depth"],
        #attention_depths=config.attention_depths
    )
    rng_init, rng_params = jax.random.split(rng)

    dummy_batch = jnp.ones((1, config["DDIM_batch_dims"][0], config["DDIM_batch_dims"][1]), dtype=jnp.float32)
    dummy_labels = jnp.ones((1,), dtype=jnp.int32)
    variables = model.init(rng_init, dummy_batch, dummy_labels, rng_params, train=True)

    tx = optax.adamw(learning_rate=learning_rate_fn,
                     weight_decay=config["DDIM_weight_decay"])
    return TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=tx,
        dropout_rng = rng,
        ema_params=variables["params"],
        ema_momentum=config["DDIM_ema_momentum"]
    )
    
def compute_ema_params(ema_params, current_params):
    ema_momentum = config["DDIM_ema_momentum"]
    return ema_momentum * ema_params + (1-ema_momentum) * current_params


@partial(jax.jit, static_argnums=4)
def train_step(state, batch, labels, rng, learning_rate_fn, dropout_rng):
    """_summary_

    Args:
        state (_type_): _description_
        batch (_type_): _description_
        rng (_type_): _description_
    """
    dropout_train_key = jax.random.fold_in(key=dropout_rng, data=state.step)
    def compute_loss(params):
        outputs = state.apply_fn(
            {
                "params": params,
                #"batch_stats": state.batch_stats
            },
            batch, labels, rng, train=True, rngs={'dropout': dropout_train_key}
        )        
        
        orig_batch, noises, pred_noises, pred_batch = outputs
        B, L, C = noises.shape
        noises = noises.reshape(B, L, C)
        pred_noises = pred_noises.reshape(B, L, C)
        #loss = jnp.linalg.norm((pred_noises - noises), ord=1, axis=-1).mean()
        #loss = losses.L2(pred_noises, noises).mean()
        loss = losses.L2(pred_noises.reshape((B, -1)), noises.reshape(B, -1)).mean()
        return loss

    grad_fn = jax.value_and_grad(compute_loss)
    loss, grads = grad_fn(state.params)
    
    # print(type(loss))
    # print(type(grads))
    # print(grads)
    # quit()
    #mutated_vars = auxillary_data
    
    new_state = state.apply_gradients(
        grads=grads
    )
    lr = learning_rate_fn(state.step)
    return new_state, loss, lr


def train() -> TrainState:
    config = Config().settings
    tf.config.experimental.set_visible_devices([], "GPU")
    rng = jax.random.PRNGKey(config["DDIM_jax_seed"])

    rng, state_rng = jax.random.split(rng)
    learning_rate_fn = create_learning_rate_fn(config["DDIM_epochs"], 78, config["DDIM_learning_rate"], config["DDIM_learning_rate"], 20)
    ddim_state = create_train_state(state_rng, learning_rate_fn)
    
    autoencoder_state = get_autoencoder(rng)

    #dataset = load_tokenised_dataset(config["tokenised_dataset_root"])
    dataset = load_afib_tokens("/home/dominik.kranz/data/ecg/inhouse_afib_dataset/tokenized/")
    
    

    latents = dataset[0]
    labels = dataset[1]
    #shuffle latents and labels in the same way
    rng, shuffle_rng = jax.random.split(rng)
    indices = jax.random.permutation(shuffle_rng, len(latents))
    latents = latents[indices]
    labels = labels[indices]
    #convert labels to int32
    labels = labels.astype(jnp.int32)
    latents = jnp.array_split(latents, len(latents) // 64)
    labels = jnp.array_split(labels, len(labels) // 64)
    #dataset = list(zip(dataset[0], dataset[1]))
    
    
    for epoch in range(config["DDIM_epochs"]):
        for i in zip(latents, labels):
            batch = i[0]
            batch_labels = i[1]
            # print(batch_labels.dtype)
            # quit()
            # print(batch.shape)
            # print(batch_labels.shape)
            # quit()
            # (64, 80, 32, 16)
            # batch = i.reshape(64, 80 * 32, 16) #(64 batches of tokens for 320s ECGs)
            rng, train_step_rng, dropout_rng = jax.random.split(rng, num=3)
            ddim_state, loss, lr = train_step(ddim_state, batch, batch_labels, train_step_rng, learning_rate_fn, dropout_rng)
        print(f"EPOCH {epoch} Loss: {loss}, Lr: {lr}")
        
        ddim_state = ddim_state.replace(epoch=epoch)
        evaluate(batch, ddim_state, autoencoder_state, rng)
        checkpoints.save_checkpoint(ckpt_dir=f'{config["checkpoint_dir"]}{epoch}/', target=ddim_state, step=epoch)
    print("Training finished")
  

def evaluate(batch, ddim_state, ae_state, rng):
    config = Config().settings
    # print(f"Batch shape: {batch.shape}")
    ddim_variables={"params": ddim_state.params}
    
    rng, gen_rng = jax.random.split(rng)
    #2 labels, one 0 and one 1
    labels = jnp.array([0, 1])
    # print(labels.shape)
    generated_batch = ddim_state.apply_fn(ddim_variables, gen_rng, 2, labels, method=DiffusionModel.generate)
    
    # print(f"generated_batch shape: {generated_batch.shape}")
    data_size, data_length, _ = generated_batch.shape
    #generated_ints =jnp.rint(64*generated_batch)
    #generated_ints = jnp.array(generated_ints, dtype=jnp.int16)
    #batched = jnp.array_split(generated_batch[0], data_length // 256)
    #batched = jnp.array(batched)
    batched_0 = jnp.array_split(generated_batch[0], data_length // (80))
    batched_0 = jnp.array(batched_0)
    batched_0 = jnp.squeeze(batched_0)
    
    batched_1 = jnp.array_split(generated_batch[1], data_length // (80))
    batched_1 = jnp.array(batched_1)
    batched_1 = jnp.squeeze(batched_1)
    #print(f"Batched shape: {batched.shape}")
    
    
    
    # print(f"Batched shape: {batched_0.shape}")
    ae_variables = {"params": ae_state.params}
    #print(generated_ints.shape)
    #print(generated_ints[0])
    #embed_vectors = ae_state.apply_fn(ae_variables, batched, method=AutoEncoder.embed_indices)
    embedded_0 = ae_state.apply_fn(ae_variables, batched_0, method=AutoEncoder.embed)
    generated_ecg_0 = ae_state.apply_fn(ae_variables, embedded_0, method= AutoEncoder.decode) #64, 2048
    
    embedded_1 = ae_state.apply_fn(ae_variables, batched_1, method=AutoEncoder.embed)
    generated_ecg_1 = ae_state.apply_fn(ae_variables, embedded_1, method= AutoEncoder.decode) #64, 2048
    
    sample_batch = batch[0]
    # print(sample_batch.shape)
    sample_batch = sample_batch.reshape((-1, 80, 16))
    #sample_batch_inted = jnp.rint(64*sample_batch)
    #sample_batch_inted = jnp.array(sample_batch_inted, dtype=jnp.int16)
    #batch_embedded = ae_state.apply_fn(ae_variables, sample_batch_inted, method=AutoEncoder.embed_indices)
    batch_decoded = ae_state.apply_fn(ae_variables, sample_batch, method= AutoEncoder.decode) #64, 2048
    #generated_2s_ecgs.append(generated_ecg)
    # print(f"generated_ecg shape: {generated_ecg_0.shape}")
    # real_batch_diffused = ddim_state.apply_fn(ddim_variables, batch, 30, 0.7, method=DiffusionModel.reverse_diffusion)
    # plt.plot(real_batch_diffused[0].flatten())
    # plt.savefig(f"{FLAGS.DDIM_img_dir}/epoch_{ddim_state.epoch}_batch_diffused.png")
    # plt.close()
    
    
    #Plot the latent space batch
    #shape is (64, 240, 16)
    #plot one set of tokens as a heatmap
    plt.figure(figsize=(20, 10))
    plt.imshow(batch[0].T, aspect="auto")
    plt.title("Latent space of a 30s ECG")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(f"{config['image_dir']}/epoch_{ddim_state.epoch}_batch_heatmap.png")
    plt.close()
    
    #plot the first 40 tokens (5 seconds)
    plt.figure(figsize=(20, 10))
    plt.imshow(batch[0][0:80].T, aspect="auto")
    plt.title("First 5 seconds of latent space of a 30s ECG")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(f"{config['image_dir']}/epoch_{ddim_state.epoch}_batch_heatmap_zoom.png")
    plt.close()
    

    #Decode the latent space batch
    plt.figure(figsize=(30, 10))
    plt.plot(batch_decoded.flatten())
    plt.xlabel("Samples")
    plt.ylabel("AU")
    plt.title("Decoded latent space")
    plt.tight_layout()
    plt.savefig(f"{config['image_dir']}/epoch_{ddim_state.epoch}_batch_decoded.png")
    plt.close()


    #Plot the DDIM output as a heatmap
    plt.figure(figsize=(20, 10))
    plt.imshow(generated_batch[0].T, aspect="auto")
    plt.title("DDIM output for category 0")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(f"{config['image_dir']}/epoch_{ddim_state.epoch}_heatmap_0.png")
    plt.close()
    
    plt.figure(figsize=(20, 10))
    plt.imshow(generated_batch[1].T, aspect="auto")
    plt.title("DDIM output for category 1")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(f"{config['image_dir']}/epoch_{ddim_state.epoch}_heatmap_1.png")
    plt.close()
    
    #Plot an category 0 and 1 ECG
    plt.figure(figsize=(40, 15))
    plt.plot(generated_ecg_0.flatten())
    plt.title("Decoded DDM output for category 0")
    plt.xlabel("Samples")
    plt.ylabel("AU")
    plt.tight_layout()
    plt.savefig(f"{config['image_dir']}/epoch_{ddim_state.epoch}_0.png")
    plt.close()
    
    plt.figure(figsize=(40, 15))
    plt.plot(generated_ecg_1.flatten())
    plt.title("Decoded DDM output for category 1")
    plt.xlabel("Samples")
    plt.ylabel("AU")
    plt.tight_layout()
    plt.savefig(f"{config['image_dir']}/epoch_{ddim_state.epoch}_1.png")
    plt.close()
    
    #zoom into start middle and end
    plt.figure(figsize=(40, 30))
    plt.subplot(3, 1, 1)
    plt.plot(generated_ecg_0.flatten()[0:5120])
    plt.xlabel("Samples")
    plt.ylabel("AU")
    plt.suptitle("First 10 seconds of decoded DDM output")
    
    plt.subplot(3, 1, 2)
    plt.plot(generated_ecg_0.flatten()[5120:10240])
    plt.xlabel("Samples")
    plt.ylabel("AU")
    plt.title("Middle 10 seconds of decoded DDM output")
    
    plt.subplot(3, 1, 3)
    plt.plot(generated_ecg_0.flatten()[10240:])
    plt.xlabel("Samples")
    plt.ylabel("AU")
    plt.title("Last 10 seconds of decoded DDM output")
    plt.tight_layout()
    plt.savefig(f"{config['image_dir']}/epoch_{ddim_state.epoch}_zoom_0.png")
    plt.close()
    
    plt.figure(figsize=(40, 30))
    plt.subplot(3, 1, 1)
    plt.plot(generated_ecg_1.flatten()[0:5120])
    plt.xlabel("Samples")
    plt.ylabel("AU")
    plt.suptitle("First 10 seconds of decoded DDM output")
    
    plt.subplot(3, 1, 2)
    plt.plot(generated_ecg_1.flatten()[5120:10240])
    plt.xlabel("Samples")
    plt.ylabel("AU")
    plt.title("Middle 10 seconds of decoded DDM output")
    
    plt.subplot(3, 1, 3)
    plt.plot(generated_ecg_1.flatten()[10240:])
    plt.xlabel("Samples")
    plt.ylabel("AU")
    plt.title("Last 10 seconds of decoded DDM output")
    plt.tight_layout()
    plt.savefig(f"{config['image_dir']}/epoch_{ddim_state.epoch}_zoom_1.png")
    plt.close()
    
    # #random control noise
    # noise_batch = jax.random.normal(rng, (64, 64, 64))
    # noise_batch = 0.2 * noise_batch + 0.5
    # ecg = ae_state.apply_fn(ae_variables, noise_batch, method=AutoEncoder.decode)
    
    # plt.plot(ecg[0])
    # plt.savefig(f"{FLAGS.DDIM_img_dir}/epoch_{ddim_state.epoch}_noise.png")
    # plt.close()

    
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a DDIM model")
    parser.add_argument('--config', type=str, default='config/diff_med_ecg_300ep_512hz.yml', help='Path to the config file')
    args = parser.parse_args()
    config = Config(f"{args.config}")
    main()
    
    
