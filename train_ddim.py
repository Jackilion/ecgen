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
from util.data_loader import load_tokenised_dataset


def get_autoencoder(rng):
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

    variables = model.init(rng_init, dummy_batch, rng_params, train=True)

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


@partial(jax.jit, static_argnums=3)
def train_step(state, batch, rng, learning_rate_fn, dropout_rng):
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
            batch, rng, train=True, rngs={'dropout': dropout_train_key}
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
    for epoch in range(config["DDIM_epochs"]):
        for i in load_tokenised_dataset(config["tokenised_dataset_root"]):
            # (64, 80, 32, 16)
            batch = i.reshape(64, 80 * 32, 16) #(64 batches of tokens for 320s ECGs)
            rng, train_step_rng, dropout_rng = jax.random.split(rng, num=3)
            ddim_state, loss, lr = train_step(ddim_state, batch, train_step_rng, learning_rate_fn, dropout_rng)
        print(f"EPOCH {epoch} Loss: {loss}, Lr: {lr}")
        
        ddim_state = ddim_state.replace(epoch=epoch)
        evaluate(batch, ddim_state, autoencoder_state, rng)
        checkpoints.save_checkpoint(ckpt_dir=f'{config["checkpoint_dir"]}{epoch}/', target=ddim_state, step=epoch)

    quit()

    rng, state_rng = jax.random.split(rng)
    learning_rate_fn = create_annealing_learning_rate_fn(config["DDIM_epochs"], batched_dataset_test.shape[0])
    ddim_state = create_train_state(state_rng, learning_rate_fn)
    
    rng, ae_rng = jax.random.split(rng)
    ae_state = get_autoencoder(ae_rng)
    
    
    for epoch in range(config.DDIM_epochs):
        pbar = tqdm(range(len(batched_dataset)), desc=f'Epoch {epoch}')
        for i in pbar:
            
            batch = batched_dataset[i]
            rng, train_step_rng = jax.random.split(rng)
            
            ddim_state, loss, lr = train_step(ddim_state, batch, train_step_rng, learning_rate_fn)
            pbar.set_postfix({"Loss": f"{loss:.5f}", "Lr": f"{lr:.5f}"})
        ddim_state = ddim_state.replace(epoch=epoch)
        rng, eval_rng = jax.random.split(rng)
        evaluate(batch, ddim_state, ae_state, eval_rng)
        checkpoints.save_checkpoint(ckpt_dir=config.checkpoint_dir, target=ddim_state, step=epoch)
        
    return ddim_state
  

def evaluate(batch, ddim_state, ae_state, rng):
    config = Config().settings
    print(f"Batch shape: {batch.shape}")
    ddim_variables={"params": ddim_state.params}
    
    rng, gen_rng = jax.random.split(rng)
    generated_batch = ddim_state.apply_fn(ddim_variables, gen_rng, 1, method=DiffusionModel.generate)
    
    print(f"generated_batch shape: {generated_batch.shape}")
    data_size, data_length, _ = generated_batch.shape
    #generated_ints =jnp.rint(64*generated_batch)
    #generated_ints = jnp.array(generated_ints, dtype=jnp.int16)
    #batched = jnp.array_split(generated_batch[0], data_length // 256)
    #batched = jnp.array(batched)
    batched = jnp.array_split(generated_batch[0], data_length // (32))
    batched = jnp.array(batched)
    batched = jnp.squeeze(batched)
    
    print(f"Batched shape: {batched.shape}")
    ae_variables = {"params": ae_state.params}
    #print(generated_ints.shape)
    #print(generated_ints[0])
    #embed_vectors = ae_state.apply_fn(ae_variables, batched, method=AutoEncoder.embed_indices)
    embedded = ae_state.apply_fn(ae_variables, batched, method=AutoEncoder.embed)
    generated_ecg = ae_state.apply_fn(ae_variables, embedded, method= AutoEncoder.decode) #64, 2048
    
    sample_batch = batch[0]
    print(sample_batch.shape)
    sample_batch = sample_batch.reshape((80, 32, 16))
    #sample_batch_inted = jnp.rint(64*sample_batch)
    #sample_batch_inted = jnp.array(sample_batch_inted, dtype=jnp.int16)
    #batch_embedded = ae_state.apply_fn(ae_variables, sample_batch_inted, method=AutoEncoder.embed_indices)
    batch_decoded = ae_state.apply_fn(ae_variables, sample_batch, method= AutoEncoder.decode) #64, 2048
    #generated_2s_ecgs.append(generated_ecg)
    print(f"generated_ecg shape: {generated_ecg.shape}")
    # real_batch_diffused = ddim_state.apply_fn(ddim_variables, batch, 30, 0.7, method=DiffusionModel.reverse_diffusion)
    # plt.plot(real_batch_diffused[0].flatten())
    # plt.savefig(f"{FLAGS.DDIM_img_dir}/epoch_{ddim_state.epoch}_batch_diffused.png")
    # plt.close()
    
    #Plot the latent space batch
    x = onp.arange(0, len(batch[0].flatten()), step= 1)
    plt.scatter(x, batch[0].flatten(), marker=".")
    plt.title("Flattened latent space of a 320s ECG")
    plt.xlabel("x")
    plt.xlabel("y")
    plt.tight_layout()
    plt.savefig(f"{config['image_dir']}/epoch_{ddim_state.epoch}_batch.png")
    plt.close()
    
    x = onp.arange(0, len(batch[0].flatten()[0:32*8*2]))
    plt.scatter(x, batch[0].flatten()[0:32*8*2], marker=".")
    plt.title("Flattened latent space of a 4s ECG")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(f"{config['image_dir']}/epoch_{ddim_state.epoch}_batch_zoom.png")
    plt.close()
    
    
    x = onp.arange(0, len(embedded[0].flatten()))
    plt.scatter(x, embedded[0].flatten(), marker=".")
    plt.title("Flattened embedded latent space of a 2 ECG")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(f"{config['image_dir']}/epoch_{ddim_state.epoch}_ddim_embedded.png")
    plt.close()
    
    x = onp.arange(0, len(embedded[0].flatten()[0:256]))
    plt.scatter(x, embedded[0].flatten()[0:256], marker=".")
    plt.title("Flattened embedded latent space of a 4 ECG")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(f"{config['image_dir']}/epoch_{ddim_state.epoch}_ddim_embedded_zoom.png")
    plt.close()
    
    

    
    #Decode the latent space batch
    plt.plot(batch_decoded.flatten())
    plt.xlabel("Samples")
    plt.ylabel("AU")
    plt.title("Decoded latent space")
    plt.tight_layout()
    plt.savefig(f"{config['image_dir']}/epoch_{ddim_state.epoch}_batch_decoded.png")
    plt.close()
    
    plt.plot(batch_decoded.flatten()[0:8192])
    plt.title("First 8 seconds of decoded latent space")
    plt.xlabel("Samples")
    plt.ylabel("AU")
    plt.tight_layout()
    plt.savefig(f"{config['image_dir']}/epoch_{ddim_state.epoch}_batch_decoded_zoom.png")
    plt.close()

    #Plot the DDIM output
    
    plt.plot(generated_batch[0].flatten())
    plt.title("Flattened DDM output")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(f"{config['image_dir']}/epoch_{ddim_state.epoch}_ddim.png")
    plt.close()
    
    #Plot the ECGs
    plt.plot(generated_ecg.flatten())
    plt.title("Decoded DDM output")
    plt.xlabel("Samples")
    plt.ylabel("AU")
    plt.tight_layout()
    plt.savefig(f"{config['image_dir']}/epoch_{ddim_state.epoch}.png")
    plt.close()
    
    #zoom into start middle and end
    plt.plot(generated_ecg.flatten()[0:8192])
    plt.xlabel("Samples")
    plt.ylabel("AU")
    plt.title("First 8 seconds of decoded DDM output")
    plt.tight_layout()
    plt.savefig(f"{config['image_dir']}/epoch_{ddim_state.epoch}_zoom_start.png")
    plt.close()
    
    plt.plot(generated_ecg.flatten()[50000:58000])
    plt.xlabel("Samples")
    plt.ylabel("AU")
    plt.title("8 seconds of decoded DDM output")
    plt.tight_layout()
    plt.savefig(f"{config['image_dir']}/epoch_{ddim_state.epoch}_zoom_middle.png")
    plt.close()
    
    plt.plot(generated_ecg.flatten()[120000:128000])
    plt.xlabel("Samples")
    plt.ylabel("AU")
    plt.title("Last 8 seconds of decoded DDM output")
    plt.tight_layout()
    plt.savefig(f"{config['image_dir']}/epoch_{ddim_state.epoch}_zoom_end.png")
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
    parser.add_argument('--config', type=str, default='config/diff_ecg_300ep_512hz.yml', help='Path to the config file')
    args = parser.parse_args()
    config = Config(f"{args.config}")
    main()
    
    
