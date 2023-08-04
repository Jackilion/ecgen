import numpy as onp
import jax.numpy as jnp
import matplotlib.pyplot as plt

from datetime import datetime

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
from model_loader import get_autoencoder
import util.losses as losses
import dataset_loader
from pathlib import Path
from absl import app, flags

FLAGS = flags.FLAGS


#! Training hyperparameter
flags.DEFINE_float("DDIM_learning_rate", 1e-4, "The autoencoders learning rate")
flags.DEFINE_float("DDIM_weight_decay", 0.1, "The autoencoders weight decay")
flags.DEFINE_float("DDIM_ema_momentum", 0.990, "The autoencoders EMA momentum")
flags.DEFINE_integer("DDIM_epochs", 100, "The autoencoders training epochs")

#! Others
flags.DEFINE_integer("DDIM_run_seed", 0, "The seed used to generate JAX prng")
flags.DEFINE_list("DDIM_batch_dims", [256, 1], "The dimensions of a latent space tensor")
flags.DEFINE_string("AE_model_path", None, "The Autoencoder checkpoint path to use for evaluation")
flags.DEFINE_string("DDIM_dataset_path", None, "The autoencoded latent space used for training")

#! Model parameter
flags.DEFINE_list("DDIM_feature_sizes", [64, 64, 64, 64, 64], "The sizes for the unet as a list of features sizes of length n. The unet will have n-1 layers, and the final feature is a resnet block at the bottom")
flags.DEFINE_integer("DDIM_block_depth", 2, "The number of times each resnet block is repeated")

#! Logging flags
now = datetime.now().strftime("%Y%m%d-%H%M%S")
flags.DEFINE_string("DDIM_output_dir", f"./outputs/ddim/{now}", "The output root directory where all the models output will be saved")
flags.DEFINE_string("DDIM_img_dir", f"./outputs/ddim/{now}/images", "The directory where evaluation images will be stored")
flags.DEFINE_string("DDIM_log_dir", f"./outputs/ddim/{now}/logs", "The directory where logs will be stored")
flags.DEFINE_string("DDIM_ckpt_dir", f"./outputs/ddim/{now}/checkpoints", "The directory where model checkpoints will be stored")



def main(argv):
    Path(FLAGS.DDIM_output_dir).mkdir(parents=True, exist_ok=True)
    Path(f"{FLAGS.DDIM_img_dir}").mkdir(parents=True, exist_ok=True)
    Path(f"{FLAGS.DDIM_log_dir}").mkdir(parents=True, exist_ok=True)
    Path(f"{FLAGS.DDIM_ckpt_dir}").mkdir(parents=True, exist_ok=True)
    
    train()
    
class TrainState(train_state.TrainState):
    batch_stats: Any
    ema_params: Any = None
    ema_momentum: float = None
    epoch: int = None


def create_train_state(rng):
    """Creates initial TrainState to hold params"""

    model = DiffusionModel(
        feature_sizes=FLAGS.DDIM_feature_sizes,
        block_depths=FLAGS.DDIM_block_depth,
        #attention_depths=config.attention_depths
    )
    rng_init, rng_params = jax.random.split(rng)

    dummy_batch = jnp.ones((1, FLAGS.DDIM_batch_dims[0], FLAGS.DDIM_batch_dims[1]), dtype=jnp.float32)

    variables = model.init(rng_init, dummy_batch, rng_params, train=True)

    tx = optax.adamw(learning_rate=FLAGS.DDIM_learning_rate,
                     weight_decay=FLAGS.DDIM_weight_decay)
    return TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=tx,
        batch_stats=variables["batch_stats"],
        ema_params=variables["params"],
        ema_momentum=FLAGS.DDIM_ema_momentum
    )
    
def compute_ema_params(ema_params, current_params):
    ema_momentum = FLAGS.DDIM_ema_momentum
    return ema_momentum * ema_params + (1-ema_momentum) * current_params


@jax.jit
def train_step(state, batch, rng):
    """_summary_

    Args:
        state (_type_): _description_
        batch (_type_): _description_
        rng (_type_): _description_
    """
    
    def compute_loss(params):
        outputs, mutated_vars = state.apply_fn(
            {
                "params": params,
                "batch_stats": state.batch_stats
            },
            batch, rng, train=True, mutable=["batch_stats"]
        )        
        
        orig_batch, noises, pred_noises, pred_batch = outputs
        B, L, C = noises.shape
        noises = noises.reshape(B, L, C)
        pred_noises = pred_noises.reshape(B, L, C)
        #loss = jnp.linalg.norm((pred_noises - noises), ord=1, axis=-1).mean()
        #loss = losses.L2(pred_noises, noises).mean()
        loss = losses.L2(pred_noises, noises).mean()
        return loss, mutated_vars

    grad_fn = jax.value_and_grad(compute_loss, has_aux=True)
    (loss, auxillary_data), grads = grad_fn(state.params)
    
    mutated_vars = auxillary_data
    
    new_state = state.apply_gradients(
        grads=grads, batch_stats=mutated_vars['batch_stats']
    )
    
    return new_state, loss


def train() -> TrainState:
    tf.config.experimental.set_visible_devices([], "GPU")
    rng = jax.random.PRNGKey(FLAGS.DDIM_run_seed)
    
    dataset = jnp.load(FLAGS.DDIM_dataset_path)
    print(dataset.shape) #e.g. 8640, 64, 256, 1 = (N, B), (LSD1, LSD2)
    dataset = jnp.reshape(dataset, (8640, -1, 1)) #(8640, 16384, 1)
    data_size, data_length, _ = dataset.shape
    batched_dataset = jnp.array_split(dataset, data_size // 32) # 270, 32, 16384

    rng, state_rng = jax.random.split(rng)
    ddim_state = create_train_state(state_rng)
    
    rng, ae_rng = jax.random.split(rng)
    ae_state = get_autoencoder(ae_rng)
    
    
    for epoch in range(FLAGS.DDIM_epochs):
        pbar = tqdm(range(len(batched_dataset)), desc=f'Epoch {epoch}')
        for i in pbar:
            
            batch = batched_dataset[i]
            rng, train_step_rng = jax.random.split(rng)
            
            ddim_state, loss = train_step(ddim_state, batch, train_step_rng)
            pbar.set_postfix({"Loss": f"{loss:.5f}"})
        ddim_state = ddim_state.replace(epoch=epoch)
        rng, eval_rng = jax.random.split(rng)
        evaluate(batch, ddim_state, ae_state, eval_rng)
        
    return ddim_state
  

def evaluate(batch, ddim_state, ae_state, rng):
    print(f"Batch shape: {batch.shape}")
    ddim_variables={"params": ddim_state.params, "batch_stats": ddim_state.batch_stats}
    
    rng, gen_rng = jax.random.split(rng)
    generated_batch = ddim_state.apply_fn(ddim_variables, gen_rng, method=DiffusionModel.generate)
    
    print(f"generated_batch shape: {generated_batch.shape}")
    data_size, data_length, _ = generated_batch.shape
    batched = jnp.array_split(generated_batch[0], data_length // 256)
    
    batched = jnp.array(batched)
    
    print(f"Batched shape: {batched.shape}")
    ae_variables = {"params": ae_state.params, "batch_stats": ae_state.batch_stats}
    generated_ecg = ae_state.apply_fn(ae_variables, batched, method= AutoEncoder.decode) #64, 2048
    
    sample_batch = batch[0].reshape((1, -1, 1))
    sample_batch = sample_batch.reshape((64, 256, 1))
    batch_decoded = ae_state.apply_fn(ae_variables, sample_batch, method= AutoEncoder.decode) #64, 2048
    #generated_2s_ecgs.append(generated_ecg)
    print(f"generated_ecg shape: {generated_ecg.shape}")
    # real_batch_diffused = ddim_state.apply_fn(ddim_variables, batch, 30, 0.7, method=DiffusionModel.reverse_diffusion)
    # plt.plot(real_batch_diffused[0].flatten())
    # plt.savefig(f"{FLAGS.DDIM_img_dir}/epoch_{ddim_state.epoch}_batch_diffused.png")
    # plt.close()
    
    #Plot the latent space batch
    plt.plot(batch[0])
    plt.savefig(f"{FLAGS.DDIM_img_dir}/epoch_{ddim_state.epoch}_batch.png")
    plt.close()
    
    plt.plot(batch[0][0:512])
    plt.savefig(f"{FLAGS.DDIM_img_dir}/epoch_{ddim_state.epoch}_batch_zoom.png")
    plt.close()
    
    #Decode the latent space batch
    plt.plot(batch_decoded.flatten())
    plt.savefig(f"{FLAGS.DDIM_img_dir}/epoch_{ddim_state.epoch}_batch_decoded.png")
    plt.close()
    
    plt.plot(batch_decoded.flatten()[0:8192])
    plt.savefig(f"{FLAGS.DDIM_img_dir}/epoch_{ddim_state.epoch}_batch_decoded_zoom.png")
    plt.close()

    #Plot the DDIM output
    
    plt.plot(generated_batch[0].flatten())
    plt.savefig(f"{FLAGS.DDIM_img_dir}/epoch_{ddim_state.epoch}_ddim.png")
    plt.close()
    
    #Plot the ECGs
    plt.plot(generated_ecg.flatten())
    plt.savefig(f"{FLAGS.DDIM_img_dir}/epoch_{ddim_state.epoch}.png")
    plt.close()
    
    #zoom into start middle and end
    plt.plot(generated_ecg.flatten()[0:8192])
    plt.savefig(f"{FLAGS.DDIM_img_dir}/epoch_{ddim_state.epoch}_zoom_start.png")
    plt.close()
    
    plt.plot(generated_ecg.flatten()[50000:58000])
    plt.savefig(f"{FLAGS.DDIM_img_dir}/epoch_{ddim_state.epoch}_zoom_middle.png")
    plt.close()
    
    plt.plot(generated_ecg.flatten()[120000:128000])
    plt.savefig(f"{FLAGS.DDIM_img_dir}/epoch_{ddim_state.epoch}_zoom_end.png")
    plt.close()
    
    # #random control noise
    # noise_batch = jax.random.normal(rng, (64, 64, 64))
    # noise_batch = 0.2 * noise_batch + 0.5
    # ecg = ae_state.apply_fn(ae_variables, noise_batch, method=AutoEncoder.decode)
    
    # plt.plot(ecg[0])
    # plt.savefig(f"{FLAGS.DDIM_img_dir}/epoch_{ddim_state.epoch}_noise.png")
    # plt.close()

    
    
    

if __name__ == "__main__":
    app.run(main)
    
    
