from functools import partial
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
from util.learning_rate_scheduler import create_annealing_learning_rate_fn
import util.losses as losses
import dataset_loader
from pathlib import Path
from absl import app, flags

FLAGS = flags.FLAGS


#! Training hyperparameter
flags.DEFINE_float("DDIM_learning_rate", 1e-4, "The autoencoders learning rate")
flags.DEFINE_float("DDIM_weight_decay", 0.01, "The autoencoders weight decay")
flags.DEFINE_float("DDIM_ema_momentum", 0.990, "The autoencoders EMA momentum")
flags.DEFINE_integer("DDIM_epochs", 100, "The autoencoders training epochs")

#! Others
flags.DEFINE_integer("DDIM_run_seed", 0, "The seed used to generate JAX prng")
flags.DEFINE_list("DDIM_batch_dims", [5120, 8], "The dimensions of a latent space tensor")
flags.DEFINE_string("AE_model_path", None, "The Autoencoder checkpoint path to use for evaluation")
flags.DEFINE_string("DDIM_dataset_path", "latent_spaces/VQ-VAE_latent_space.npy", "The autoencoded latent space used for training")

#! Model parameter
flags.DEFINE_list("DDIM_feature_sizes", [128, 128, 128, 128, 128], "The sizes for the unet as a list of features sizes of length n. The unet will have n-1 layers, and the final feature is a resnet block at the bottom")
flags.DEFINE_integer("DDIM_block_depth", 2, "The number of times each resnet block is repeated")

#! Logging flags
now = datetime.now().strftime("%Y%m%d-%H%M%S")
flags.DEFINE_string("DDIM_output_dir", f"./outputs/ddim/{now}", "The output root directory where all the models output will be saved")
flags.DEFINE_string("DDIM_img_dir", f"./outputs/ddim/{now}/images", "The directory where evaluation images will be stored")
flags.DEFINE_string("DDIM_log_dir", f"./outputs/ddim/{now}/logs", "The directory where logs will be stored")
flags.DEFINE_string("DDIM_ckpt_dir", f"./outputs/ddim/{now}/checkpoints", "The directory where model checkpoints will be stored")

def evaluate_finished_model(ddim_state):
    from generate_fake_dataset import main as fake_datasets_main
    fake_datasets_main(None, ddim_state=ddim_state)

def main(argv):
    Path(FLAGS.DDIM_output_dir).mkdir(parents=True, exist_ok=True)
    Path(f"{FLAGS.DDIM_img_dir}").mkdir(parents=True, exist_ok=True)
    Path(f"{FLAGS.DDIM_log_dir}").mkdir(parents=True, exist_ok=True)
    Path(f"{FLAGS.DDIM_ckpt_dir}").mkdir(parents=True, exist_ok=True)
    
    ddim_state = train()
    #Evaluate after training
    evaluate_finished_model(ddim_state)
    
    
class TrainState(train_state.TrainState):
    batch_stats: Any
    ema_params: Any = None
    ema_momentum: float = None
    epoch: int = None


def create_train_state(rng, learning_rate_fn):
    """Creates initial TrainState to hold params"""

    model = DiffusionModel(
        feature_sizes=FLAGS.DDIM_feature_sizes,
        block_depths=FLAGS.DDIM_block_depth,
        #attention_depths=config.attention_depths
    )
    rng_init, rng_params = jax.random.split(rng)

    dummy_batch = jnp.ones((1, FLAGS.DDIM_batch_dims[0], FLAGS.DDIM_batch_dims[1]), dtype=jnp.float32)

    variables = model.init(rng_init, dummy_batch, rng_params, train=True)

    tx = optax.adamw(learning_rate=learning_rate_fn,
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


@partial(jax.jit, static_argnums=3)
def train_step(state, batch, rng, learning_rate_fn):
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
        loss = losses.L2(pred_noises.reshape((B, -1)), noises.reshape(B, -1)).mean()
        return loss, mutated_vars

    grad_fn = jax.value_and_grad(compute_loss, has_aux=True)
    (loss, auxillary_data), grads = grad_fn(state.params)
    
    mutated_vars = auxillary_data
    
    new_state = state.apply_gradients(
        grads=grads, batch_stats=mutated_vars['batch_stats']
    )
    lr = learning_rate_fn(state.step)
    return new_state, loss, lr


def train() -> TrainState:
    tf.config.experimental.set_visible_devices([], "GPU")
    rng = jax.random.PRNGKey(FLAGS.DDIM_run_seed)
    
    dataset = jnp.load(FLAGS.DDIM_dataset_path)
    print(dataset.shape) #e.g. 8640, 64, 32 = (N, B, embedding_indices)
    #dataset = dataset / 64.0
    print(dataset[0][0].shape)
    #print(dataset[0][0])
    dataset = jnp.reshape(dataset, (-1, 160*32, 8)) #(3456, 5120, 8)
    print(dataset.shape)
    #quit()
    data_size, data_length, _, = dataset.shape
    batched_dataset = jnp.array_split(dataset, data_size // 16) # 216, 16, 5120
    batched_dataset_test = jnp.array(batched_dataset)
    print(batched_dataset_test.shape)
    #quit()
    rng, state_rng = jax.random.split(rng)
    learning_rate_fn = create_annealing_learning_rate_fn(FLAGS.DDIM_epochs, batched_dataset_test.shape[0])
    ddim_state = create_train_state(state_rng, learning_rate_fn)
    
    rng, ae_rng = jax.random.split(rng)
    ae_state = get_autoencoder(ae_rng)
    
    
    for epoch in range(FLAGS.DDIM_epochs):
        pbar = tqdm(range(len(batched_dataset)), desc=f'Epoch {epoch}')
        for i in pbar:
            
            batch = batched_dataset[i]
            rng, train_step_rng = jax.random.split(rng)
            
            ddim_state, loss, lr = train_step(ddim_state, batch, train_step_rng, learning_rate_fn)
            pbar.set_postfix({"Loss": f"{loss:.5f}", "Lr": f"{lr:.5f}"})
        ddim_state = ddim_state.replace(epoch=epoch)
        rng, eval_rng = jax.random.split(rng)
        evaluate(batch, ddim_state, ae_state, eval_rng)
        checkpoints.save_checkpoint(ckpt_dir=FLAGS.DDIM_ckpt_dir, target=ddim_state, step=epoch)
        
    return ddim_state
  

def evaluate(batch, ddim_state, ae_state, rng):
    print(f"Batch shape: {batch.shape}")
    ddim_variables={"params": ddim_state.params, "batch_stats": ddim_state.batch_stats}
    
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
    ae_variables = {"params": ae_state.params, "batch_stats": ae_state.batch_stats}
    #print(generated_ints.shape)
    #print(generated_ints[0])
    #embed_vectors = ae_state.apply_fn(ae_variables, batched, method=AutoEncoder.embed_indices)
    embedded = ae_state.apply_fn(ae_variables, batched, method=AutoEncoder.embed)
    generated_ecg = ae_state.apply_fn(ae_variables, embedded, method= AutoEncoder.decode) #64, 2048
    
    sample_batch = batch[0]
    sample_batch = sample_batch.reshape((160, 32, 8))
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
    plt.savefig(f"{FLAGS.DDIM_img_dir}/epoch_{ddim_state.epoch}_batch.png")
    plt.close()
    
    x = onp.arange(0, len(batch[0].flatten()[0:32*8*2]))
    plt.scatter(x, batch[0].flatten()[0:32*8*2], marker=".")
    plt.title("Flattened latent space of a 4s ECG")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(f"{FLAGS.DDIM_img_dir}/epoch_{ddim_state.epoch}_batch_zoom.png")
    plt.close()
    
    
    x = onp.arange(0, len(embedded[0].flatten()))
    plt.scatter(x, embedded[0].flatten(), marker=".")
    plt.title("Flattened embedded latent space of a 2 ECG")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(f"{FLAGS.DDIM_img_dir}/epoch_{ddim_state.epoch}_ddim_embedded.png")
    plt.close()
    
    x = onp.arange(0, len(embedded[0].flatten()[0:256]))
    plt.scatter(x, embedded[0].flatten()[0:256], marker=".")
    plt.title("Flattened embedded latent space of a 4 ECG")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(f"{FLAGS.DDIM_img_dir}/epoch_{ddim_state.epoch}_ddim_embedded_zoom.png")
    plt.close()
    
    

    
    #Decode the latent space batch
    plt.plot(batch_decoded.flatten())
    plt.xlabel("Samples")
    plt.ylabel("AU")
    plt.title("Decoded latent space")
    plt.tight_layout()
    plt.savefig(f"{FLAGS.DDIM_img_dir}/epoch_{ddim_state.epoch}_batch_decoded.png")
    plt.close()
    
    plt.plot(batch_decoded.flatten()[0:8192])
    plt.title("First 8 seconds of decoded latent space")
    plt.xlabel("Samples")
    plt.ylabel("AU")
    plt.tight_layout()
    plt.savefig(f"{FLAGS.DDIM_img_dir}/epoch_{ddim_state.epoch}_batch_decoded_zoom.png")
    plt.close()

    #Plot the DDIM output
    
    plt.plot(generated_batch[0].flatten())
    plt.title("Flattened DDM output")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(f"{FLAGS.DDIM_img_dir}/epoch_{ddim_state.epoch}_ddim.png")
    plt.close()
    
    #Plot the ECGs
    plt.plot(generated_ecg.flatten())
    plt.title("Decoded DDM output")
    plt.xlabel("Samples")
    plt.ylabel("AU")
    plt.tight_layout()
    plt.savefig(f"{FLAGS.DDIM_img_dir}/epoch_{ddim_state.epoch}.png")
    plt.close()
    
    #zoom into start middle and end
    plt.plot(generated_ecg.flatten()[0:8192])
    plt.xlabel("Samples")
    plt.ylabel("AU")
    plt.title("First 8 seconds of decoded DDM output")
    plt.tight_layout()
    plt.savefig(f"{FLAGS.DDIM_img_dir}/epoch_{ddim_state.epoch}_zoom_start.png")
    plt.close()
    
    plt.plot(generated_ecg.flatten()[50000:58000])
    plt.xlabel("Samples")
    plt.ylabel("AU")
    plt.title("8 seconds of decoded DDM output")
    plt.tight_layout()
    plt.savefig(f"{FLAGS.DDIM_img_dir}/epoch_{ddim_state.epoch}_zoom_middle.png")
    plt.close()
    
    plt.plot(generated_ecg.flatten()[120000:128000])
    plt.xlabel("Samples")
    plt.ylabel("AU")
    plt.title("Last 8 seconds of decoded DDM output")
    plt.tight_layout()
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
    
    
