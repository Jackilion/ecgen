import argparse
import flax
import jax
import optax
import jax.numpy as jnp
import dataset_loader
from tqdm import tqdm
from model.autoencoder import AutoEncoder
from train_autoencoder import TrainState

SERIES_LENGTH = 2048
BATCH_SIZE = 64

def get_autoencoder(config, rng):
    model = AutoEncoder(block_depths=1)
    rng_params, rng = jax.random.split(rng)
    dummy_ecg = jnp.ones((1, SERIES_LENGTH), dtype=jnp.float32)
    variables = model.init(rng_params, dummy_ecg, train=True)

    tx = optax.adamw(learning_rate=config.learning_rate,
                     weight_decay=config.weight_decay)

    #vars = flax.training.checkpoints.restore_checkpoint(ckpt_dir=config.model_path, target=None, step=99)
    
    init_state = TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=tx,
        batch_stats=variables["batch_stats"],
        ema_params=None,
        ema_momentum=config.ema_momentum
    )
    
    return flax.training.checkpoints.restore_checkpoint(ckpt_dir=config.model_path, target=init_state, step=29)
    
def produce_ls(dataset, autoencoder_state):
    """Calls the autoencoders encoder function with the given dataset and saves the dataset, labels and latent space

    Args:
        dataset (_type_): _description_
        autoencoder_state (_type_): _description_
    """
    ecgs, labels = dataset
    latent_spaces = []
    for i in tqdm(range(len(ecgs)), desc="Creating latent space from dataset"):
        batch = ecgs[i]
        latent_space_batch = autoencoder_state.apply_fn(
            {
                "params": autoencoder_state.params,
                "batch_stats": autoencoder_state.batch_stats
            },
        batch,
        method = AutoEncoder.encode
        )
        latent_spaces.append(latent_space_batch)
    output = jnp.array(latent_spaces)
    
    return output, ecgs, labels

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-mp', '--model-path', type=str, default="None")
    parser.add_argument("-s", "--seed", type=int, default=0)
    
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-4)
    parser.add_argument('-ema', '--ema-momentum', type=float, default=0.990)
    
    parser.add_argument('-wd', '--weight-decay', type=float, default=1e-4)
    
    args = parser.parse_args()
    rng = jax.random.PRNGKey(args.seed)
    autoencoder_state = get_autoencoder(args, rng)    
    data_rng, rng = jax.random.split(rng)
    dataset = dataset_loader.load_ecg_dataset(data_rng, SERIES_LENGTH, BATCH_SIZE)
    
    latent_space, ecgs, labels = produce_ls(dataset, autoencoder_state)
    
    jnp.save("ecgs", ecgs)
    jnp.save("latent_space", latent_space)
    jnp.save("labels", labels)