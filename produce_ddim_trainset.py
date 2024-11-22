import argparse
import flax
import jax
import optax
import jax.numpy as jnp
import dataset_loader
import numpy as onp
from tqdm import tqdm
import matplotlib.pyplot as plt
from model.autoencoder import AutoEncoder
from train_autoencoder import TrainState
from config.config import Config
from util.data_loader import data_loader

SERIES_LENGTH = 2048
BATCH_SIZE = 64


def get_autoencoder(rng):
    config = Config().settings
    model = AutoEncoder(
        block_depths=config["AE_block_depths"],
        embed_size_K=config["AE_embed_size_K"],
        embed_dim_D=config["AE_embed_dim_D"],
        commitment_loss_beta=config["AE_commitment_loss_beta"]
        )
    rng_params, rng = jax.random.split(rng)
    dummy_ecg = jnp.ones((64, SERIES_LENGTH), dtype=jnp.float32)
    variables = model.init(rng_params, dummy_ecg, train=True)

    tx = optax.adamw(learning_rate=config["AE_learning_rate"],
                        weight_decay=config["AE_weight_decay"])
    
    init_state = TrainState.create(
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




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/ae_ecg_300ep_512hz.yml")
    args = parser.parse_args()
    config = Config(f"{args.config}").settings

    rng = jax.random.PRNGKey(0)
    autoencoder_state = get_autoencoder(rng)


    

    # Load dataset
    latent_spaces = []
    pbar = tqdm(data_loader(config["AE_dataset_chunks"], config["AE_dataset_root"]), desc="Loading dataset")
    for batch in pbar:
        latent_space_batch, embedding_indices_batch = autoencoder_state.apply_fn(
            {
                "params": autoencoder_state.params,
            },
        batch,
        method = AutoEncoder.encode
        )
        # print(batch.shape)
        # print(latent_space_batch.shape)
        # quit()
        latent_spaces.append(latent_space_batch)

        #if latent space too large, save it to disk
        if len(latent_spaces) > 100:
            onp.savez_compressed("/home/dominik.kranz/data/ecg/tokenised/" f"latent_space_{pbar.n}.npz", *latent_spaces)
            latent_spaces = []



