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
        commitment_loss_beta=config["AE_commitment_loss_beta"],
        convolution_filters=config["AE_convolution_filters"],
        kernel_sizes=config["AE_convolution_kernels"],
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
    path = config["output_root_dir"] + config["checkpoint_dir"] + "25/"
    return flax.training.checkpoints.restore_checkpoint(ckpt_dir=path, target=init_state, step=25)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/ae_ecg_300ep_512hz.yml")
    args = parser.parse_args()
    config = Config(f"{args.config}").settings

    rng = jax.random.PRNGKey(0)
    autoencoder_state = get_autoencoder(rng)


    

    # Load dataset
    latent_spaces = []
    #pbar = tqdm(data_loader(config["AE_dataset_chunks"], config["AE_dataset_root"]), desc="Loading dataset")
    segments, labels = dataset_loader.load_afib_dataset_30s()
    #create iterator for segments and labels
    pbar = tqdm(zip(segments, labels), desc="Loading dataset")
    for i, (batch, label) in enumerate(pbar):
        if batch.shape != (32, 15360):
            print("OH NO")
            continue
        #cut into 5s segments
        # print(batch.shape)
        # print(label.shape)
        batch = jnp.reshape(batch, (-1, 5*512)) # (32, L) => (192, L2)
        # print(batch.shape)
        #labels are shape (32,), reshape to (192,)
        #label = jnp.repeat(label, 6)
        
        # print(batch.shape)
        # print()
        
        # quit()
        latent_space_batch, embedding_indices_batch = autoencoder_state.apply_fn(
            {
                "params": autoencoder_state.params,
            },
        batch,
        method = AutoEncoder.encode
        )
        # print(latent_space_batch.shape) 
        # now reshape (192, 80, 16) => (32, 480, 16)
        latent_space_batch = jnp.reshape(latent_space_batch, (32, 480, 16))
        
        
        # quit()
        # print(batch.shape)
        # print(latent_space_batch.shape)
        # quit()
        latent_spaces.append(latent_space_batch)

        #if latent space too large, save it to disk
        if len(latent_spaces) > 100:
            # onp.savez_compressed("/home/dominik.kranz/data/ecg/inhouse_afib_dataset/tokenized/" f"latent_space_{pbar.n}.npz", *latent_spaces)
            latent_spaces = []
    print("Done")
    #save remaining latent spaces
    # onp.savez_compressed("/home/dominik.kranz/data/ecg/inhouse_afib_dataset/tokenized/" f"latent_space_{pbar.n}.npz", *latent_spaces)
    #save labels
    # onp.save("/home/dominik.kranz/data/ecg/inhouse_afib_dataset/tokenized/labels.npy", labels)



