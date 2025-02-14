import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from util.estimate_noise_alternative import estimate_ecg_noise
import neurokit2 as nk
from model.autoencoder import AutoEncoder
from train_autoencoder import TrainState
from config.config import Config
import jax
import jax.numpy as jnp
import flax
import os
import optax


SERIES_LENGTH = 2048


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

config = Config("config/ae_ecg_300ep_512hz.yml").settings
rng = jax.random.PRNGKey(0)
autoencoder_state = get_autoencoder(rng)


data_out = []
chunk_counter = 0

data_folder = "/home/dominik.kranz/data/ecg/inhouse_afib_dataset/sinus_90s"


#iterate all subfolders
files = os.listdir(data_folder)

for file in tqdm(files):
    #get files within folder
    #check if directory
    if not file.endswith(".parquet"):
        continue
    data = pd.read_parquet(f"{data_folder}/{file}")
    # print(data.head(1))
    # quit()

    ecgs = data["ecg_processed"]
    for i, ecg in enumerate(ecgs):
        #normalise min max
        ecg = (ecg - np.min(ecg)) / (np.max(ecg) - np.min(ecg) + 1e-6)
        batch = jnp.array(ecg)
        batch = jnp.reshape(batch, (-1, 5*512))
        
        latent_space_batch, embedding_indices_batch = autoencoder_state.apply_fn(
            {
                "params": autoencoder_state.params,
            },
        batch,
        method = AutoEncoder.encode
        )
        latent_space = jnp.reshape(latent_space_batch, (1440, 16))
        embedding_indices = jnp.reshape(embedding_indices_batch, (1440,))
        data_out.append(
            {
            "pseudonym": data.iloc[i]["pseudonym"],
            "tokens": np.array(latent_space).tolist(),
            "embedding_indices": np.array(embedding_indices).tolist(),
            "sample_rate": 512,
            "snr": data.iloc[i]["snr"],
            }
        )

        
        if len(data_out) > 10000:
            #flush
            overhang = data_out[10000:]
            rows = data_out[:10000]
            chunk_counter += 1
            df = pd.DataFrame(rows)
            df.to_parquet(f"/home/dominik.kranz/data/ecg/ecgen_medium_dataset/{chunk_counter}.parquet")
            data_out = overhang
        

#flush out remaining data
df = pd.DataFrame(data_out)
df.to_parquet(f"/home/dominik.kranz/data/ecg/ecgen_medium_dataset/{chunk_counter + 1}.parquet")

print("+++ Done +++")