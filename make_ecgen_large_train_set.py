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

data_folder = "/home/dominik.kranz/data/ecg/raw"


#iterate all subfolders
folders = os.listdir(data_folder)

for folder in tqdm(folders):
    #get files within folder
    #check if directory
    if not os.path.isdir(f"{data_folder}/{folder}"):
        continue
    
    files = os.listdir(f"{data_folder}/{folder}")

    parquet_file = [file for file in files if file.endswith(".parquet")][0]
    data = pd.read_parquet(f"{data_folder}/{folder}/{parquet_file}")
    # print(data.head(1))
    # quit()




    spl_prd = data.iloc[0]["c_sample_period"]
    sample_frequency = int(1000 / spl_prd)
    if sample_frequency != 500:
        print("Error: sample frequency is not 500Hz")
        continue
    #check if spl_prd contains only 2
    ecg = np.concatenate(data["c_value"])


    #split into 320s segments
    segment_length = 320 * sample_frequency

    segments = []
    for i in range(0, len(ecg), segment_length):
        segment = ecg[i:i+segment_length]
        segments.append(segment)

    segments = segments[:-1]

    for i, segment in enumerate(segments):
        cleaned = nk.ecg_clean(segment, sampling_rate=sample_frequency)
        resampled = nk.signal_resample(cleaned, sampling_rate=500, desired_sampling_rate=512)
        rpeaks = nk.ecg_findpeaks(cleaned, sampling_rate=sample_frequency)["ECG_R_Peaks"]
        noise, snr = estimate_ecg_noise(cleaned, rpeaks, sample_frequency)
        
        if snr < 12:
            continue
        
        #normalise min max
        resampled = (resampled - np.min(resampled)) / (np.max(resampled) - np.min(resampled) + 1e-6)
        
        
        batch = jnp.array(resampled)
        batch = jnp.reshape(batch, (-1, 5*512))

        
        latent_space_batch, embedding_indices_batch = autoencoder_state.apply_fn(
            {
                "params": autoencoder_state.params,
            },
        batch,
        method = AutoEncoder.encode
        )
        latent_space = jnp.reshape(latent_space_batch, (5120, 16))
        embedding_indices = jnp.reshape(embedding_indices_batch, (5120,))
        # print(latent_space_batch.shape)
        # quit()
        
        data_out.append(
            {
            "pseudonym": data.iloc[0]["c_patient_id"],
            "segment_id": i,
            # "ecg_processed": resampled,
            # noise: noise, #commented out, because it consumes too much space
            "tokens": np.array(latent_space).tolist(),
            "embedding_indices": np.array(embedding_indices).tolist(),
            "snr": snr,
            "sample_rate": 512,
            "rpeaks": rpeaks
            }
        )
        
        if len(data_out) > 10000:
            #flush
            overhang = data_out[10000:]
            rows = data_out[:10000]
            chunk_counter += 1
            df = pd.DataFrame(rows)
            df.to_parquet(f"/home/dominik.kranz/data/ecg/ecgen_large_dataset/{chunk_counter}.parquet")
            data_out = overhang
        

#flush out remaining data
df = pd.DataFrame(data_out)
df.to_parquet(f"/home/dominik.kranz/data/ecg/ecgen_large_dataset/{chunk_counter + 1}.parquet")

print("+++ Done +++")