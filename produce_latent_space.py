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
    ecgs = dataset
    latent_spaces = []
    embedding_indices = []
    for i in tqdm(range(len(ecgs)), desc="Creating latent space from dataset"):
        batch = ecgs[i]
        latent_space_batch, embedding_indices_batch = autoencoder_state.apply_fn(
            {
                "params": autoencoder_state.params,
                "batch_stats": autoencoder_state.batch_stats
            },
        batch,
        method = AutoEncoder.encode
        )
        latent_spaces.append(latent_space_batch)
        embedding_indices.append(embedding_indices_batch)
        
        if (i == 0):
            #Test a roundtrip to make sure we save it correctly
            print(embedding_indices_batch.shape)
            print(embedding_indices_batch[0])
            
            embeds_from_indices = autoencoder_state.apply_fn({
                "params": autoencoder_state.params,
                "batch_stats": autoencoder_state.batch_stats
            }, embedding_indices_batch, method=AutoEncoder.embed_indices)
            
            decoded_from_embeds = autoencoder_state.apply_fn({
                "params": autoencoder_state.params,
                "batch_stats": autoencoder_state.batch_stats
            }, latent_space_batch, method=AutoEncoder.decode)
                        
            decoded_from_indices = autoencoder_state.apply_fn({
                "params": autoencoder_state.params,
                "batch_stats": autoencoder_state.batch_stats
            }, embeds_from_indices, method=AutoEncoder.decode)
            print(latent_space_batch.shape)
            
            plt.plot(latent_space_batch[1].flatten())
            plt.savefig("test_roundtrip_latents")
            plt.figure()
            plt.plot(batch[1])
            plt.savefig("test_roundtrip_orig")
            plt.figure()
            plt.plot(decoded_from_embeds[1])
            plt.savefig("test_roundtrip1.png")
            plt.figure()
            plt.plot(decoded_from_indices[1])
            plt.savefig("test_roundtrip2.png")
            plt.close()
            
    output = jnp.array(latent_spaces)
    output_indices = jnp.array(embedding_indices)
    
    return output, output_indices, ecgs

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-mp', '--model-path', type=str, default="outputs/autoencoder/20230808-180427-VQ-VAE/checkpoints")
    parser.add_argument("-s", "--seed", type=int, default=0)
    
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-4)
    parser.add_argument('-ema', '--ema-momentum', type=float, default=0.990)
    
    parser.add_argument('-wd', '--weight-decay', type=float, default=1e-4)
    
    args = parser.parse_args()
    rng = jax.random.PRNGKey(args.seed)
    autoencoder_state = get_autoencoder(args, rng)    
    data_rng, rng = jax.random.split(rng)
    #dataset = dataset_loader.load_ecg_dataset(data_rng, SERIES_LENGTH, BATCH_SIZE, normalise=True, dataset_path="data/ecgs_within_65_and_75_hr.npy")
    dataset = onp.load("data/ecgs_within_65_and_75_hr.npy")
    ecgs_2s = dataset.reshape((-1, 2048))
    data_size, data_length = ecgs_2s.shape
    batched = onp.array_split(ecgs_2s, data_size // 64)
    
    
    latent_space, embedding_indices, ecgs = produce_ls(batched, autoencoder_state)
    print(f"Latent space shape: {latent_space.shape}")
    print(f"Embedding indices shape: {embedding_indices.shape}")
    jnp.save("ecgs", ecgs)
    jnp.save("latent_space", latent_space)
    jnp.save("embedding_indices", embedding_indices)