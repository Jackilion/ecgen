import numpy as onp
import matplotlib.pyplot as plt
import jax
from dataset_loader import load_ecg_dataset
from util.ecg_utils import detect_ecg_qrs
from util.adaptive_filter import adaptive_hrv_filter
from util.hrv_utils import TimeDomainParameters, FrequencyDomainParameters, calculate_time_domain_parameters
import jax.numpy as jnp
from scipy.signal import welch
from train_ddim import TrainState as DDIMTrainState
import optax
import flax
from train_autoencoder import TrainState as AETrainState
from model.ddim import DiffusionModel
from model.autoencoder import AutoEncoder
from tqdm import tqdm
from PIL import Image


from absl import flags, app

# SERIES_LENGTH = 30_720
# BATCH_SIZE = 64

FLAGS = flags.FLAGS

def get_ddim(rng, checkpoint_path) -> DDIMTrainState:
    model = DiffusionModel(FLAGS.DDIM_feature_sizes)
    dummy_ecg = jnp.ones((1, FLAGS.DDIM_batch_dims[0], FLAGS.DDIM_batch_dims[1]), dtype=jnp.float32)
    variables = model.init(rng, dummy_ecg, rng, train=False)
    param_count = sum(x.size for x in jax.tree_leaves(variables))
    print(f"PARAMETER COUNT DDIM: {param_count}")
    tx = optax.adamw(learning_rate=1,
                     weight_decay=FLAGS.DDIM_weight_decay)
    init_state = DDIMTrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=tx,
        batch_stats=variables["batch_stats"],
        ema_params=variables["params"],
        ema_momentum=FLAGS.DDIM_ema_momentum
    )
    return flax.training.checkpoints.restore_checkpoint(ckpt_dir=checkpoint_path, target=init_state, step=99)


def get_autoencoder(rng, checkpoint_path) -> AETrainState:
    model = AutoEncoder(block_depths=1)
    rng_params, rng = jax.random.split(rng)
    dummy_ecg = jnp.ones((1, FLAGS.AE_ecg_length), dtype=jnp.float32)
    variables = model.init(rng_params, dummy_ecg, train=True)
    param_count = sum(x.size for x in jax.tree_leaves(variables))
    print(f"PARAMETER COUNT AE: {param_count}")
    tx = optax.adamw(learning_rate=FLAGS.AE_learning_rate,
                     weight_decay=FLAGS.AE_weight_decay)

    #vars = flax.training.checkpoints.restore_checkpoint(ckpt_dir=config.model_path, target=None, step=99)
    
    init_state = AETrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=tx,
        batch_stats=variables["batch_stats"],
        ema_params=None,
        ema_momentum=FLAGS.AE_ema_momentum
    )
    return flax.training.checkpoints.restore_checkpoint(ckpt_dir=checkpoint_path, target=init_state, step=29)


def diffusion_schedule(diffusion_times):
    schedule_type = "linear"
    start_log_snr = 2.5
    end_log_snr = -7.5
    start_snr = jnp.exp(start_log_snr)
    end_snr = jnp.exp(end_log_snr)
    
    start_noise_power = 1.0 / (1.0 + start_snr)
    end_noise_power = 1.0 / (1.0 + end_snr)
    
    if schedule_type == "linear":
        noise_powers = start_noise_power + diffusion_times * (
            end_noise_power - start_noise_power
        )

    elif schedule_type == "cosine":
        start_angle = jnp.arcsin(start_noise_power ** 0.5)
        end_angle = jnp.arcsin(end_noise_power ** 0.5)
        diffusion_angles = start_angle + \
            diffusion_times * (end_angle - start_angle)

        noise_powers = jnp.sin(diffusion_angles) ** 2

    elif schedule_type == "log-snr-linear":
        noise_powers = start_snr ** diffusion_times / (
            start_snr * end_snr**diffusion_times + start_snr ** diffusion_times
        )

    else:
        raise NotImplementedError("Unsupported sampling schedule")
    
    #signal + noise = 1
    signal_powers = 1.0 - noise_powers
    
    signal_rates = signal_powers**0.5
    noise_rates = noise_powers**0.5
    
    return noise_rates, signal_rates



def main(argv, ddim_state=None):
    rng = jax.random.PRNGKey(0)
    
    ###load model states###
    if ddim_state == None:
        ddim_state = get_ddim(rng, "outputs/ddim/20230914-121044_mu_zero_sigma_03/checkpoints")
    
    ae_state = get_autoencoder(rng, "outputs/autoencoder/20230808-180427-VQ-VAE/checkpoints")
    
    ddim_variables = {"params": ddim_state.params, "batch_stats": ddim_state.batch_stats}
    ae_variables = {"params": ae_state.params, "batch_stats": ae_state.batch_stats}
    
    #! Load original dataset
    #original_dataset = jnp.load(FLAGS.DDIM_dataset_path)
    original_dataset = jnp.load("latent_spaces/VQ-VAE_latent_space.npy")
    original_dataset = jnp.reshape(original_dataset, (-1, 160*32, 8)) #(3456, 5120, 8)
    data_size, data_length, _, =original_dataset.shape
    
    batched_dataset = jnp.array_split(original_dataset, data_size // 16) # 216, 16, 5120, 8
    batched_dataset_test = jnp.array(batched_dataset)

    
    #same shape as the real ECG dataset
    ecgs = []
    for i in range(216):
        pbar = tqdm(range(16), desc=f'Generating batch number {i}')
        rng, gen_rng = jax.random.split(rng)
        
        #! get real ECGs
        real_batch = batched_dataset[i]

        dims = real_batch.shape
        
        #! Cross fade with noise
        mu = 0.0
        sigma = 0.3
        rng, rand_rng = jax.random.split(rng)
        noise = jax.random.normal(rand_rng, dims)
        noise = sigma * noise + mu
        
        #diffusion_times = jnp.ones_like(real_batch)
        diffusion_times = jnp.ones((dims[0], 1, 1), dtype=real_batch.dtype)
        diffusion_times = diffusion_times - 0.1
        noise_rates, signal_rates = diffusion_schedule(diffusion_times)
        if i == 0:
            print(diffusion_times.shape)
            print(noise_rates.shape)
            print(signal_rates.shape)
            print(real_batch.shape)
            noisy_batch = real_batch * signal_rates + noise * noise_rates
            
            
            print(noisy_batch.shape)
            
            print(real_batch[20][0])
            print(noisy_batch[20][0])
        
            x = onp.arange(0, 5120 * 8, step=1)
            print("Saving . . .")
            plt.scatter(x, noisy_batch[0].flatten(), marker=".")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title("Noisy, flattened latent space of a 320s ECG")
            plt.tight_layout()
            plt.savefig("Batch_noisy.png")
            
            plt.figure()
            plt.scatter(x, real_batch[0].flatten(), marker=".")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title("Flattened latent space of a 320s ECG")
            plt.tight_layout()
            plt.savefig("Batch_clean.png")
        
            plt.figure()
            x = onp.arange(0, 64 * 8, step=1)
            plt.scatter(x, noisy_batch[0][0:64].flatten(), marker=".")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title("Noisy, flattened latent space of a 4s ECG")
            plt.tight_layout()
            plt.savefig("Batch_noisy_zoom.png")
            
            
            plt.figure()
            plt.scatter(x, real_batch[0][0:64].flatten(), marker=".")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title("Flattened latent space of a 4s ECG")
            plt.tight_layout()
            plt.savefig("Batch_clean_zoom.png")
        
        ddim_batch =  ddim_state.apply_fn(ddim_variables, noisy_batch, 0.1, method=DiffusionModel.generate_from_noise)
        ae_batch = []
        for j in pbar:
            data_size, data_length, _ = ddim_batch.shape

            batched = jnp.array_split(ddim_batch[j], data_length // 32)
            batched=jnp.array(batched).squeeze()
            
            embedded = ae_state.apply_fn(ae_variables, batched, method=AutoEncoder.embed)
            generated_ecg_chunks = ae_state.apply_fn(ae_variables, embedded, method=AutoEncoder.decode)
            generated_ecg = onp.array(generated_ecg_chunks.flatten())
            ae_batch.append(generated_ecg)
        ae_batch = onp.array(ae_batch)
        ecgs.append(ae_batch)
    ecgs = onp.array(ecgs)
    print(f"FINAL ECG SHAPE: {ecgs.shape}")
    onp.save("fake_dataset.npy", ecgs)
            
    
if __name__ == "__main__":
    app.run(main)