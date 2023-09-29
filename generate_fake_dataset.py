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


def main(argv, ddim_state=None):
    rng = jax.random.PRNGKey(0)
    
    ###load model states###
    if ddim_state == None:
        ddim_state = get_ddim(rng, "outputs/ddim/20230927-173339_mu_zero_sigma_13/checkpoints")
    
    ae_state = get_autoencoder(rng, "outputs/autoencoder/20230808-180427-VQ-VAE/checkpoints")
    
    ddim_variables = {"params": ddim_state.params, "batch_stats": ddim_state.batch_stats}
    ae_variables = {"params": ae_state.params, "batch_stats": ae_state.batch_stats}
    
    #same shape as the real ECG dataset
    ecgs = []
    for i in range(54):
        pbar = tqdm(range(64), desc=f'Generating batch number {i}')
        rng, gen_rng = jax.random.split(rng)
        ddim_batch =  ddim_state.apply_fn(ddim_variables, gen_rng, 64, method=DiffusionModel.generate)
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
    #print(f"FINAL ECG SHAPE: {ecgs.shape}")
    onp.save("fake_dataset.npy", ecgs)
            
    
if __name__ == "__main__":
    app.run(main)