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


from absl import flags, app

# SERIES_LENGTH = 30_720
# BATCH_SIZE = 64

FLAGS = flags.FLAGS

def get_ddim(rng, checkpoint_path) -> DDIMTrainState:
    model = DiffusionModel(FLAGS.DDIM_feature_sizes)
    dummy_ecg = jnp.ones((1, FLAGS.DDIM_batch_dims[0], FLAGS.DDIM_batch_dims[1]), dtype=jnp.float32)
    variables = model.init(rng, dummy_ecg, rng, train=False)
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


def analyse_ecg(ecg):
    r_peaks = detect_ecg_qrs(ecg, samplerate=1024)
    r_peaks_time = r_peaks / 1024 * 1000
    r_peak_diff = onp.abs(onp.diff(r_peaks_time, 1))

    filtered = adaptive_hrv_filter(r_peak_diff)["filtered_hrv"]

    td_params = calculate_time_domain_parameters(hrv_series=filtered)
    req, pow = welch(filtered)
    
    
    # y_peaks = []
    # for pos in r_peaks:
    #      y_peaks.append(ecg[pos])
    
    plt.figure()
    plt.plot(req, pow)
    plt.savefig("test4.png")

    
    return td_params

def main(argv):
    rng = jax.random.PRNGKey(0)

    ###Model###
    ddim_state = get_ddim(rng, "outputs/ddim/20230811-153112/checkpoints/")
    ae_state = get_autoencoder(rng, "outputs/autoencoder/20230808-180427-VQ-VAE/checkpoints")
    #generate a batch
    ddim_variables = {"params": ddim_state.params, "batch_stats": ddim_state.batch_stats}
    ae_variables = {"params": ae_state.params, "batch_stats": ae_state.batch_stats}
    gen_rng, rng = jax.random.split(rng)
    generated_batch = ddim_state.apply_fn(ddim_variables, gen_rng, 64, method=DiffusionModel.generate)
    
    generated_batch_rmssd = []
    generated_batch_hr = []
    for i in range(64):
        data_size, data_length, _ = generated_batch.shape
        batched = jnp.array_split(generated_batch[i], data_length // 32)
        batched=jnp.array(batched).squeeze()
        
        embedded = ae_state.apply_fn(ae_variables, batched, method=AutoEncoder.embed)
        generated_ecg_chunks = ae_state.apply_fn(ae_variables, embedded, method=AutoEncoder.decode)
        generated_ecg = onp.array(generated_ecg_chunks.flatten())
        
        td_params = analyse_ecg(generated_ecg)
        generated_batch_rmssd.append(td_params.rmssd)
        #generated_ecgs.append(generated_ecg)
        #print(td_params.hr)
        generated_batch_hr.append(td_params.hr)
    print(f"Generated batch mean rmssd: {onp.average(generated_batch_rmssd)}")
    print(f"Generated batch std rmssd: {onp.std(generated_batch_rmssd)}")
    print(f"Generated batch mean hr: {onp.average(generated_batch_hr)}")
    print(f"Generated batch std hr: {onp.std(generated_batch_hr)}")
    
    
    ###Dataset###

    data, labels = load_ecg_dataset(rng, 1024 * 5 * 64, batch_size=64, normalise=False)
    print("DATASET SHAPE:")
    print(onp.array(data).shape)

    # ecg = data[0][0]

    batch_rmssd = []
    batch_hr = []
    for i in range(64):
        ecg = data[15][i]
        r_peaks = detect_ecg_qrs(ecg, samplerate=1024)
        # y_peaks = []
        # for pos in r_peaks:
        #     y_peaks.append(ecg[pos])
        

        r_peaks_time = r_peaks / 1024 * 1000
        r_peak_diff = onp.abs(onp.diff(r_peaks_time, 1))

        filtered = adaptive_hrv_filter(r_peak_diff)["filtered_hrv"]

        td_params = calculate_time_domain_parameters(hrv_series=filtered)

        batch_rmssd.append(td_params.rmssd)
        batch_hr.append(td_params.hr)

        # freq, pow = welch(filtered)
    print(f"dataset batch mean rmssd: {onp.average(batch_rmssd)}")
    print(f"dataset batch std rmssd: {onp.std(batch_rmssd)}")
    print(f"dataset batch mean hr: {onp.average(batch_hr)}")
    print(f"dataset batch std hr: {onp.std(batch_hr)}")
    
    
if __name__ == "__main__":
    app.run(main)
    