import flax
import jax
from model.autoencoder import AutoEncoder
import jax.numpy as jnp
import optax
from train_autoencoder import TrainState as AETrainState

from absl import flags

# SERIES_LENGTH = 30_720
# BATCH_SIZE = 64

FLAGS = flags.FLAGS

def get_autoencoder(rng) -> AETrainState:
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
    
    return flax.training.checkpoints.restore_checkpoint(ckpt_dir=FLAGS.AE_model_path, target=init_state, step=29)