import jax
import jax.numpy as jnp

from absl import flags

flags.DEFINE_float("AE_beta", 0.25, "A weight applied to the regulirasitation loss")

EPSILON = 0.0005

@jax.vmap
def L1(predictions, targets):
    #predictions = predictions.flatten()
    #targets = targets.flatten()
    return jnp.abs(jnp.subtract(predictions, targets))

@jax.vmap
def L2(predictions, targets):
    return jnp.square(jnp.subtract(predictions, targets))


def AE_loss(predictions, targets, epoch):
    l1 = L1(predictions, targets)
    l2 = L2(predictions, targets)
    sigmoided_l1 = (1 -  1 / (1 + jnp.exp(-1*(epoch - 5)))) * l1
    sigmoided_l2 =  1 / (1 + jnp.exp(-1*(epoch - 5))) * l2
    # sigmoided_l4 = 1 / (1 + jnp.exp(-1*(epoch - 15))) * l4
    return sigmoided_l1 + sigmoided_l2
    


@jax.vmap
def KLD(P, Q):
    return -0.5 * jnp.sum(1 + Q - jnp.square(P) - jnp.exp(Q))

@jax.vmap
def L4(predictions, targets):
    return jnp.square(jnp.square(predictions - targets))

@jax.vmap
def EXP(predictions, targets):
    return jnp.exp(jnp.abs(predictions - targets))


def vae_loss(recon_loss, regu_loss, epoch):
    """This loss lets the reconstruction loss "warm up" a bit.

    Args:
        recon_loss (_type_): _description_
        regu_loss (_type_): _description_
    """
    sigmoided = 1 / (1 + jnp.exp(-1*(epoch - 25))) * regu_loss

    #return recon_loss + jnp.heaviside(epoch - 10, 1.0) * sigmoided
    return recon_loss + flags.FLAGS.AE_beta * sigmoided
