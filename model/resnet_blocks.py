import flax.linen as nn
import jax
import jax.numpy as jnp

def nonlinearity(x):
    return nn.swish(x)
    #return nn.leaky_relu(x, negative_slope=0.1)


class ResnetBlock(nn.Module):
    features: int = None
    dropout: float = 0.1
    kernel_size: int = 10
    # strides: tuple = (1,)
    # resample: Optional[str] = None

    @nn.compact
    def __call__(self, h_in, train: bool):
        residual = nn.WeightNorm(nn.Conv(features=self.features, kernel_size=[1]))(h_in)
        # h = nn.BatchNorm(use_running_average=not train,
        #                  use_bias=False, use_scale=False)(h_in)
        h = nn.WeightNorm(nn.Conv(features=self.features, kernel_size=[self.kernel_size]))(h_in)
        h = nn.Dropout(rate=self.dropout)(h, deterministic=not train)
        h = nonlinearity(h)
        h = nn.WeightNorm(nn.Conv(features=self.features, kernel_size=[self.kernel_size]))(h)
        return h + residual


class DownBlock(nn.Module):
    features: int = None
    return_skips: bool = True
    block_depth: int = None
    dropout: float = 0.0
    kernel_size: int = 10

    @nn.compact
    def __call__(self, h_in, train: bool):
        B, L, C = h_in.shape
        skips = []
        h = h_in
        for _ in range(self.block_depth):
            h = ResnetBlock(features=self.features,
                            dropout=self.dropout,
                            kernel_size=self.kernel_size
                            )(h, train=train)
            skips.append(h)

        h = nn.avg_pool(h, (2,), strides=(2,))
        return [h, skips] if self.return_skips else h


class UpBlock(nn.Module):
    features: int = None
    block_depth: int = None
    upscale_factor: int = 2
    dropout: float = 0.0
    kernel_size: int = 10

    @nn.compact
    def __call__(self, x, skips=None, train: bool = True):
        B, L, C = x.shape  # seperate in Batch, length, and channels
        x = jax.image.resize(
            x, shape=(B, L * self.upscale_factor, C), method="bilinear")
        for _ in range(self.block_depth):
            if skips != None:
                x = jnp.concatenate([x, skips.pop()], axis=-1)
            x = ResnetBlock(features=self.features,
                            kernel_size=self.kernel_size)(x, train=train)
        return x
