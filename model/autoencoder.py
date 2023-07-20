from dataclasses import field
from typing import List
import jax
import jax.numpy as jnp
import flax.linen as nn
from .resnet_blocks import ResnetBlock, DownBlock, UpBlock


class Encoder(nn.Module):
    block_depth: int = 1

    @nn.compact
    def __call__(self, x, train: bool):
        # Input shape is (B, 30_720, 1)
        # Reshape it to (B, 256, 240)
        B, C, L = x.shape
        reshaped = jnp.reshape(x, (-1, 1024, 30))  # 1024 = 2seconds
        # return ResnetBlock(8, 0, kernel_size=10)(x, train=train)
        # return x

        # Go down to (B, 64, 64)
        down1 = DownBlock(128, block_depth=self.block_depth,
                          return_skips=False)(reshaped, train=train)
        resnet1 = ResnetBlock(128, kernel_size=8)(down1, train=train)
        down2 = DownBlock(128, block_depth=self.block_depth,
                          return_skips=False)(resnet1, train=train)
        resnet2 = ResnetBlock(128, kernel_size=8)(down2, train=train)

        down3 = DownBlock(128, kernel_size=8, block_depth=self.block_depth,
                          return_skips=False)(resnet2, train=train)
        down4 = DownBlock(128, kernel_size=8, block_depth=self.block_depth,
                          return_skips=False)(down3, train=train)
        # down5 = DownBlock(128, kernel_size=4, block_depth=self.block_depth,
        #                   return_skips=False)(down4, train=train)
        # Go down in feature dim only, to arrive at (B, 64, 64)
        # resnet3 = ResnetBlock(64, kernel_size=4)(down5, train=train)
        latent_space = ResnetBlock(64, kernel_size=4)(
            down4, train=train)  # This is the latent space vector
        
        #flattened = latent_space.reshape((B, -1))
        
        #non_stochastic_ls = nn.Dense(2048)(flattened)
        #mean_ls = nn.Dense(2048)(flattened)
        #logvar_ls = nn.Dense(2048, kernel_init=nn.initializers.zeros)(flattened)
        
        #Normalise the latent space:
        #normalised_latent_space = nn.sigmoid(latent_space)
        
        #normalised_latnet_space = nn.LayerNorm()
        #return non_stochastic_ls, mean_ls, logvar_ls
        
        return latent_space
        


class Decoder(nn.Module):
    block_depth: int = 1

    @nn.compact
    def __call__(self, x, train: bool):
        # input shape is (B, 64, 64)
        # Go up again
        # resnet0 = ResnetBlock(64, kernel_size=4)(x, train=train)
        resnet1 = ResnetBlock(128, kernel_size=4)(x, train=train)
        up1 = UpBlock(128, kernel_size=8,
                      block_depth=self.block_depth)(resnet1, train=train)
        resnet2 = ResnetBlock(128, kernel_size=8)(up1, train=train)
        up2 = UpBlock(128, kernel_size=10, block_depth=self.block_depth)(
            resnet2, train=train)
        resnet3 = ResnetBlock(128, kernel_size=8)(up2, train=train)
        up3 = UpBlock(128, kernel_size=10, block_depth=self.block_depth)(
            resnet3, train=train)
        up4 = UpBlock(128, kernel_size=10, block_depth=self.block_depth)(
            up3, train=train)
        # up5 = UpBlock(128, kernel_size=10, block_depth=self.block_depth)(
        #     up4, train=train)
        output = nn.Conv(30, kernel_size=[4],
                         kernel_init=nn.initializers.zeros)(up4)
        B, L, C = output.shape
        reshaped = jnp.reshape(output, newshape=(B, L * C, 1))
        return reshaped
    
    
def reparameterize(z_mean, z_log_var, rng):
        B, C = z_mean.shape
        epsilon = jax.random.normal(rng, shape=(B, C))
        
        return z_mean + jnp.exp(0.5 * z_log_var) * epsilon


class AutoEncoder(nn.Module):
    block_depths: int = 2
    sample_rng: jax.random.KeyArray = jax.random.PRNGKey(0)

    def setup(self):
        self.encoder = Encoder(block_depth=self.block_depths)
        self.decoder = Decoder(block_depth=self.block_depths)

    def __call__(self, batch, train: bool = True):
        B, L = batch.shape
        input_batch = jnp.reshape(batch, (B, L, 1))
        latent_space = self.encoder(input_batch, train=train)
        #deterministic_ls, mean_ls, log_var_ls = self.encoder(input_batch, train=train)
        #sample_rng, rng = jax.random.split(self.sample_rng)
        #sampled = reparameterize(mean_ls, log_var_ls, sample_rng)
        
        #concat = jnp.concatenate([deterministic_ls, sampled], axis=-1) #(B, 4096)
        #sampled_reshaped = concat.reshape((B, 64, 64))
        output = self.decoder(latent_space, train=train)
        B, L, C = output.shape
        output = jnp.reshape(output, (B, L))
        #return output, (deterministic_ls, mean_ls, log_var_ls)
        return output, latent_space
    
    def encode(self, batch, train: bool = False):
        return self.encoder(batch, train=train)
    
    def decode(self, batch, train: bool = False):
        return self.decoder(batch, train=train)
