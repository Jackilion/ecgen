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
        # Input shape is (B, 2048, 1)
        B, C, L = x.shape
        #reshaped = jnp.reshape(x, (-1, 1024, 30))  # 1024 = 2seconds

        # Go down to (B, 256, 1)
        down1 = DownBlock(32, kernel_size=32, block_depth=self.block_depth, return_skips=False)(x, train=train) #(B, 1024, 32)
        down2 = DownBlock(32, kernel_size=16, block_depth=self.block_depth, return_skips=False)(down1, train=train) #(B, 512, 32)
        down3 = DownBlock(32, kernel_size=8, block_depth=self.block_depth, return_skips=False)(down2, train=train) #(B, 256, 32)
        
        resnet1 = nn.Conv(16, kernel_size=[8])(down3) #(B, 256, 16)
        resnet2 = nn.Conv(8, kernel_size=[8])(resnet1) #(B, 256, 8)
        output = nn.Conv(1, kernel_size=[1])(resnet2) #(B, 256, 1)
        return output
        down1 = DownBlock(32, block_depth=self.block_depth,
                          return_skips=False)(x, train=train) #1024, 16
        # resnet1 = ResnetBlock(16, kernel_size=8)(down1, train=train)
        down2 = DownBlock(32, block_depth=self.block_depth,
                          return_skips=False)(down1, train=train) #512, 16
        # resnet2 = ResnetBlock(16, kernel_size=8)(down2, train=train)

        down3 = DownBlock(32, kernel_size=8, block_depth=self.block_depth,
                          return_skips=False)(down2, train=train) #256, 16

        # resnet3 = ResnetBlock(16, kernel_size=8)(down3, train=train)
        down4 = DownBlock(32, kernel_size=4, block_depth=self.block_depth, return_skips=False)(down3, train=train) #128, 8
        
        # resnet4 = ResnetBlock(8, kernel_size=4)(down4, train=train)
        down5 = DownBlock(16, kernel_size=4, block_depth=self.block_depth, return_skips=False)(down4, train=train) # B, 64, 4
        
        resnet5 = ResnetBlock(4, kernel_size=4)(down5, train=train)
        
        latent_space = jnp.reshape(resnet5, (B, 256, 1)) # This is the latent space vector, B, 256
        
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
        # input shape is (B, 256, 1)
        # Go up again
        # resnet0 = ResnetBlock(64, kernel_size=4)(x, train=train)
        
        resnet1 = nn.Conv(8, kernel_size=[4])(x) #(B, 256, 8)
        resnet2 = nn.Conv(16, kernel_size=[8])(resnet1) #(B, 256, 16)
        
        up1 = UpBlock(32, kernel_size=8, block_depth=self.block_depth)(resnet2, train=train) #(B, 512, 32)
        up2 = UpBlock(32, kernel_size=16, block_depth=self.block_depth)(up1, train=train) #(B, 1024, 32)
        up3 = UpBlock(32, kernel_size=32, block_depth=self.block_depth)(up2, train=train) #(B, 2048, 32)
        output = nn.Conv(1, kernel_size=[1])(up3)
        return output
        
        
        B, L, C = x.shape
        reshaped = jnp.reshape(x, (B, 64, 4))
        up1 = UpBlock(8, kernel_size=4,
                      block_depth=self.block_depth)(reshaped, train=train) #128, 4
        resnet2 = ResnetBlock(8, kernel_size=4)(up1, train=train)
        up2 = UpBlock(16, kernel_size=4, block_depth=self.block_depth)(
            up1, train=train) #256, 16
        resnet3 = ResnetBlock(16, kernel_size=8)(up2, train=train)
        up3 = UpBlock(32, kernel_size=8, block_depth=self.block_depth)(
            up2, train=train) #512, 16
        resnet4 = ResnetBlock(16, kernel_size=8)(up3, train=train)
        
        
        up4 = UpBlock(32, kernel_size=8, block_depth=self.block_depth)(
            up3, train=train) #1024, 16
        resnet5 = ResnetBlock(16, kernel_size=16)(up4, train=train)
        up5 = UpBlock(32, kernel_size=16, block_depth=self.block_depth)(resnet5, train=train) #2048, 16
        output = nn.Conv(1, kernel_size=[1])(up5)
        # output = nn.Conv(30, kernel_size=[4],
        #                  kernel_init=nn.initializers.zeros)(up4)
        #B, L, C = output.shape
        #reshaped = jnp.reshape(output, newshape=(B, L * C, 1))
        return output
    
    
def reparameterize(z_mean, z_log_var, rng):
        B, C = z_mean.shape
        epsilon = jax.random.normal(rng, shape=(B, C))
        
        return z_mean + jnp.exp(0.5 * z_log_var) * epsilon


class AutoEncoder(nn.Module):
    block_depths: int = 1
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
