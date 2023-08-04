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
        
        # conv1 = nn.Conv(32, kernel_size= [33], padding="VALID")(x)
        # down1 = nn.avg_pool(conv1, (2,), strides=(2,))
        # conv2 = nn.Conv(32, kernel_size=[17], padding="VALID")(down1)
        # down2 = nn.avg_pool(conv2, (2,), strides=(2,)) #B, 496, 32
        # conv3 = nn.Conv(32, kernel_size=[9], padding="VALID")(down2) #B, 488, 32
        # down3 = nn.avg_pool(conv3, (2,), strides=(2,)) #B, 244, 32
        # dim1 = nn.Conv(1, kernel_size=[1])(down3) #B, 244, 1
        # reshaped = jax.image.resize(dim1, (B, 256, 1), method="bilinear")
        # sigmoided = nn.sigmoid(reshaped)
        # return sigmoided
        # down1 = nn.avg_pool(conv1, (4,), strides=(4,)) # B, 512, 32
        # conv2 = nn.Conv(32, kernel_size=[16])(down1)
        # down2 = nn.avg_pool(conv2, (2,), strides=(2,))
        # conv3 = nn.Conv(32, kernel_size=[8])(down2)
        # output = nn.Conv(1, kernel_size=[1])(conv3)
        # sigmoided = nn.sigmoid(output)
        # return sigmoided
                                                
        down1 = DownBlock(32, kernel_size=32, block_depth=self.block_depth, return_skips=False)(x, train=train) #(B, 1024, 32)
        down2 = DownBlock(32, kernel_size=16, block_depth=self.block_depth, return_skips=False)(down1, train=train) #(B, 512, 32)
        down3 = DownBlock(32, kernel_size=8, block_depth=self.block_depth, return_skips=False)(down2, train=train) #(B, 256, 32)
        
        resnet1 = nn.Conv(16, kernel_size=[8])(down3) #(B, 256, 16)
        resnet2 = nn.Conv(8, kernel_size=[8])(resnet1) #(B, 256, 8)
        output_conv = nn.Conv(1, kernel_size=[1])(resnet2) #(B, 256, 1)
        sigmoided = nn.sigmoid(output_conv)
        
        return sigmoided
        flattened = jnp.reshape(output_conv, (B, -1))
        mean_ls = nn.Dense(256)(flattened)
        logvar_ls = nn.Dense(256)(flattened)
        
        return mean_ls, logvar_ls  


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
        sigmoided = nn.sigmoid(output)
        return output
    
    
def reparameterize(z_mean, z_log_var, rng):
        B, L = z_mean.shape
        epsilon = jax.random.normal(rng, shape=(B, L))
        
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
        #latent_space = self.encoder(input_batch, train=train)
        mean_ls, log_var_ls = self.encoder(input_batch, train=train)
        sample_rng, rng = jax.random.split(self.sample_rng)
        sampled = reparameterize(mean_ls, log_var_ls, sample_rng) #(B, 256)
        sampled_reshaped = sampled.reshape((B, -1, 1))
        
        #concat = jnp.concatenate([deterministic_ls, sampled], axis=-1) #(B, 4096)
        #sampled_reshaped = concat.reshape((B, 64, 64))
        output = self.decoder(sampled_reshaped, train=train)
        B, L, C = output.shape
        output = jnp.reshape(output, (B, L))
        #return output, (deterministic_ls, mean_ls, log_var_ls)
        return output, (mean_ls, log_var_ls)
    
    def encode(self, batch, train: bool = False):
        B, L = batch.shape
        input_batch = jnp.reshape(batch, (B, L, 1))
        
        return self.encoder(input_batch, train=train)
    
    def decode(self, batch, train: bool = False):
        return self.decoder(batch, train=train)
