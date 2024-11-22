from dataclasses import field
from typing import List
import jax
import jax.numpy as jnp
import flax.linen as nn
from .resnet_blocks import ResnetBlock, DownBlock, UpBlock

nonlinearity = nn.swish

class Encoder(nn.Module):
    block_depth: int = 1

    @nn.compact
    def __call__(self, x, train: bool):
        # Input shape is (B, 2560, 1)
        B, C, L = x.shape

                                                
        down1 = DownBlock(64, kernel_size=32, block_depth=self.block_depth, return_skips=False)(x, train=train) #(B, 1280, 64)
        down2 = DownBlock(64, kernel_size=16, block_depth=self.block_depth, return_skips=False)(down1, train=train) #(B, 640, 64)
        down3 = DownBlock(64, kernel_size=8, block_depth=self.block_depth, return_skips=False)(down2, train=train) #(B, 320, 64)
        down4 = DownBlock(64, kernel_size=8, block_depth=self.block_depth, return_skips=False)(down3, train=train) #(B, 160, 64)
        down5 = DownBlock(64, kernel_size=4, block_depth=self.block_depth, return_skips=False)(down4, train=train) #(B, 80, 64)
        down6 = DownBlock(64, kernel_size=4, block_depth=self.block_depth, return_skips=False)(down5, train=train) #(B, 40, 64)
        down7 = ResnetBlock(32, kernel_size=4)(down6, train=train) #(B, 40, 32)
        down8 = ResnetBlock(16, kernel_size=2)(down7, train=train) #(B, 40, 16)
        return down8



class Decoder(nn.Module):
    block_depth: int = 1

    @nn.compact
    def __call__(self, x, train: bool):
        # input shape is (B, 40, 16), output should be (B, 2560, 1)
        B, L, C = x.shape  # seperate in Batch, length, and channels
        up1 = UpBlock(32, block_depth=self.block_depth, upscale_factor=2)(x, train=train) #(B, 80, 32)
        up2 = UpBlock(64, block_depth=self.block_depth, upscale_factor=2)(up1, train=train) #(B, 160, 64)
        up3 = UpBlock(64, block_depth=self.block_depth, upscale_factor=2)(up2, train=train) #(B, 320, 64)
        up4 = UpBlock(64, block_depth=self.block_depth, upscale_factor=2)(up3, train=train) #(B, 640, 64)
        up5 = UpBlock(64, block_depth=self.block_depth, upscale_factor=2)(up4, train=train) #(B, 1280, 64)
        up6 = UpBlock(64, block_depth=self.block_depth, upscale_factor=2)(up5, train=train) #(B, 2560, 64)
        up7 = ResnetBlock(32, kernel_size=4)(up6, train=train) #(B, 2560, 32)
        up8 = nn.Conv(1, kernel_size=(1,))(up7) #(B, 2560, 1)
        return up8

class Quantizer(nn.Module):
    embed_size_K: int
    embed_dim_D: int
    commitment_loss_beta: float = 0.025

    def setup(self):
        self.codebook = self.param('embedding_space', nn.initializers.variance_scaling(scale=1, mode="fan_avg", distribution="uniform"), (self.embed_size_K, self.embed_dim_D) )
        
    
    def __call__(self, z_e):
        """_summary_

        Args:
            z_e (Tensor): The outputs of the encoder

        Returns:
            _type_: _description_
        """
        # Shape (K, D)
        
        #print(f"codebook shape: {codebook.shape}")
        #print(f"z_e shape: {z_e.shape}")
        
        
        #print(f"z_e shape: {z_e.shape}")
        flattened = jnp.reshape(z_e, (-1, self.embed_dim_D))
        
        #print(f"flattened shape: {flattened.shape}")
        # shape N x 1
        flattened_sqr = jnp.sum(flattened**2, axis=-1, keepdims=True)
        
        #print(f"flattened_sqr shape: {flattened_sqr.shape}")
        
        
        # shape 1 x K
        codebook_sqr = jnp.sum(self.codebook**2, axis=-1, keepdims=True).T
        
        #print(f"codebook_sqr shape: {codebook_sqr.shape}")
        
        
        # shape N x K
        distances = flattened_sqr - 2 * (flattened @ self.codebook.T) + codebook_sqr # (a-b)^2
        
        #print(f"distances shape: {distances.shape}")
        
        
        # shape A1 x ... x An
        encoding_indices = jnp.reshape(jnp.argmin(distances, axis=-1), z_e.shape[:-1])
        
        #print(f"encoding_indices shape: {encoding_indices.shape}")

        
        #shape A1 x ... x An x D
        quantize = self.codebook[encoding_indices]
        
        #print(f"quantize shape: {quantize.shape}")
        
        # loss = ||sg[z_e(x)] - e|| + beta ||z_e(x) - sg[e]||
        encoding_loss = jnp.mean((jax.lax.stop_gradient(z_e) - quantize)**2)
        commitment_loss = jnp.mean((z_e - jax.lax.stop_gradient(quantize)) ** 2)
        loss = encoding_loss + self.commitment_loss_beta * commitment_loss
        
        # this is here so the gradients can flow from decoder to encoder for the reconstruction loss
        #quantize_expanded = jnp.expand_dims(quantize, -1)
        z_q = z_e + jax.lax.stop_gradient(quantize - z_e)
        
        #print(f"z_q shape: {z_q.shape}")
        
        return z_q, encoding_indices, loss
    
    def embed(self, indices):
        #codebook = self.param('embedding_space', nn.initializers.variance_scaling(scale=1, mode="fan_avg", distribution="uniform"), (self.embed_size_K, self.embed_dim_D) )
        
        outshape = indices.shape + (self.embed_dim_D,)
        x = self.codebook[indices].reshape(outshape)
        return x
                                    
        
class AutoEncoder(nn.Module):
    block_depths: int = 1
    embed_size_K: int = 64
    embed_dim_D: int = 8
    commitment_loss_beta: float = 0.25
    #sample_rng = jax.random.PRNGKey(0)

    def setup(self):
        self.encoder = Encoder(block_depth=self.block_depths)
        self.quantizer = Quantizer(embed_dim_D=self.embed_dim_D, embed_size_K=self.embed_size_K, commitment_loss_beta=self.commitment_loss_beta)
        self.decoder = Decoder(block_depth=self.block_depths)

    def __call__(self, batch, train: bool = True):
        B, L = batch.shape
        input_batch = jnp.reshape(batch, (B, L, 1))
        #latent_space = self.encoder(input_batch, train=train)
        z_e = self.encoder(input_batch, train=train)
        
        z_q, encoding_indices, embedding_space_loss = self.quantizer(z_e)
        #concat = jnp.concatenate([deterministic_ls, sampled], axis=-1) #(B, 4096)
        #sampled_reshaped = concat.reshape((B, 64, 64))
        output = self.decoder(z_q, train=train)
        B, L, C = output.shape
        output = jnp.reshape(output, (B, L))
        #return output, (deterministic_ls, mean_ls, log_var_ls)
        return output, z_e, embedding_space_loss
    def encode(self, batch, train: bool = False):
        B, L = batch.shape
        input_batch = jnp.reshape(batch, (B, L, 1))
        
        z_e = self.encoder(input_batch, train=train)
        z_q, encoding_indices, _ = self.quantizer(z_e)
        
        return z_q, encoding_indices
        
    def embed_indices(self, indices):
        return self.quantizer.embed(indices)
    
    def embed(self, vectors):
        z_q, ind, loss = self.quantizer(vectors)
        return z_q
    
    def decode(self, batch, train: bool = False):
        return self.decoder(batch, train=train)
