from dataclasses import field
from typing import List
import jax
import jax.numpy as jnp
import flax.linen as nn
from .resnet_blocks import ResnetBlock, DownBlock, UpBlock

nonlinearity = nn.swish

class Encoder(nn.Module):
    block_depth: int = 1
    convolution_filters: List[int] = field(default_factory=lambda: [64, 64, 64, 64, 64, 64])
    kernel_sizes: List[int] = field(default_factory=lambda: [32, 16, 8, 4, 4, 4])
    embed_dimension: int = 32
    dropout: float = 0.1
    
    @nn.compact
    def __call__(self, x, train: bool):
        # Input shape is (B, 2560, 1)
        B, C, L = x.shape
        
        for i in range(len(self.convolution_filters)):
            x = DownBlock(self.convolution_filters[i], kernel_size=self.kernel_sizes[i], block_depth=self.block_depth, dropout=self.dropout, return_skips=False)(x, train=train)
        
        x = ResnetBlock(self.embed_dimension, dropout=self.dropout, kernel_size=4)(x, train=train)
        x = ResnetBlock(self.embed_dimension, kernel_size=4, dropout=self.dropout)(x, train=train)                                
        
        return x



class Decoder(nn.Module):
    block_depth: int = 1
    convolution_filters: List[int] = field(default_factory=lambda: [64, 64, 64, 64, 64, 64])
    kernel_sizes: List[int] = field(default_factory=lambda: [32, 16, 8, 4, 4, 4])
    embed_dimension: int = 32
    dropout: float = 0.1
    
    @nn.compact
    def __call__(self, x, train: bool):
        # input shape is (B, 40, 16), output should be (B, 2560, 1)
        B, L, C = x.shape  # seperate in Batch, length, and channels
        #
    
        #iterate backwards
        for i in range(len(self.convolution_filters)-1, -1, -1):
            x = UpBlock(self.convolution_filters[i], kernel_size=self.kernel_sizes[i], block_depth=self.block_depth, dropout=self.dropout)(x, train=train)
        
        x = nn.Conv(1, kernel_size=(1,))(x) # (B, 2560, 1)
        
        return x
        

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
    convolution_filters: List[int] = field(default_factory=lambda: [64, 64, 64, 64, 64, 64])
    kernel_sizes: List[int] = field(default_factory=lambda: [32, 16, 8, 4, 4, 4])
    dropout: float = 0.1
    #sample_rng = jax.random.PRNGKey(0)

    def setup(self):
        self.encoder = Encoder(block_depth=self.block_depths, convolution_filters=self.convolution_filters, kernel_sizes=self.kernel_sizes, embed_dimension=self.embed_dim_D, dropout=self.dropout)
        self.quantizer = Quantizer(embed_dim_D=self.embed_dim_D, embed_size_K=self.embed_size_K, commitment_loss_beta=self.commitment_loss_beta)
        self.decoder = Decoder(block_depth=self.block_depths, convolution_filters=self.convolution_filters, kernel_sizes=self.kernel_sizes, embed_dimension=self.embed_dim_D, dropout=self.dropout)

    def __call__(self, batch, train: bool = True, return_encoding_indices=False):
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
        if return_encoding_indices:
            return output, z_e, embedding_space_loss, encoding_indices
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
