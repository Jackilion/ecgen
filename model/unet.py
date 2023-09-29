from dataclasses import field
import math
from typing import List
import flax.linen as nn
import jax
import jax.numpy as jnp

from model.attention import efficient_dot_product_attention

from model.resnet_blocks import DownBlock, ResnetBlock, UpBlock

class UNet(nn.Module):
    embedding_dims: int = 32
    feature_sizes: List[int] = field(default_factory= lambda: [96, 128, 160])
    block_depths: int = 2
    attention_depths: int = 4
    
    @nn.compact
    def __call__(self, x, variance, train:bool=True):
        B, L, C = x.shape
        #Input shape is B, 5120, 8
        embedded_variance = SinEmbed(embedding_dims=self.embedding_dims)(variance)
        #embedded_variance = jnp.repeat(embedded_variance, L, axis = 1)
        
        h = nn.Conv(32, kernel_size=[4])(x)

        
        #go down
        skips = []
        for index, features in enumerate(self.feature_sizes[:-1]):
            #Concat the variance to the input
            B, L, C = h.shape
            emb_var_repeated = jnp.repeat(embedded_variance, L, axis = 1)
            h = jnp.concatenate([h, emb_var_repeated], axis=-1)
            h, skip = DownBlock(features=features, block_depth=self.block_depths, return_skips=True)(h, train=train)
            
            # if index > self.attention_depths:
            skips.append(skip)
        
        for _ in range(self.block_depths):
            B, L, C = h.shape
            emb_var_repeated = jnp.repeat(embedded_variance, L, axis = 1)
            h = jnp.concatenate([h, emb_var_repeated], axis=-1)
            
            h = nn.SelfAttention(4)(h)
            h = ResnetBlock(self.feature_sizes[-1])(h, train=train)
            h = nn.SelfAttention(4)(h)
            h = ResnetBlock(self.feature_sizes[-1] // 2)(h, train=train)
            # h = ResnetBlock(self.feature_sizes[-1])(h, train=train)
            # h = nn.SelfAttention(4)(h)

        
        #go up
        for index, features in enumerate(reversed(self.feature_sizes[:-1])):
            skip = skips.pop()
            B, L, C = h.shape
            emb_var_repeated = jnp.repeat(embedded_variance, L, axis = 1)
            h = jnp.concatenate([h, emb_var_repeated], axis=-1)
            h = UpBlock(features=features, block_depth=self.block_depths)(h, skip, train=train)
            # if index < self.attention_depths and self.attention_depths < len(self.feature_sizes):



            
        
        h = nn.Conv(8, kernel_size=[4], kernel_init=nn.initializers.zeros)(h)
        #h = nn.sigmoid(h)
        #h = nn.sigmoid(h)
        
        return h
            
        
        
        
class SinEmbed(nn.Module):
    """
    Embeds an input through Sin and Cos.
    Outputs a Tensor of Shape (BatchSize, 1, EmbedDims)
    """
    embedding_dims: int = 32
    embedding_max_frequency: float = 1000.0
    embedding_min_frequency: float = 1.0

    @nn.compact
    def __call__(self, x):
        frequencies = jnp.exp(
            jnp.linspace(
                jnp.log(self.embedding_min_frequency),
                jnp.log(self.embedding_max_frequency),
                self.embedding_dims // 2
            )
        )

        angular_speeds = 2.0 * math.pi * frequencies
        angular_speeds = jnp.expand_dims(angular_speeds, 0)

        embeddings = jnp.concatenate(
            [
                jnp.sin(angular_speeds * x),
                jnp.cos(angular_speeds * x)
            ],
            axis=2
        )
        return embeddings
    
    
class MeanNNEmbed(nn.Module):
    """
    Embeds the meanNN of the ECG through a sin embedding
    """
    embedding_dims: int = 32
    
    @nn.compact
    def __call__(self, x):
        #x is of shape (B, 1, 1)
        embed = SinEmbed(embedding_dims=self.embedding_dims)(x)
        return embed