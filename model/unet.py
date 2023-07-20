from dataclasses import field
import math
from typing import List
import flax.linen as nn
import jax
import jax.numpy as jnp

from model.resnet_blocks import DownBlock, ResnetBlock, UpBlock

class UNet(nn.Module):
    embedding_dims: int = 32
    feature_sizes: List[int] = field(default_factory= lambda: [96, 128, 160])
    block_depths: int = 2
    
    @nn.compact
    def __call__(self, x, variance, train:bool=True):
        B, L, C = x.shape
        embedded_variance = SinEmbed(embedding_dims=self.embedding_dims)(variance)
        embedded_variance = jnp.repeat(embedded_variance, L, axis = 1)
        h = jnp.concatenate([x, embedded_variance], axis=-1)
        
        #start with some convolutions
        for i in range(5):
            h = ResnetBlock(features=96, kernel_size= 20 - i)(h, train=train)
            h = nn.swish(h)
        
        #go down
        skips = []
        for _, features in enumerate(self.feature_sizes[:-1]):
            h, skip = DownBlock(features=features, block_depth=self.block_depths, return_skips=True)(h, train=train)
            skips.append(skip)
            
        for _ in range(self.block_depths):
            h = ResnetBlock(self.feature_sizes[-1])(h, train=train)
        
        #go up
        for _, features in enumerate(reversed(self.feature_sizes[:-1])):
            skip = skips.pop()
            h = UpBlock(features=features, block_depth=self.block_depths)(h, skip, train=train)
        
        h = nn.Conv(64, kernel_size=[4], kernel_init=nn.initializers.zeros)(h)
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