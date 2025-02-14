from dataclasses import field
import math
from typing import List
import flax.linen as nn
import jax
import jax.numpy as jnp

from model.attention import efficient_dot_product_attention

from model.resnet_blocks import DownBlock, ResnetBlock, UpBlock


class UNet30s(nn.Module):
    embedding_dims: int = 16
    feature_sizes: List[int] = field(default_factory= lambda: [96, 128, 160])
    block_depths: int = 2
    attention_depths: int = 4
    
    @nn.compact
    def __call__(self, x, labels, variance, train:bool=True):
        B, L, C = x.shape
        #Input shape is B, 480, 16
        embedded_variance = SinEmbed(embedding_dims=self.embedding_dims)(variance)
        # embedded_label = SinEmbed(embedding_dims=self.embedding_dims)(labels)
        # print(labels.shape)
        embedded_label = LearnableEmbedding(num_embeddings=2, embedding_dim=self.embedding_dims)(labels)
        # print(embedded_label.shape)
        # quit()
        #embedded_variance = jnp.repeat(embedded_variance, L, axis = 1)
        
        h = nn.Conv(32, kernel_size=[6])(x)
        

        
        #go down
        skips = []
        for index, features in enumerate(self.feature_sizes[:-1]):
            #Concat the variance to the input
            B, L, C = h.shape
            emb_var_repeated = jnp.repeat(embedded_variance, L, axis = 1)
            #emb_label_repetaed = jnp.repeat(embedded_label, L, axis = 1)
            # print(emb_label_long.shape)
            # print(poistional_mod.shape)
            # print(emb_label_pos.shape)
            # print(h.shape)
            # print(emb_var_repeated.shape)
            # quit()
            h = jnp.concatenate([h, emb_var_repeated], axis=-1)
            
            #add label information
            emb_label_long = expand_embedding(embedded_label, L)
            poistional_mod =  PositionalModulation(length=L, embedding_dim=self.embedding_dims)()
            emb_label_pos = emb_label_long + poistional_mod
            #project into the same space as the input
            emb_labels = nn.Dense(C + self.embedding_dims)(emb_label_pos)
            h = h + emb_labels
            
            
            
            h, skip = DownBlock(features=features, block_depth=self.block_depths, return_skips=True)(h, train=train)
            
            # if index > self.attention_depths:
            skips.append(skip)
        
        for _ in range(self.block_depths):
            B, L, C = h.shape
            emb_var_repeated = jnp.repeat(embedded_variance, L, axis = 1)
            # emb_label_repetaed = jnp.repeat(embedded_label, L, axis = 1)
            
            h = jnp.concatenate([h, emb_var_repeated], axis=-1)
            
            #add label information
            emb_label_long = expand_embedding(embedded_label, L)
            poistional_mod =  PositionalModulation(length=L, embedding_dim=self.embedding_dims)()
            emb_label_pos = emb_label_long + poistional_mod
            #project into the same space as the input
            emb_labels = nn.Dense(C + self.embedding_dims)(emb_label_pos)
            h = h + emb_labels
            
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
            # emb_label_repetaed = jnp.repeat(embedded_label, L, axis = 1)
            h = jnp.concatenate([h, emb_var_repeated], axis=-1)
            
            #add label information
            emb_label_long = expand_embedding(embedded_label, L)
            poistional_mod =  PositionalModulation(length=L, embedding_dim=self.embedding_dims)()
            emb_label_pos = emb_label_long + poistional_mod
            #project into the same space as the input
            emb_labels = nn.Dense(C + self.embedding_dims)(emb_label_pos)
            h = h + emb_labels
            
            
            h = UpBlock(features=features, block_depth=self.block_depths)(h, skip, train=train)
            # if index < self.attention_depths and self.attention_depths < len(self.feature_sizes):
            
        h = nn.Conv(16, kernel_size=[4], kernel_init=nn.initializers.zeros)(h)
        #h = nn.sigmoid(h)
        #h = nn.sigmoid(h)
        
        return h
    
    
class UNet90s(nn.Module):
    embedding_dims: int = 16
    feature_sizes: List[int] = field(default_factory= lambda: [96, 128, 160])
    block_depths: int = 2
    attention_depths: int = 4
    
    @nn.compact
    def __call__(self, x, variance, train:bool=True):
        B, L, C = x.shape
        #Input shape is B, 1440, 16
        embedded_variance = SinEmbed(embedding_dims=self.embedding_dims)(variance)
        
        h = nn.Conv(32, kernel_size=[6])(x)
        

        
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
            # emb_label_repetaed = jnp.repeat(embedded_label, L, axis = 1)
            
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
            # emb_label_repetaed = jnp.repeat(embedded_label, L, axis = 1)
            h = jnp.concatenate([h, emb_var_repeated], axis=-1)
            
            h = UpBlock(features=features, block_depth=self.block_depths)(h, skip, train=train)
            # if index < self.attention_depths and self.attention_depths < len(self.feature_sizes):
            
        h = nn.Conv(16, kernel_size=[4], kernel_init=nn.initializers.zeros)(h)
        
        return h

class UNet320s(nn.Module):
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



            
        
        h = nn.Conv(16, kernel_size=[4], kernel_init=nn.initializers.zeros)(h)
        #h = nn.sigmoid(h)
        #h = nn.sigmoid(h)
        
        return h

def expand_embedding(embedding, length):
    batch_size, embedding_dim = embedding.shape
    return jnp.broadcast_to(embedding[:, None, :], (batch_size, length, embedding_dim))            
        
class LearnableEmbedding(nn.Module):
    num_embeddings: int  # Number of categories
    embedding_dim: int  # Dimension of each embedding vector

    @nn.compact
    def __call__(self, indices):
        # Initialize embeddings as a learnable parameter
        embeddings = self.param(
            'embeddings',
            nn.initializers.normal(stddev=0.02),
            (self.num_embeddings, self.embedding_dim)
        )
        return embeddings[indices]
    
class PositionalModulation(nn.Module):
    length: int
    embedding_dim: int

    @nn.compact
    def __call__(self):
        pos_embed = self.param(
            "positional_embedding",
            nn.initializers.normal(stddev=0.02),
            (self.length, self.embedding_dim)
        )
        return pos_embed 
        
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