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
        down4 = DownBlock(32, kernel_size=4, block_depth=self.block_depth, return_skips=False)(down3, train=train) #(B, 128, 32)
        down5 = DownBlock(32, kernel_size=4, block_depth=self.block_depth, return_skips=False)(down4, train=train) #(B, 64, 32)
        down6 = DownBlock(32, kernel_size=4, block_depth=self.block_depth, return_skips=False)(down5, train=train) #(B, 32, 32)
        
        resnet1 = nn.Conv(16, kernel_size=[8])(down6) #(B, 32, 16)
        resnet2 = nn.Conv(8, kernel_size=[8])(resnet1) #(B, 32, 8)
        #output_conv = nn.Conv(1, kernel_size=[1])(resnet2) #(B, 256, 1)
        #sigmoided = nn.sigmoid(output_conv)
        
        return resnet2
        flattened = jnp.reshape(output_conv, (B, -1))
        mean_ls = nn.Dense(256)(flattened)
        logvar_ls = nn.Dense(256)(flattened)
        
        return mean_ls, logvar_ls  


class Decoder(nn.Module):
    block_depth: int = 1

    @nn.compact
    def __call__(self, x, train: bool):
        # input shape is (B, 32, 8)
        # Go up again
        # resnet0 = ResnetBlock(64, kernel_size=4)(x, train=train)
        
        #resnet1 = nn.Conv(8, kernel_size=[4])(x) #(B, 256, 8)
        resnet2 = nn.Conv(16, kernel_size=[8])(x) #(B, 256, 16)
        
        up_2 = UpBlock(32, kernel_size=4, block_depth=self.block_depth)(resnet2, train=train) #(B, 64, 32)
        up_1 = UpBlock(32, kernel_size=4, block_depth=self.block_depth)(up_2, train=train) #(B, 128, 32)

        up0 = UpBlock(32, kernel_size=4, block_depth=self.block_depth)(up_1, train=train) #(B, 256, 32)
        up1 = UpBlock(32, kernel_size=8, block_depth=self.block_depth)(up0, train=train) #(B, 512, 32)
        up2 = UpBlock(32, kernel_size=16, block_depth=self.block_depth)(up1, train=train) #(B, 1024, 32)
        up3 = UpBlock(32, kernel_size=32, block_depth=self.block_depth)(up2, train=train) #(B, 2048, 32)
        output = nn.Conv(1, kernel_size=[1])(up3)
        sigmoided = nn.sigmoid(output)
        return sigmoided

class Quantizer(nn.Module):
    embed_size_K: int
    embed_dim_D: int
    commitment_loss_beta: float = 0.025

    @nn.compact
    def __call__(self, z_e):
        """_summary_

        Args:
            z_e (Tensor): The outputs of the encoder

        Returns:
            _type_: _description_
        """
        # Shape (K, D)
        codebook = self.param('embedding_space', nn.initializers.variance_scaling(scale=1, mode="fan_avg", distribution="uniform"), (self.embed_size_K, self.embed_dim_D) )
        
        #print(f"codebook shape: {codebook.shape}")
        #print(f"z_e shape: {z_e.shape}")
        
        # ###
        # flattened_z_e = z_e.reshape((-1, self.embed_dim_D))
        # distances = (flattened_z_e - codebook) ** 2
        # min_ind = jnp.argmin(distances)
        # print(distances.shape)
        # print(min_ind.shape)
        # quit()
        # ###
        # return
        
        #print(f"z_e shape: {z_e.shape}")
        flattened = jnp.reshape(z_e, (-1, self.embed_dim_D))
        
        #print(f"flattened shape: {flattened.shape}")
        # shape N x 1
        flattened_sqr = jnp.sum(flattened**2, axis=-1, keepdims=True)
        
        #print(f"flattened_sqr shape: {flattened_sqr.shape}")
        
        
        # shape 1 x K
        codebook_sqr = jnp.sum(codebook**2, axis=-1, keepdims=True).T
        
        #print(f"codebook_sqr shape: {codebook_sqr.shape}")
        
        
        # shape N x K
        distances = flattened_sqr - 2 * (flattened @ codebook.T) + codebook_sqr # (a-b)^2
        
        #print(f"distances shape: {distances.shape}")
        
        
        # shape A1 x ... x An
        encoding_indices = jnp.reshape(jnp.argmin(distances, axis=-1), z_e.shape[:-1])
        
        #print(f"encoding_indices shape: {encoding_indices.shape}")

        
        #shape A1 x ... x An x D
        quantize = codebook[encoding_indices]
        
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
        codebook = self.param('embedding_space', nn.initializers.variance_scaling(distribution="uniform") )
        
        outshape = indices.shape + (self.embed_dim_D,)
        x = codebook[indices].reshape(outshape)
        return x
                                    
        
class AutoEncoder(nn.Module):
    block_depths: int = 1
    sample_rng: jax.random.KeyArray = jax.random.PRNGKey(0)

    def setup(self):
        self.encoder = Encoder(block_depth=self.block_depths)
        self.quantizer = Quantizer(embed_dim_D=8, embed_size_K=64)
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
        return output, z_q, embedding_space_loss
    def encode(self, batch, train: bool = False):
        B, L = batch.shape
        input_batch = jnp.reshape(batch, (B, L, 1))
        
        z_e = self.encoder(input_batch, train=train)
        z_q, encoding_indices, _ = self.quantizer(z_e)
        
        return z_q, encoding_indices
        
    
    def decode(self, batch, train: bool = False):
        return self.decoder(batch, train=train)
