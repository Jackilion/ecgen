from dataclasses import field
from typing import List
import flax.linen as nn
import jax
import jax.numpy as jnp

from model.unet import UNet


from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_integer("DDIM_gen_diffusion_steps", 29, "The amount of times the noise goes through the model during inference time")


class DiffusionModel(nn.Module):
    feature_sizes: List[int] = field(default_factory= lambda: [64, 96, 128])
    block_depths: int = 2
    start_log_snr: float = 2.5
    end_log_snr: float = -7.5
    schedule_type: str = "linear"
    
    noise_mu: float = 0 #0.5
    noise_sigma: float = 5 #0.05
    
    
    def setup(self):
        self.network = UNet(feature_sizes=self.feature_sizes, block_depths=self.block_depths)
        
    def __call__(self, batch, rng, train: bool):
        B, L, C = batch.shape
        rng, t_rng = jax.random.split(rng)
        diffusion_times = jax.random.uniform(t_rng, (B, 1, 1))
        
        rng, n_rng = jax.random.split(rng)
        noises = jax.random.normal(n_rng, (B, L, C), dtype = batch.dtype)
        noises = noises * self.noise_sigma + self.noise_mu
        
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        noisy_batch = signal_rates * batch + noise_rates * noises
        
        
        pred_noises, pred_series = self.denoise(noisy_batch, noise_rates, signal_rates, train=train)
        
        return batch, noises, pred_noises, pred_series
        
    
    def diffusion_schedule(self, diffusion_times):
        start_snr = jnp.exp(self.start_log_snr)
        end_snr = jnp.exp(self.end_log_snr)
        
        start_noise_power = 1.0 / (1.0 + start_snr)
        end_noise_power = 1.0 / (1.0 + end_snr)
        
        if self.schedule_type == "linear":
            noise_powers = start_noise_power + diffusion_times * (
                end_noise_power - start_noise_power
            )

        elif self.schedule_type == "cosine":
            start_angle = jnp.arcsin(start_noise_power ** 0.5)
            end_angle = jnp.arcsin(end_noise_power ** 0.5)
            diffusion_angles = start_angle + \
                diffusion_times * (end_angle - start_angle)

            noise_powers = jnp.sin(diffusion_angles) ** 2

        elif self.schedule_type == "log-snr-linear":
            noise_powers = start_snr ** diffusion_times / (
                start_snr * end_snr**diffusion_times + start_snr ** diffusion_times
            )

        else:
            raise NotImplementedError("Unsupported sampling schedule")
        
        #signal + noise = 1
        signal_powers = 1.0 - noise_powers
        
        signal_rates = signal_powers**0.5
        noise_rates = noise_powers**0.5
        
        return noise_rates, signal_rates
    
    def denoise(self, noisy_batch, noise_rates, signal_rates, train: bool):
        pred_batch = self.network(noisy_batch, noise_rates**2,  train=train)
        
        pred_noises = (noisy_batch-pred_batch*signal_rates)/noise_rates
        #pred_batch = (noisy_batch - noise_rates * pred_noises) / signal_rates
        
        return pred_noises,pred_batch
    
    def reverse_diffusion(self, initial_noise, steps, step_offset=0.0):
        """Takes noise as an input and calls the model steps number of times, then returns the final result

        Args:
            initial_noise (_type_): _description_
            steps (_type_): _description_
        """
        
        num_tensors = initial_noise.shape[0]
        step_size = (1.0 - step_offset) / steps
        
        next_noisy_batch = initial_noise
        
        for step in range(steps):
            noisy_batch = next_noisy_batch
            diffusion_times = jnp.ones((num_tensors, 1, 1), dtype=initial_noise.dtype) - step * step_size - step_offset
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            
            pred_noises, pred_batch = self.denoise(noisy_batch, noise_rates, signal_rates, train=False)
            
            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(next_diffusion_times)
            
            next_noisy_batch = (next_signal_rates * pred_batch + next_noise_rates * pred_noises)
        return pred_batch
    
    def generate(self, rng):
        steps = FLAGS.DDIM_gen_diffusion_steps
        rng, noise_rng = jax.random.split(rng)
        initial_noise = jax.random.normal(noise_rng, (64, 64, 64))
        initial_noise = self.noise_sigma * initial_noise + self.noise_mu
        
        generated_batch = self.reverse_diffusion(initial_noise, steps, step_offset=0.1)
        return generated_batch