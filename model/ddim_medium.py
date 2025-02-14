from dataclasses import field
from typing import List
import flax.linen as nn
import jax
import jax.numpy as jnp

from model.unet import UNet90s




#flags.DEFINE_integer("DDIM_gen_diffusion_steps", 29, "The amount of times the noise goes through the model during inference time")


class DiffusionModelMedium(nn.Module):
    feature_sizes: List[int] = field(default_factory= lambda: [64, 96, 128])
    block_depths: int = 2
    start_log_snr: float = 2.5
    end_log_snr: float = -7.5
    schedule_type: str = "linear"
    
    noise_mu: float = 0.0
    noise_sigma: float = 1.0
    
    
    def setup(self):
        self.network = UNet90s(feature_sizes=self.feature_sizes, block_depths=self.block_depths)
        
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
        
        return pred_noises, pred_batch
    
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
    
    
    # def inpaint(self, original_signal, mask, rng, steps=29, step_offset=0.0):
    #     """
    #     Inpaint missing (masked) regions in `original_signal`.
        
    #     Args:
    #         original_signal: shape (B, L, C), the partially known signal.
    #         mask: shape (B, L, C), 1.0 where the signal is known, 0.0 where it's missing.
    #         rng: PRNG key.
    #         steps: number of reverse diffusion steps.
    #         step_offset: can be used to start from partway in the schedule.
            
    #     Returns:
    #         inpainted_signal: shape (B, L, C) with the known regions intact 
    #                           and the missing regions inpainted.
    #     """
    #     # 1) Create random noise for the unknown region only
    #     rng, noise_rng = jax.random.split(rng)
    #     noise_init = jax.random.normal(noise_rng, original_signal.shape)
    #     noise_init = noise_init * self.noise_sigma + self.noise_mu
        
    #     # 2) Combine known region from original_signal with noise in the missing region
    #     #mask has shape (B, 1440, 1), original_signal has shape (B, 1440, 16), so we need just repeat the mask 16 times
    #     mask = mask.reshape(mask.shape[0], mask.shape[1], 1)
    #     mask = jnp.repeat(mask, original_signal.shape[-1], axis=-1)
    #     print(mask.shape)
        
    #     x = mask * original_signal + (1.0 - mask) * noise_init
        
    #     #cross fade to noise
    #     x = x * 0.6 + noise_init * 0.4
        
    #     # 3) Reverse diffusion steps
    #     step_size = (1.0 - step_offset) / steps
    #     for step in range(steps):
    #         diffusion_times = jnp.ones((x.shape[0], 1, 1), dtype=x.dtype) \
    #                           - step * step_size - step_offset
    #         noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            
    #         # Predict noise & sample using current estimate 'x'
    #         pred_noises, pred_batch = self.denoise(x, noise_rates, signal_rates, train=False)
            
    #         # Move one step backward in the diffusion chain
    #         next_diffusion_times = diffusion_times - step_size
    #         next_noise_rates, next_signal_rates = self.diffusion_schedule(next_diffusion_times)
    #         x_next = next_signal_rates * pred_batch + next_noise_rates * pred_noises
            
    #         # 4) Overwrite known regions with original signal so they remain unchanged
    #         x = mask * original_signal + (1.0 - mask) * x_next
        
    #     return x
    
    def inpaint(self, original_signal, mask, rng, steps=29, step_offset=0.0):
        """
        Inpaint missing (masked) regions in `original_signal`. Instead of injecting the clean 
        known signal, we compute its noisy version according to the forward diffusion process, 
        ensuring that both known and unknown regions have the same noise characteristics.

        Args:
            original_signal: Array of shape (B, L, C), the partially known signal.
            mask: Array of shape (B, L, C) with 1.0 in known regions and 0.0 in missing regions.
            rng: PRNG key.
            steps: Number of reverse diffusion steps.
            step_offset: An offset into the diffusion schedule.

        Returns:
            inpainted_signal: Array of shape (B, L, C) with the known regions noised consistently 
                            and the missing regions inpainted.
        """
        B = original_signal.shape[0]
        # mask = mask.reshape(mask.shape[0], mask.shape[1], 1)
        # mask = jnp.repeat(mask, original_signal.shape[-1], axis=-1)
        # Use t_init = 1.0 as the starting diffusion time for inpainting.
        t_init = jnp.ones((B, 1, 1), dtype=original_signal.dtype)
        noise_rates_init, signal_rates_init = self.diffusion_schedule(t_init)
        
        # Sample a fixed noise for the known regions. This noise will be used to compute the 
        # forward diffusion version of the known signal at any time step.
        rng, known_rng = jax.random.split(rng)
        noise_known = jax.random.normal(known_rng, original_signal.shape)
        
        # For the unknown region, sample independent noise.
        rng, unknown_rng = jax.random.split(rng)
        noise_unknown = jax.random.normal(unknown_rng, original_signal.shape)
        
        # Compute the initial noised version for the known region using the forward process.
        known_noisy = signal_rates_init * original_signal + noise_rates_init * noise_known
        # For the unknown region, we use pure noise at the same noise level.
        unknown_noisy = noise_rates_init * noise_unknown
        
        # Combine both regions to get the starting x at t_init.
        x = mask * known_noisy + (1.0 - mask) * unknown_noisy
        
        # Reverse diffusion loop.
        step_size = (1.0 - step_offset) / steps
        for step in range(steps):
            diffusion_times = jnp.ones((B, 1, 1), dtype=x.dtype) - step * step_size - step_offset
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            
            # Predict the denoised sample.
            pred_noises, pred_batch = self.denoise(x, noise_rates, signal_rates, train=False)
            
            # Compute the next time step.
            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(next_diffusion_times)
            x_next = next_signal_rates * pred_batch + next_noise_rates * pred_noises
            
            # For the known regions, recompute the forward-diffused value at the new time.
            known_next = next_signal_rates * original_signal + next_noise_rates * noise_known
            
            # Combine: enforce that the known parts follow their forward diffusion.
            x = mask * known_next + (1.0 - mask) * x_next
        x = mask * original_signal + (1.0 - mask) * pred_batch
        # print(signal_rates)
            
        return x

    
    
    
    def generate(self, rng, batch_size):
        steps = 29
        rng, noise_rng = jax.random.split(rng)
        initial_noise = jax.random.normal(noise_rng, (batch_size, 1440, 16))
        #initial_noise = jax.random.uniform(noise_rng, (batch_size, 16*64*5, 8))
        initial_noise = self.noise_sigma * initial_noise + self.noise_mu
        


        generated_batch = self.reverse_diffusion(initial_noise, steps, step_offset=0.0)
        return generated_batch
    
    def generate_from_noise(self, noise, step_offset):
        steps = 29
        generated_batch = self.reverse_diffusion(noise, steps, step_offset=step_offset)
        return generated_batch