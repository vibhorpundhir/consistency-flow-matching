import dataclasses
from functools import partial
import jax
import numpy as np
from jax import numpy as jnp
from jax import random as jr
from jax.scipy.special import erf


# Simplified pseudo huber loss
def pseudo_huber_loss(inputs, targets):
    c = 0.00054 * jnp.sqrt(np.prod(inputs.shape[1:3]))
    return jnp.sqrt(jnp.square(inputs - targets) + c**2) - c


# Simplified sigma computation
def compute_sigmas(num_timesteps, sigma_min=0.002, sigma_max=80.0, rho=7.0):
    rho_inv = 1 / rho
    steps = np.linspace(0, 1, num_timesteps)
    return ((sigma_min ** rho_inv + steps * (sigma_max ** rho_inv - sigma_min ** rho_inv)) ** rho)


# Simplified loss weighting
def loss_weighting(sigmas):
    return 1 / (sigmas[1:] - sigmas[:-1])


# Timestep schedule
def timestep_schedule(rng_key, n_samples, sigmas, mean, std):
    logits = erf((jnp.log(sigmas[1:]) - mean) / (std * jnp.sqrt(2))) - erf(
        (jnp.log(sigmas[:-1]) - mean) / (std * jnp.sqrt(2))
    )
    return jr.categorical(rng_key, logits=logits, shape=(n_samples,))


# Scale the model predictions
def scale_outputs(sigma, sigma_data, sigma_min):
    skip_scale = (sigma_data ** 2) / ((sigma - sigma_min) ** 2 + sigma_data ** 2)
    out_scale = (sigma_data * (sigma - sigma_min)) / (sigma_data ** 2 + sigma ** 2) ** 0.5
    return skip_scale, out_scale


# Forward model step
def forward_step(rng_key, params, model_fn, inputs, sigma, sigma_data, sigma_min):
    preds = model_fn({"params": params}, rngs={"dropout": rng_key}, inputs=inputs, times=sigma)
    skip_scale, out_scale = scale_outputs(sigma, sigma_data, sigma_min)
    return skip_scale * inputs + out_scale * preds


@dataclasses.dataclass
class ConsistencyMatchingConfig:
    n_total_train_steps: int
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    rho: float = 7.0
    sigma_data: float = 0.5
    lognormal_mean: float = -1.1
    lognormal_std: float = 2.0


class ConsistencyMatching:
    def __new__(cls, model_fn, config: ConsistencyMatchingConfig):
        def loss_fn(params, rng_key, inputs, sigmas, context=None, is_training=True):
            B = inputs.shape[0]
            score_model = partial(model_fn, context=context, is_training=is_training)

            timesteps_key, noise_key, apply_key = jr.split(rng_key, 3)
            timesteps = timestep_schedule(timesteps_key, B, sigmas, config.lognormal_mean, config.lognormal_std)

            noise = jr.normal(noise_key, inputs.shape)
            current_sigmas, next_sigmas = sigmas[timesteps], sigmas[timesteps + 1]

            pred_next = forward_step(
                apply_key, params, score_model, inputs + next_sigmas * noise, next_sigmas, config.sigma_data, config.sigma_min
            )

            pred_current = forward_step(
                apply_key, params, model_fn, inputs + current_sigmas * noise, current_sigmas, config.sigma_data, config.sigma_min
            )
            pred_current = jax.lax.stop_gradient(pred_current)

            loss = pseudo_huber_loss(pred_next, pred_current) * loss_weighting(sigmas)[timesteps]
            return loss.mean()

        def sigma_schedule_fn(n_step):
            num_timesteps = jnp.minimum(config.n_total_train_steps // (n_step + 1), config.n_total_train_steps)
            return compute_sigmas(num_timesteps, config.sigma_min, config.sigma_max, config.rho)

        def sample_fn(rng_key, state, sample_shape=(), context=None, **kwargs):
            yt = jr.normal(rng_key, sample_shape) * config.sigma_max
            return state.apply_fn(
                variables={"params": state.ema_params},
                inputs=yt,
                times=jnp.full(sample_shape[0], config.sigma_max),
                context=context,
                is_training=False,
            )

        return loss_fn, sigma_schedule_fn, sample_fn
