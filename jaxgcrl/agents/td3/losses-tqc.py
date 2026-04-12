"""TQC losses (based on TD3).

Truncated Quantile Critics: replaces TD3's min(Q1,Q2) with
distributional quantile critics + sorted truncation.

See: https://arxiv.org/abs/2005.04269
"""

from typing import Any

import jax
import jax.numpy as jnp
from brax.training import types
from brax.training.types import Params, PRNGKey

from . import networks

Transition = types.Transition


def quantile_huber_loss(
    current_quantiles: jnp.ndarray,
    target_quantiles: jnp.ndarray,
    kappa: float = 1.0,
) -> jnp.ndarray:
    """Quantile Huber loss as in QR-DQN / TQC.

    Args:
        current_quantiles: (batch, n_critics, n_quantiles) — predicted quantiles.
        target_quantiles:  (batch, 1, n_target_quantiles) — target quantiles (broadcastable).
        kappa: threshold for Huber loss (default 1.0).

    Returns:
        Scalar loss averaged over batch, critics, and quantile pairs.
    """
    n_quantiles = current_quantiles.shape[-1]

    # Cumulative probabilities for each quantile midpoint: tau_hat_i = (2i+1) / (2N)
    # Shape: (n_quantiles,)
    tau_hat = (2 * jnp.arange(n_quantiles, dtype=jnp.float32) + 1) / (2.0 * n_quantiles)

    # Pairwise TD errors: (batch, n_critics, n_quantiles, 1) - (batch, 1, 1, n_target_quantiles)
    # -> (batch, n_critics, n_quantiles, n_target_quantiles)
    pairwise_delta = target_quantiles[:, :, jnp.newaxis, :] - current_quantiles[:, :, :, jnp.newaxis]

    # Huber loss element-wise
    abs_delta = jnp.abs(pairwise_delta)
    huber = jnp.where(abs_delta <= kappa, 0.5 * pairwise_delta ** 2, kappa * (abs_delta - 0.5 * kappa))

    # Asymmetric weighting: |tau - I(delta < 0)|
    # tau_hat shape broadcast: (1, 1, n_quantiles, 1)
    tau_weight = jnp.abs(tau_hat[jnp.newaxis, jnp.newaxis, :, jnp.newaxis] - (pairwise_delta < 0).astype(jnp.float32))

    quantile_loss = tau_weight * huber / kappa

    # Mean over target quantiles, then sum over predicted quantiles (as in TQC paper),
    # then mean over critics and batch
    loss = quantile_loss.mean(axis=-1).sum(axis=-1).mean()
    return loss


def make_losses(
    td3_network: networks.TD3Networks,
    reward_scaling: float,
    discounting: float,
    smoothing: float,
    noise_clip: float,
    max_action: float = 1.0,
    bc: bool = False,
    alpha: float = 2.5,
    n_critics: int = 2,
    n_quantiles: int = 25,
    top_quantiles_to_drop_per_net: int = 2,
):
    """Creates the TQC losses."""
    policy_network = td3_network.policy_network
    q_network = td3_network.q_network

    n_target_quantiles = n_critics * n_quantiles - top_quantiles_to_drop_per_net * n_critics

    def critic_loss(
        q_params: Params,
        target_q_params: Params,
        target_policy_params: Params,
        normalizer_params: Any,
        transitions: Transition,
        key: PRNGKey,
    ) -> jnp.ndarray:
        """Calculates the TQC critic loss with quantile Huber loss."""

        # Current quantiles: (batch, n_critics, n_quantiles)
        current_quantiles = q_network.apply(
            normalizer_params, q_params, transitions.observation, transitions.action
        )

        # Target action with smoothing noise
        next_actions = policy_network.apply(
            normalizer_params, target_policy_params, transitions.next_observation
        )
        smoothing_noise = (jax.random.normal(key, next_actions.shape) * smoothing).clip(
            -noise_clip, noise_clip
        )
        next_actions = (next_actions + smoothing_noise).clip(-max_action, max_action)

        # Next quantiles from target critic: (batch, n_critics, n_quantiles)
        next_quantiles = q_network.apply(
            normalizer_params,
            target_q_params,
            transitions.next_observation,
            next_actions,
        )

        # --- TQC truncation ---
        batch_size = next_quantiles.shape[0]
        # Flatten critics and quantiles: (batch, n_critics * n_quantiles)
        next_quantiles_flat = next_quantiles.reshape(batch_size, -1)
        # Sort ascending and drop the top (largest) quantiles
        next_quantiles_sorted = jnp.sort(next_quantiles_flat, axis=-1)
        next_quantiles_truncated = next_quantiles_sorted[:, :n_target_quantiles]

        # Bellman target: r + gamma * truncated_quantiles
        # Shape: (batch, n_target_quantiles)
        target_q = jax.lax.stop_gradient(
            transitions.reward[:, jnp.newaxis] * reward_scaling
            + transitions.discount[:, jnp.newaxis] * discounting * next_quantiles_truncated
        )
        # Reshape for broadcasting with current_quantiles: (batch, 1, n_target_quantiles)
        target_q = target_q[:, jnp.newaxis, :]

        q_loss = quantile_huber_loss(current_quantiles, target_q)
        return q_loss

    def actor_loss(
        policy_params: Params,
        q_params: Params,
        normalizer_params: Any,
        transitions: Transition,
    ) -> jnp.ndarray:
        """Calculates the TQC actor loss.

        Uses mean over all quantiles and critics as the Q-value estimate.
        """
        new_actions = policy_network.apply(normalizer_params, policy_params, transitions.observation)
        # (batch, n_critics, n_quantiles)
        q_new_actions = q_network.apply(normalizer_params, q_params, transitions.observation, new_actions)
        # Mean over quantiles, then mean over critics -> (batch,)
        q_values = q_new_actions.mean(axis=-1).mean(axis=-1, keepdims=True)

        lmbda = jax.lax.stop_gradient(bc * alpha / jnp.mean(jnp.abs(q_values)) + (1 - bc))
        q_mean = jnp.mean(q_values)
        return -lmbda * q_mean + bc * mean_squared_error(new_actions, transitions.action)

    def mean_squared_error(predictions, targets):
        return jnp.mean(jnp.square(predictions - targets))

    return critic_loss, actor_loss