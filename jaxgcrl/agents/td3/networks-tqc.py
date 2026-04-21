"""TQC networks (based on TD3)."""

from typing import Sequence, Tuple

import jax
import jax.numpy as jnp
from brax.training import networks, types
from brax.training.networks import ActivationFn, FeedForwardNetwork, Initializer
from brax.training.types import PRNGKey
from flax import linen, struct


@struct.dataclass
class TD3Networks:
    policy_network: networks.FeedForwardNetwork
    q_network: networks.FeedForwardNetwork


def make_inference_fn(td3_networks: TD3Networks):
    """Creates params and inference function for the TQC agent."""

    def make_policy(
        params: types.PolicyParams, exploration_noise=0.0, noise_clip=0.0, deterministic=False
    ) -> types.Policy:
        def policy(observations: types.Observation, key_noise: PRNGKey) -> Tuple[types.Action, types.Extra]:
            actions = td3_networks.policy_network.apply(*params, observations)
            noise = (jax.random.normal(key_noise, actions.shape) * exploration_noise).clip(
                -noise_clip, noise_clip
            )
            return actions + noise, {}

        return policy

    return make_policy


class MLP(linen.Module):
    """MLP module."""

    layer_sizes: Sequence[int]
    activation: ActivationFn = linen.relu
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform()
    activate_final: bool = False
    bias: bool = True
    layer_norm: bool = False

    @linen.compact
    def __call__(self, data: jnp.ndarray):
        hidden = data
        for i, hidden_size in enumerate(self.layer_sizes):
            hidden = linen.Dense(
                hidden_size,
                name=f"hidden_{i}",
                kernel_init=self.kernel_init,
                use_bias=self.bias,
            )(hidden)
            if i != len(self.layer_sizes) - 1 or self.activate_final:
                if self.layer_norm:
                    hidden = linen.LayerNorm()(hidden)
                hidden = self.activation(hidden)
        return hidden


class QuantileCritic(linen.Module):
    """Quantile critic: n_critics independent networks, each outputting n_quantiles values.

    Output shape: (batch, n_critics, n_quantiles)
    """

    n_critics: int = 2
    n_quantiles: int = 25
    hidden_layer_sizes: Sequence[int] = (256, 256)
    activation: ActivationFn = linen.relu
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform()

    @linen.compact
    def __call__(self, obs: jnp.ndarray, actions: jnp.ndarray):
        x = jnp.concatenate([obs, actions], axis=-1)
        critics = []
        for i in range(self.n_critics):
            q = MLP(
                layer_sizes=list(self.hidden_layer_sizes) + [self.n_quantiles],
                activation=self.activation,
                kernel_init=self.kernel_init,
            )(x)
            critics.append(q)
        # Stack: (batch, n_critics, n_quantiles)
        return jnp.stack(critics, axis=-2)


def make_policy_network(
    param_size: int,
    obs_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.relu,
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform(),
    layer_norm: bool = False,
) -> FeedForwardNetwork:
    """Creates a policy network."""
    policy_module = MLP(
        layer_sizes=list(hidden_layer_sizes) + [param_size],
        activation=activation,
        kernel_init=kernel_init,
        layer_norm=layer_norm,
    )

    def apply(processor_params, policy_params, obs):
        obs = preprocess_observations_fn(obs, processor_params)
        raw_actions = policy_module.apply(policy_params, obs)
        return linen.tanh(raw_actions)

    dummy_obs = jnp.zeros((1, obs_size))
    return FeedForwardNetwork(init=lambda key: policy_module.init(key, dummy_obs), apply=apply)


def make_quantile_q_network(
    observation_size: int,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.relu,
    n_critics: int = 2,
    n_quantiles: int = 25,
) -> FeedForwardNetwork:
    """Creates a quantile Q-network for TQC.

    Returns a FeedForwardNetwork whose apply outputs shape (batch, n_critics, n_quantiles).
    """
    critic_module = QuantileCritic(
        n_critics=n_critics,
        n_quantiles=n_quantiles,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
    )

    dummy_obs = jnp.zeros((1, observation_size))
    dummy_action = jnp.zeros((1, action_size))

    def init(key):
        return critic_module.init(key, dummy_obs, dummy_action)

    def apply(processor_params, q_params, obs, actions):
        obs = preprocess_observations_fn(obs, processor_params)
        return critic_module.apply(q_params, obs, actions)

    return FeedForwardNetwork(init=init, apply=apply)


def make_td3_networks(
    observation_size: int,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: networks.ActivationFn = linen.relu,
    n_critics: int = 5,
    n_quantiles: int = 25,
) -> TD3Networks:
    """Make TQC networks (drop-in replacement for TD3 networks)."""
    policy_network = make_policy_network(
        action_size,
        observation_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
    )

    q_network = make_quantile_q_network(
        observation_size,
        action_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        n_critics=n_critics,
        n_quantiles=n_quantiles,
    )

    return TD3Networks(policy_network=policy_network, q_network=q_network)