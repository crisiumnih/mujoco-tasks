import brax
import functools
import jax
from brax import envs
from brax.io import model
from brax.training.agents.ppo import train as ppo

# Initialize the environment
env_name = 'ant'  
backend = 'generalized'  # Can be 'positional' or 'spring' depending on requirements
env = envs.get_environment(env_name=env_name, backend=backend)

# Define PPO training function
train_fn = functools.partial(
    ppo.train,
    num_timesteps=50_000_000,  # Total training steps
    num_evals=10,
    reward_scaling=10,
    episode_length=1000,
    normalize_observations=True,
    action_repeat=1,
    unroll_length=5,
    num_minibatches=32,
    num_updates_per_batch=4,
    discounting=0.97,
    learning_rate=3e-4,
    entropy_cost=1e-2,
    num_envs=4096,  # Number of parallel environments
    batch_size=2048,  # Training batch size
    seed=1
)

make_inference_fn, params, _ = train_fn(environment=env)

# Save trained model parameters
model.save_params('ant_ppo_policy', params)
