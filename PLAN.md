# Plan: Running Disco RL on Craftax

## Overview

This document outlines the plan for integrating Google DeepMind's Disco RL (discovered reinforcement learning algorithms) with the Craftax environment, combining state-of-the-art discovered RL update rules with a challenging open-ended RL benchmark.

### What is Disco RL?

Disco RL is a JAX-based framework from the Nature publication "Discovering State-of-the-art Reinforcement Learning Algorithms." It provides:
- Pre-trained weights for the **Disco103** discovered update rule
- Meta-training capabilities to discover new RL algorithms
- Minimal, modular JAX implementation with Haiku neural networks

### What is Craftax?

Craftax is a lightning-fast, JAX-based benchmark for open-ended RL that:
- Combines Crafter and NetHack mechanics
- Runs **250x faster** than original Crafter (1 billion interactions in <1 hour on single GPU)
- Requires deep exploration, long-term planning, memory, and continual adaptation
- Follows the gymnax interface for seamless JAX integration

## Prerequisites

### System Requirements
- NVIDIA GPU with CUDA 12 support (recommended)
- Python 3.8+
- 16GB+ RAM
- 10GB+ disk space

### Core Dependencies
- JAX with GPU support
- Haiku (neural networks)
- Optax (optimization)
- Gymnax (environment interface)
- NumPy, Matplotlib (utilities)

## Phase 1: Environment Setup

### Step 1.1: Create Python Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# or: venv\Scripts\activate  # On Windows

# Upgrade pip
pip install --upgrade pip
```

### Step 1.2: Install JAX with GPU Support
```bash
# For NVIDIA GPU (CUDA 12)
pip install -U "jax[cuda12]"

# Verify GPU detection
python -c "import jax; print(jax.devices())"
```

### Step 1.3: Install Craftax
```bash
# Option A: Via pip
pip install craftax

# Option B: From source (for development)
git clone https://github.com/MichaelTMatthews/Craftax.git
cd Craftax
pip install -e ".[dev]"
cd ..
```

### Step 1.4: Install Disco RL
```bash
pip install git+https://github.com/google-deepmind/disco_rl.git
```

### Step 1.5: Install Additional Dependencies
```bash
pip install dm-haiku optax chex gymnax
```

## Phase 2: Understanding the Integration Challenge

### Disco RL Architecture
- **Environment Interface**: Uses custom `EnvInterface` with Catch environment
- **Agent**: Generic RL agent compatible with multiple update rules
- **Update Rules**: Modular learned algorithms (Disco103, etc.)
- **Network**: MLP or LSTM-based value functions

### Craftax Interface
- **API**: Follows gymnax standard
- **Observation Space**: High-dimensional (symbolic or pixel-based)
- **Action Space**: Discrete action space
- **Auto-reset**: Configurable for training efficiency

### Key Integration Points
1. **Environment Adapter**: Create a wrapper to adapt Craftax to Disco RL's interface
2. **Network Architecture**: Adapt Disco RL's networks for Craftax's observation space
3. **Update Rule**: Use Disco103 or meta-train new rules
4. **Evaluation**: Implement Craftax-specific metrics

## Phase 3: Implementation Steps

### Step 3.1: Create Environment Adapter
Create `craftax_env_adapter.py`:

```python
"""Adapter to make Craftax compatible with Disco RL interface."""
import jax
import jax.numpy as jnp
from craftax import make_craftax_env_from_name
from disco_rl import env_interface

class CraftaxEnvInterface(env_interface.EnvInterface):
    """Wraps Craftax to match Disco RL's expected interface."""

    def __init__(self, env_name="Craftax-Symbolic-v1", **kwargs):
        self.env = make_craftax_env_from_name(env_name, auto_reset=True)
        self.env_params = self.env.default_params
        self._action_spec = self.env.action_space(self.env_params)
        self._obs_spec = self.env.observation_space(self.env_params)

    def reset(self, rng):
        """Reset environment."""
        obs, state = self.env.reset(rng, self.env_params)
        return self._process_obs(obs), state

    def step(self, rng, state, action):
        """Step environment."""
        obs, state, reward, done, info = self.env.step(
            rng, state, action, self.env_params
        )
        return self._process_obs(obs), state, reward, done, info

    def _process_obs(self, obs):
        """Process observation to match Disco RL expectations."""
        # May need flattening or reshaping depending on Disco RL requirements
        if isinstance(obs, dict):
            # Handle dict observations (flatten or select key)
            return jnp.concatenate([v.flatten() for v in obs.values()])
        return obs

    @property
    def action_spec(self):
        return self._action_spec

    @property
    def observation_spec(self):
        return self._obs_spec
```

### Step 3.2: Adapt Network Architecture
Create `craftax_networks.py`:

```python
"""Neural network architectures for Craftax observations."""
import haiku as hk
import jax.numpy as jnp

class CraftaxValueNetwork(hk.Module):
    """Value network adapted for Craftax's observation space."""

    def __init__(self, hidden_size=256, name=None):
        super().__init__(name=name)
        self.hidden_size = hidden_size

    def __call__(self, obs):
        """Forward pass."""
        # Flatten if necessary
        if len(obs.shape) > 2:
            obs = hk.Flatten()(obs)

        # MLP layers
        x = hk.Linear(self.hidden_size)(obs)
        x = jnp.tanh(x)
        x = hk.Linear(self.hidden_size)(x)
        x = jnp.tanh(x)

        # Value head
        value = hk.Linear(1)(x)
        return value.squeeze(-1)

class CraftaxLSTMNetwork(hk.RNNCore):
    """LSTM-based network for Craftax (for memory/planning)."""

    def __init__(self, hidden_size=256, name=None):
        super().__init__(name=name)
        self.hidden_size = hidden_size
        self.lstm = hk.LSTM(hidden_size)

    def __call__(self, obs, state):
        """Forward pass with recurrent state."""
        if len(obs.shape) > 2:
            obs = hk.Flatten()(obs)

        # Embedding
        x = hk.Linear(self.hidden_size)(obs)
        x = jnp.tanh(x)

        # LSTM
        x, new_state = self.lstm(x, state)

        # Value head
        value = hk.Linear(1)(x)
        return value.squeeze(-1), new_state

    def initial_state(self, batch_size):
        return self.lstm.initial_state(batch_size)
```

### Step 3.3: Create Training Script
Create `train_disco_craftax.py`:

```python
"""Main training script for Disco RL on Craftax."""
import jax
import jax.numpy as jnp
from disco_rl import agent, update_rules
from craftax_env_adapter import CraftaxEnvInterface
from craftax_networks import CraftaxValueNetwork
import pickle

def create_agent_config():
    """Configure agent for Craftax."""
    return {
        'network_factory': lambda: CraftaxValueNetwork(hidden_size=512),
        'optimizer': 'adam',
        'learning_rate': 3e-4,
        'discount': 0.99,
        'entropy_cost': 0.01,
    }

def train(
    num_steps=1_000_000_000,  # 1 billion steps
    eval_interval=10_000_000,  # Eval every 10M steps
    checkpoint_interval=50_000_000,  # Save every 50M steps
    seed=0,
):
    """Train Disco RL agent on Craftax."""
    rng = jax.random.PRNGKey(seed)

    # Create environment
    env = CraftaxEnvInterface("Craftax-Symbolic-v1")

    # Create agent with Disco103 update rule
    rng, agent_rng = jax.random.split(rng)
    agent_state = agent.initialize(
        agent_rng,
        env,
        update_rule=update_rules.disco103,
        **create_agent_config()
    )

    # Training loop
    step = 0
    while step < num_steps:
        # Rollout collection
        rng, rollout_rng = jax.random.split(rng)
        trajectories, agent_state = agent.collect_trajectories(
            rollout_rng, env, agent_state, num_steps=2048
        )

        # Update agent using Disco103 rule
        rng, update_rng = jax.random.split(rng)
        agent_state, metrics = agent.update(
            update_rng, agent_state, trajectories
        )

        step += 2048

        # Logging
        if step % eval_interval == 0:
            eval_metrics = evaluate(rng, env, agent_state)
            print(f"Step {step}: {eval_metrics}")

        # Checkpointing
        if step % checkpoint_interval == 0:
            save_checkpoint(agent_state, step)

    return agent_state

def evaluate(rng, env, agent_state, num_episodes=100):
    """Evaluate agent on Craftax."""
    returns = []
    achievements = []

    for _ in range(num_episodes):
        rng, episode_rng = jax.random.split(rng)
        episode_return, episode_info = run_episode(
            episode_rng, env, agent_state
        )
        returns.append(episode_return)
        achievements.append(episode_info.get('achievements', 0))

    return {
        'mean_return': jnp.mean(jnp.array(returns)),
        'mean_achievements': jnp.mean(jnp.array(achievements)),
    }

def run_episode(rng, env, agent_state, max_steps=10000):
    """Run single episode."""
    rng, reset_rng = jax.random.split(rng)
    obs, env_state = env.reset(reset_rng)

    total_return = 0
    step = 0
    done = False

    while not done and step < max_steps:
        rng, action_rng, step_rng = jax.random.split(rng, 3)

        # Get action from agent
        action = agent.select_action(action_rng, agent_state, obs)

        # Step environment
        obs, env_state, reward, done, info = env.step(
            step_rng, env_state, action
        )

        total_return += reward
        step += 1

    return total_return, info

def save_checkpoint(agent_state, step):
    """Save agent checkpoint."""
    with open(f'checkpoints/agent_step_{step}.pkl', 'wb') as f:
        pickle.dump(agent_state, f)

if __name__ == '__main__':
    trained_agent = train()
```

### Step 3.4: Create Evaluation Script
Create `evaluate_disco_craftax.py`:

```python
"""Evaluation and visualization for trained agents."""
import jax
from train_disco_craftax import evaluate, run_episode
from craftax_env_adapter import CraftaxEnvInterface
import pickle
import matplotlib.pyplot as plt

def load_checkpoint(checkpoint_path):
    """Load agent checkpoint."""
    with open(checkpoint_path, 'rb') as f:
        return pickle.load(f)

def visualize_episode(rng, env, agent_state, save_path='episode.gif'):
    """Render and save episode video."""
    # Implementation depends on Craftax's rendering API
    pass

def evaluate_agent(checkpoint_path, num_episodes=100):
    """Comprehensive evaluation."""
    rng = jax.random.PRNGKey(42)
    env = CraftaxEnvInterface("Craftax-Symbolic-v1")
    agent_state = load_checkpoint(checkpoint_path)

    metrics = evaluate(rng, env, agent_state, num_episodes)

    print("Evaluation Results:")
    print(f"  Mean Return: {metrics['mean_return']:.2f}")
    print(f"  Mean Achievements: {metrics['mean_achievements']:.2f}")

    return metrics

if __name__ == '__main__':
    evaluate_agent('checkpoints/agent_step_1000000000.pkl')
```

## Phase 4: Testing and Validation

### Step 4.1: Unit Tests
Create `tests/test_adapter.py`:
```python
"""Test environment adapter."""
import jax
from craftax_env_adapter import CraftaxEnvInterface

def test_environment_reset():
    """Test environment reset."""
    rng = jax.random.PRNGKey(0)
    env = CraftaxEnvInterface()
    obs, state = env.reset(rng)
    assert obs is not None
    assert state is not None

def test_environment_step():
    """Test environment step."""
    rng = jax.random.PRNGKey(0)
    env = CraftaxEnvInterface()
    obs, state = env.reset(rng)

    rng, action_rng, step_rng = jax.random.split(rng, 3)
    action = env.action_spec.sample(action_rng)
    obs, state, reward, done, info = env.step(step_rng, state, action)

    assert obs is not None
    assert isinstance(reward, (int, float))
    assert isinstance(done, bool)
```

### Step 4.2: Smoke Tests
```bash
# Test environment setup
python -c "from craftax_env_adapter import CraftaxEnvInterface; env = CraftaxEnvInterface(); print('Environment OK')"

# Test network initialization
python -c "from craftax_networks import CraftaxValueNetwork; net = CraftaxValueNetwork(); print('Network OK')"

# Quick training test (1000 steps)
python train_disco_craftax.py --num_steps 1000
```

### Step 4.3: Baseline Comparison
Compare Disco103 performance against:
- PPO baseline (from Craftax Baselines repo)
- DQN baseline
- Random policy

## Phase 5: Optimization and Tuning

### Hyperparameter Tuning
Key hyperparameters to tune:
- Learning rate: [1e-4, 3e-4, 1e-3]
- Network hidden size: [256, 512, 1024]
- Discount factor: [0.99, 0.995, 0.999]
- Entropy coefficient: [0.001, 0.01, 0.1]
- Batch size / rollout length

### Performance Optimization
1. **JIT Compilation**: Ensure all loops are jitted
2. **Vectorization**: Use vmap for parallel episodes
3. **Memory**: Profile and optimize memory usage
4. **Mixed Precision**: Consider using bfloat16

## Phase 6: Meta-Training (Advanced)

### Discover New Update Rules
If Disco103 doesn't perform well, use meta-training:

```python
"""Meta-train new update rule on Craftax."""
from disco_rl import meta_train

# Configure meta-training
meta_config = {
    'num_meta_iterations': 1000,
    'inner_steps': 10000,
    'meta_batch_size': 8,
}

# Meta-train
new_update_rule = meta_train.train(
    env=CraftaxEnvInterface(),
    **meta_config
)
```

## Phase 7: Evaluation Metrics

### Craftax-Specific Metrics
- **Achievement Score**: Number of milestones reached
- **Exploration Coverage**: Percentage of world discovered
- **Survival Time**: Episode length
- **Item Collection**: Types and quantities of items gathered
- **Combat Success**: Monsters defeated

### Learning Metrics
- Sample efficiency curve
- Wall-clock time to target performance
- Convergence stability
- Generalization across seeds

## Potential Challenges and Solutions

### Challenge 1: Observation Space Mismatch
**Problem**: Craftax has complex, high-dimensional observations (symbolic or visual).
**Solution**:
- Use observation preprocessing/flattening
- Consider convolutional networks for pixel observations
- Use attention mechanisms for structured observations

### Challenge 2: Long Episode Horizons
**Problem**: Craftax episodes can be very long (10k+ steps).
**Solution**:
- Use LSTM networks for memory
- Implement value function bootstrapping
- Increase discount factor (γ=0.999)

### Challenge 3: Sparse Rewards
**Problem**: Achievements and progress are infrequent.
**Solution**:
- Add intrinsic motivation (curiosity, exploration bonuses)
- Use reward shaping from Craftax's achievement system
- Implement hindsight experience replay

### Challenge 4: Compilation Time
**Problem**: JAX compilation takes time on first run (~30s for rendering, ~20s for first step).
**Solution**:
- Pre-compile functions during initialization
- Use ahead-of-time (AOT) compilation
- Cache compiled functions between runs

### Challenge 5: Memory Requirements
**Problem**: Large rollout buffers for long episodes.
**Solution**:
- Use gradient checkpointing
- Implement trajectory truncation
- Use replay buffers with sampling

## Expected Timeline

1. **Setup (1-2 days)**: Environment installation, dependency setup
2. **Integration (3-5 days)**: Adapter implementation, initial testing
3. **Training (1-2 weeks)**: First full training run, debugging
4. **Optimization (1-2 weeks)**: Hyperparameter tuning, performance optimization
5. **Evaluation (1 week)**: Comprehensive benchmarking, visualization
6. **Meta-Training (2-4 weeks, optional)**: Discovering Craftax-specific algorithms

## Success Criteria

### Minimum Viable Success
- Agent successfully trains without crashes
- Performance exceeds random policy
- Achieves at least 10% of achievements

### Target Success
- Matches or exceeds PPO baseline from Craftax paper
- Achieves 50%+ of achievements
- Demonstrates emergent exploration behavior

### Stretch Goals
- Outperforms all published Craftax baselines
- Meta-trains Craftax-specific update rule
- Achieves 90%+ of achievements
- Generalizes to Craftax-MA (multi-agent variant)

## References and Resources

### Papers
- **Disco RL**: "Discovering State-of-the-art Reinforcement Learning Algorithms" (Nature, 2025)
  - DOI: 10.1038/s41586-025-09761-x
- **Craftax**: "Craftax: A Lightning-Fast Benchmark for Open-Ended Reinforcement Learning" (ICML 2024 Spotlight)
  - arXiv: 2402.16801

### Repositories
- **Disco RL**: https://github.com/google-deepmind/disco_rl
- **Craftax**: https://github.com/MichaelTMatthews/Craftax
- **Craftax Baselines**: https://github.com/MichaelTMatthews/Craftax_Baselines

### Documentation
- **Craftax Website**: https://craftaxenv.github.io/
- **JAX Documentation**: https://jax.readthedocs.io/
- **Gymnax**: https://github.com/RobertTLange/gymnax

## Next Steps

1. Set up development environment
2. Install all dependencies
3. Run Craftax and Disco RL examples independently
4. Begin adapter implementation
5. Start with simple integration test
6. Iterate and expand

---

**Note**: This is a living document. Update as you make progress and discover new challenges or solutions.
