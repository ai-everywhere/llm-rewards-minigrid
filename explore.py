"""Explore the MiniGrid DoorKey environment and measure a random agent."""
import gymnasium as gym
import minigrid # noqa: F401 (registers the MiniGrid environments)
from minigrid.wrappers import FlatObsWrapper

ENV_ID = "MiniGrid-DoorKey-5x5-v0"

env = gym.make(ENV_ID)
obs, info = env.reset(seed=0)
base = env.unwrapped

print("Mission:", obs["mission"])
print("Grid:")
print(base)
print("Image observation shape:", obs["image"].shape)
print("Actions:", [a.name for a in base.actions])
print("Max steps per episode:", base.max_steps)
print("Flat obserbation size:", FlatObsWrapper(env).observation_space.shape)

# A random agent, to see how hard the sparse reward is
n_episodes = 200
successes = 0
for i in range(n_episodes):
    obs, _ = env.reset(seed=i)
    done = False
    while not done:
        obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
        done = terminated or truncated
    successes += int(reward > 0)
print(f"Random agent success rate: {successes / n_episodes:.0%}")