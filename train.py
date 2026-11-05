"""
Train PPO (Proximal Policy Optimization) on MiniGrid DoorKey with a chosen reward, then evaluate it

Examples:
    python train.py --reward sparse --seed 0
    python train.py --reward handmade --seed 0
    python train.py --reward llm_1 --seed 0 --env MiniGrid-DoorKey-8x8-v0 --steps 1000000
"""

import argparse
import importlib.util
import json
import os

import gymnasium as gym
import minigrid # noqa: F401 (registers the MiniGrid environments)

from minigrid.wrappers import FlatObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

def load_reward(name):
    """Return the reward function from rewards/<name>.py, or None for the sparse reward."""
    if name == "sparse":
        return None
    path = os.path.join("rewards", f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.reward

class ShapedReward(gym.Wrapper):
    """Adds a bonus to the original reward. The bonus is clipped to [-1, 1] per step."""
    
    def __init__(self, env, reward_fn):
        super().__init__(env)
        self.reward_fn = reward_fn
        self.state = {}
        
    def reset(self, **kwargs):
        self.state = {} # memory for the reward function, emptied at each episode
        return self.env.reset(**kwargs)
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.reward_fn is not None:
            bonus = float(self.reward_fn(self.env.unwrapped, self.state))
            reward += max(-1.0, min(1.0, bonus))
        return obs, reward, terminated, truncated, info
    
def evaluate(model, env_id, n_episodes=100, first_seed=10_000):
    """Success rate and mean episode length with the ORIGINAL reward, on unseen seeds."""
    env = FlatObsWrapper(gym.make(env_id))
    successes, lengths = 0, 0
    for i in range(n_episodes):
        obs, _ = env.reset(seed=first_seed + i)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(int(action))
            done = terminated or truncated
        successes += int(reward > 0)
        lengths += env.unwrapped.step_count
    return successes / n_episodes, lengths / n_episodes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward", default="sparse", help="sparse, handmade, llm_1, ...")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env", default="MiniGrid-DoorKey-5x5-v0")
    parser.add_argument("--steps", type=int, default=300_000)
    args = parser.parse_args()
    
    run_name = f"{args.env}/{args.reward}/seed{args.seed}"
    reward_fn = load_reward(args.reward)
    
    # The Monitor (inside make_vec_env) sits UNDER our wrapper,
    # so the logs in logs/ always record the ORIGINAL reward.
    env = make_vec_env(
        args.env,
        n_envs=8,
        seed=args.seed,
        wrapper_class=lambda e: FlatObsWrapper(ShapedReward(e, reward_fn)),
        monitor_dir=os.path.join("logs", run_name),
    )
    model = PPO(
        "MlpPolicy", env,
        n_steps=128, batch_size=64, n_epochs=10,
        learning_rate=2.5e-4, gamma=0.99, gae_lambda=0.95,
        clip_range=0.2, ent_coef=0.01,
        seed=args.seed, verbose=0,
    )
    model.learn(total_timesteps=args.steps)
    success, mean_length = evaluate(model, args.env)
    result = {
        "env": args.env, "reward": args.reward, "seed": args.seed,
        "steps": args.steps, "eval_success_rate": success, "eval_mean_length": mean_length,
    }
    os.makedirs(os.path.join("results", args.env, args.reward), exist_ok=True)
    with open(os.path.join("results", run_name + ".json"), "w") as f:
        json.dump(result, f, indent=2)
        print(result)
        
if __name__ == "__main__":
    main()