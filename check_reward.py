"""
Check a reward function with a random agent before training.

Example:
    python check_reward.py llm_1
"""
import argparse

import gymnasium as gym
import minigrid #noqa: F401 (registers the MiniGrid environments)

from train import load_reward

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reward", help="handmade, llm_1, ...")
    parser.add_argument("--env", default="MiniGrid-DoorKey-5x5-v0")
    parser.add_argument("--episodes", type=int, default=50)
    args = parser.parse_args()
    
    reward_fn = load_reward(args.reward)
    env = gym.make(args.env)
    totals, largest_step = [], 0.0
    for i in range(args.episodes):
        env.reset(seed=i)
        state, total, done = {}, 0.0, False
        while not done:
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            bonus = float(reward_fn(env.unwrapped, state))
            total += bonus
            largest_step = max(largest_step, abs(bonus))
            done = terminated or truncated
        totals.append(total)
        
    print(f"Mean bonus per episode (random agent): {sum(totals) / len(totals):.3f}")
    print(f"Largest bonus in one episode: {max(totals):.3f}")
    print(f"Largest bonus in one step (absolute): {largest_step:.3f}")
    if max(totals) > 1.0:
        print("WARNING: a random agent can collect more than 1.0 of bonus in one episode.")
        print("The agent may learn to farm the bonus instead of solving the task (reward hacking).")
    
    
if __name__ == "__main__":
    main()
    