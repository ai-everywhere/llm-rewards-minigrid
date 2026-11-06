"""
Run every reward with every seed. Runs that already have a result are skipped, so script can be stopped and restarted late

Examples:
    python run_all.py
    python run_all.py --env MiniGrid-DoorKey-8x8-v0 --steps 1000000
"""
import argparse
import os
import subprocess
import sys

REWARDS = ["sparse", "handmade", "llm_1", "llm_2", "llm_3"]
SEEDS = [0, 1, 2, 3, 4]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="MiniGrid-DoorKey-5x5-v0")
    parser.add_argument("--steps", type=int, default=200_000)
    args = parser.parse_args()
    
    for reward in REWARDS:
        for seed in SEEDS:
            result_file = os.path.join("results", args.env, reward, f"seed{seed}.json")
            if os.path.exists(result_file):
                print(f"Skip {reward} seed {seed} (already done)")
                continue
            print(f"Run {reward} seed {seed} ...", flush=True)
            subprocess.run([sys.executable, "train.py", "--reward", reward, "--seed", str(seed), 
                            "--env", args.env, "--steps", str(args.steps)], check=True
                           )
            

if __name__ == "__main__":
    main()