""" 
Make the learning-curve figure and the results table for one environment.

Examples:
    python analyze.py
    python analyze.py --env MiniGrid-DoorKey-8x8-v0 --steps 600000
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REWARDS = ["sparse", "handmade", "llm_1", "llm_2", "llm_3"]


def success_curve(run_dir, grid):
    """Rolling success rate (original reward > 0) against total environment steps."""
    files = glob.glob(os.path.join(run_dir, "*.monitor.csv"))
    episodes = pd.concat(pd.read_csv(f, skiprows=1) for f in files).sort_values("t")
    steps = episodes["l"].cumsum().to_numpy()
    # Require a full window: with fewer episodes, one lucky success early on would count as 100%
    success = (episodes["r"] > 0).astype(float).rolling(100).mean().to_numpy()
    valid = ~np.isnan(success)
    # NaN before the first full window, so the curve starts where the data starts
    return np.interp(grid, steps[valid], success[valid], left=np.nan)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="MiniGrid-DoorKey-5x5-v0")
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--threshold", type=float, default=0.8)
    args = parser.parse_args()
    
    grid = np.linspace(0, args.steps, 101)
    os.makedirs("figures", exist_ok=True)
    rows = []
    plt.figure(figsize=(7, 4.5))
    
    for reward in REWARDS:
        run_dirs = sorted(glob.glob(os.path.join("logs", args.env, reward, "seed*")))
        if not run_dirs:
            continue
        curves = np.array([success_curve(d, grid) for d in run_dirs])
        mean, std = curves.mean(axis=0), curves.std(axis=0)
        plt.plot(grid, mean, label=reward)
        plt.fill_between(grid, np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1), alpha=0.2)
        
        # Steps needed to reach the threshold, for each seed
        reach = []
        for curve in curves:
            above = np.where(curve >= args.threshold)[0]
            reach.append(grid[above[0]] if len(above) else np.nan)
        reach = np.array(reach)
        
        evals = []
        for path in glob.glob(os.path.join("results", args.env, reward, "seed*.json")):
            with open(path) as f:
                evals.append(json.load(f)["eval_success_rate"])
        evals = np.array(evals)
        
        rows.append({
            "reward": reward,
            "seeds":len(run_dirs),
            "final eval success (mean ± std)": f"{evals.mean():.0%} ± {evals.std():.0%}" if len(evals) else "n/a",
            f"steps to {args.threshold:.0%} success (mean)": (f"{np.nanmean(reach):.0f}" if np.any(~np.isnan(reach)) else "not reached"),
            f"seeds that reached {args.threshold:.0%}": f"{int(np.sum(~np.isnan(reach)))}/{len(reach)}",
        })
        
    plt.xlabel("Environment steps")
    plt.ylabel("Success rate (rolling mean of 100 episodes)")
    plt.title(f"{args.env}: mean ± std over seeds")
    plt.ylim(0, 1.02)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    figure_path = os.path.join("figures", f"{args.env}_success.png")
    plt.savefig(figure_path, dpi=150)
    
    table = pd.DataFrame(rows).to_markdown(index=False)
    table_path = os.path.join("results", f"{args.env}_summary.md")
    with open(table_path, "w") as f:
        f.write(table +"\n")
    print(table)
    print(f"\nSaved {figure_path} and {table_path}")


if __name__ == "__main__":
    main()
    