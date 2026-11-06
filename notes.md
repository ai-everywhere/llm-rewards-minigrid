# Results

PPO on MiniGrid DoorKey, 5 seeds per reward (5x5: 200k steps, 8x8: 600k steps).
*Steps to 80%*: training steps until the rolling success rate (100 episodes) reaches 80%, mean over seeds.
*Eval*: success rate of the deterministic policy on 100 unseen layouts.

| Reward   | 5x5 steps to 80% | 5x5 eval  | 8x8 steps to 80%  | 8x8 eval  |
|----------|-----------------:|----------:|------------------:|----------:|
| sparse   | 28k              | 99 ± 2 %  | 216k (1/5 seeds)  | 18 ± 35 % |
| handmade | 16.8k            | 99 ± 2 %  | 86.4k             | 91 ± 6 %  |
| llm_1    | 16.8k            | 84 ± 23 % | 99.6k             | 79 ± 11 % |
| llm_2    | 16k              | 79 ± 28 % | 81.6k             | 90 ± 7 %  |
| llm_3    | 14.4k            | 99 ± 1 %  | 70.8k             | 89 ± 3 %  |

**1. LLM rewards vs. sparse.** All three learn faster: 40–49 % fewer steps on 5x5. On 8x8, sparse learns with only 1 seed out of 5, while every LLM reward learns with all 5.

**2. LLM rewards vs. handmade.** Comparable. llm_3 learns faster on both grids with the same final success. llm_2 is about equal. llm_1 is slower on 8x8 and less reliable at evaluation.

**3. Consistency across LLM samples.** During training the three are close; all reach 100 % success. The ranking is the same on both grids (llm_3 > llm_2 > llm_1). The same prompt thus yields rewards of different quality, but with 5 seeds only the llm_3 / llm_1 gap is clear.

**4. Reward hacking.** None observed. Three 5x5 runs have a low eval score (llm_1 seed 4: 38 %; llm_2 seeds 1 and 2: 32 % and 63 %). These same runs solve 100 % of their training episodes in about 11 steps. The failures come from evaluation instead: on unseen layouts, the deterministic policy can loop until the time limit. The reward code also leaves nothing to farm: key and door bonuses are paid once, llm_2 only rewards new distance records, and llm_3 caps the total bonus. One caveat: llm_1 uses `Φ(s') − Φ(s)` instead of `γΦ(s') − Φ(s)`, so it is not strictly policy-invariant, though no effect is visible.
*Limitation:* the shaped bonus is not logged during training, so this conclusion relies on success rates, episode lengths and the reward code.

**5. 8x8 grid.** All shaped rewards still work (5/5 seeds, 79–91 % eval), while sparse fails with 4 seeds out of 5. Shaping matters more as exploration gets harder.
