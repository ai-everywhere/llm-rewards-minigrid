You are designing a dense reward function for a reinforcement learning agent.

Environment: MiniGrid DoorKey. The agent sees a 7x7 view in front of it and must pick up a key, use it to open a locked door, then reach the green goal square.
The original reward is sparse: 1 - 0.9 * (step_count / max_steps) when the agent reaches the goal, and 0 at every other step. The agent is trained with PPO.

Write a Python function with this exact signature:

    def reward(env, state):
        ...
        return bonus  # a float

It is called after every step. The returned bonus is ADDED to the original reward.

You can read only these attributes:
- env.agent_pos: (x, y) position of the agent
- env.agent_dir: direction of the agent (0 = right, 1 = down, 2 = left, 3 = up)
- env.carrying: None, or the object the agent holds (a key has obj.type == "key")
- env.grid.width, env.grid.height
- env.grid.get(x, y): the object at (x, y), or None for an empty cell
- every object has obj.type, one of "wall", "key", "door", "goal"
- a door also has obj.is_open and obj.is_locked (booleans)
- env.step_count and env.max_steps

`state` is an empty dict at the start of each episode. Use it to remember progress inside the episode, for example to give a bonus only once.

Rules:
- Do not import anything.
- Keep the total bonus for one episode below 1.0, so the original reward stays important.
- Avoid bonuses that the agent can collect again and again without making progress.

Return only the Python code of the function, with short comments.