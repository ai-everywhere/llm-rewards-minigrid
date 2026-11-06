def reward(env, state):
    bonus = 0.0

    # Initialize episode progress.
    if not state:
        state["has_key"] = False
        state["door_opened"] = False
        state["visited"] = set()
        state["previous_distance"] = None
        state["goal_rewarded"] = False

    x, y = env.agent_pos

    # Reward collecting the key only once.
    if env.carrying is not None and env.carrying.type == "key":
        if not state["has_key"]:
            bonus += 0.20
            state["has_key"] = True

    # Reward opening the locked door only once.
    for gx in range(env.grid.width):
        for gy in range(env.grid.height):
            obj = env.grid.get(gx, gy)
            if obj is not None and obj.type == "door":
                if obj.is_open and not state["door_opened"]:
                    bonus += 0.25
                    state["door_opened"] = True

    # Small exploration reward for visiting new positions.
    position = (x, y)
    if position not in state["visited"]:
        state["visited"].add(position)
        bonus += 0.005

    # Reward reaching the goal only once.
    if not state["goal_rewarded"]:
        obj = env.grid.get(x, y)
        if obj is not None and obj.type == "goal":
            bonus += 0.30
            state["goal_rewarded"] = True

    # Keep the total auxiliary reward below 1.0.
    state["total_bonus"] = state.get("total_bonus", 0.0)
    remaining = max(0.0, 0.99 - state["total_bonus"])
    bonus = min(bonus, remaining)
    state["total_bonus"] += bonus

    return float(bonus)