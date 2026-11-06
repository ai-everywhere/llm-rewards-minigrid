def reward(env, state):
    # Locate the key, door, and goal.
    key_pos = None
    door_pos = None
    goal_pos = None
    door_open = False

    for x in range(env.grid.width):
        for y in range(env.grid.height):
            obj = env.grid.get(x, y)
            if obj is None:
                continue

            if obj.type == "key":
                key_pos = (x, y)
            elif obj.type == "door":
                door_pos = (x, y)
                door_open = obj.is_open
            elif obj.type == "goal":
                goal_pos = (x, y)

    agent_pos = env.agent_pos
    max_dist = env.grid.width + env.grid.height

    # Normalized Manhattan-distance progress.
    def proximity(target):
        if target is None:
            return 0.0

        distance = (
            abs(agent_pos[0] - target[0])
            + abs(agent_pos[1] - target[1])
        )
        return 1.0 - min(distance / max_dist, 1.0)

    # Remember key acquisition even if it is no longer on the grid.
    if env.carrying is not None and env.carrying.type == "key":
        state["key_collected"] = True

    # Build a phase-dependent potential.
    if door_open:
        state["door_opened"] = True
        potential = 0.4 + 0.2 * proximity(goal_pos)

    elif state.get("key_collected", False):
        potential = 0.2 + 0.2 * proximity(door_pos)

    else:
        potential = 0.2 * proximity(key_pos)

    # Reward only changes in progress, preventing repeated farming.
    previous = state.get("previous_potential", 0.0)
    bonus = potential - previous

    state["previous_potential"] = potential

    return float(bonus)