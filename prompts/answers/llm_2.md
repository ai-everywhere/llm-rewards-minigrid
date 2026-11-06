def reward(env, state):
    bonus = 0.0

    # Find relevant objects in the grid.
    keys = []
    doors = []
    goals = []

    for x in range(env.grid.width):
        for y in range(env.grid.height):
            obj = env.grid.get(x, y)
            if obj is None:
                continue

            if obj.type == "key":
                keys.append((x, y))
            elif obj.type == "door":
                doors.append((x, y, obj.is_open, obj.is_locked))
            elif obj.type == "goal":
                goals.append((x, y))

    ax, ay = env.agent_pos
    carrying_key = (
        env.carrying is not None
        and env.carrying.type == "key"
    )

    door_open = any(d[2] for d in doors)

    # Detect key pickup (one-time bonus).
    if carrying_key and not state.get("key_rewarded", False):
        bonus += 0.12
        state["key_rewarded"] = True

    # Detect door opening (one-time bonus).
    if door_open and not state.get("door_rewarded", False):
        bonus += 0.12
        state["door_rewarded"] = True

    # Select the current objective.
    if not carrying_key and not state.get("key_rewarded", False) and keys:
        phase = "key"
        targets = keys
    elif not door_open and doors:
        phase = "door"
        targets = [(d[0], d[1]) for d in doors if not d[2]]
    else:
        phase = "goal"
        targets = goals

    # Calculate normalized proximity to the current objective.
    if targets:
        distance = min(
            abs(ax - x) + abs(ay - y)
            for x, y in targets
        )

        max_distance = env.grid.width + env.grid.height
        potential = 1.0 - min(distance / max_distance, 1.0)
    else:
        potential = 0.0

    # Reward only new distance records, preventing farming through loops.
    best_key = "best_" + phase
    previous_best = state.get(best_key, potential)

    if potential > previous_best:
        bonus += 0.18 * (potential - previous_best)
        state[best_key] = potential

    # Initialize the record for a new objective phase.
    if best_key not in state:
        state[best_key] = potential

    return float(bonus)