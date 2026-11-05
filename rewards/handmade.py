"""Hand-written reward: a one-time bonus for picking up the key and for opening the door."""
from minigrid.core.world_object import Door

def reward(env, state):
    bonus = 0.0
    if env.carrying is not None and not state.get("key"):
        state["key"] = True
        bonus += 0.2
    door_open = any(isinstance(obj, Door) and obj.is_open for obj in env.grid.grid if obj is not None)
    if door_open and not state.get("door"):
        state["door"] = True
        bonus += 0.2
    return bonus