import numpy as np
import inspect

def push(env, grid_lists):
    """
    Pushes the first grid from each grid list onto the environment's memory stack,
    only if all grid lists have at least one grid.
    """
    if all(len(grid_list) > 0 for grid_list in grid_lists):
        grids_to_push = [grid_list[0] for grid_list in grid_lists]
        env.memory = [grids_to_push] + env.memory[:-1]
        new_state = env._compute_memory_state_for_grids(grids_to_push)
        env.memory_state = np.roll(env.memory_state, 1, axis=0)
        env.memory_state[0] = new_state
    return grid_lists

def pop(env, grid_lists):
    """
    Removes the most recently added grids from the environment's memory stack,
    only if there are grids in memory.
    """
    if env.memory[0] is not None:
        env.memory = env.memory[1:] + [None]
        env.memory_state = np.roll(env.memory_state, -1, axis=0)
        env.memory_state[-1] = np.zeros(env.encoding_dim, dtype=np.float32)
    return grid_lists

def paste(env, grid_lists):
    """
    Inserts the memorized grids at the beginning of corresponding grid lists,
    only if there are memorized grids and all grid lists are non-empty.
    """
    if env.memory[0] is not None and all(len(grid_list) > 0 for grid_list in grid_lists):
        result = []
        for i, grid_list in enumerate(grid_lists):
            if i < len(env.memory[0]) and env.memory[0][i] is not None:
                result.append([env.memory[0][i]] + grid_list)
            else:
                result.append(grid_list)
        return result
    return grid_lists

# def push(env, grid_lists):
#     """
#     Pushes the first grid from each grid list onto the environment's memory stack.

#     This function:
#     1. Collects the first grid from each grid list (or None if the list is empty).
#     2. If any grids are collected, it:
#        a. Adds these grids to the front of the environment's memory stack.
#        b. Removes the oldest memory if the stack is full.
#        c. Computes a new memory state for the pushed grids.
#        d. Updates the environment's memory state, shifting existing states and adding the new one.

#     Args:
#     env: The environment object containing the memory stack and state.
#     grid_lists (list of lists): The current grid lists.

#     Returns:
#     list of lists: The original grid lists, unchanged.

#     Side effects:
#     - Modifies env.memory by adding new grids and potentially removing old ones.
#     - Updates env.memory_state to reflect the new memory contents.
#     """
#     grids_to_push = [grid_list[0] if grid_list else None for grid_list in grid_lists]
#     if any(grid is not None for grid in grids_to_push):
#         env.memory = [grids_to_push] + env.memory[:-1]
#         new_state = env._compute_memory_state_for_grids(grids_to_push)
#         env.memory_state = np.roll(env.memory_state, 1, axis=0)
#         env.memory_state[0] = new_state
#     return grid_lists

# def pop(env, grid_lists):
#     """
#     Removes the most recently added grids from the environment's memory stack.

#     This function:
#     1. Checks if there are any grids in the memory stack.
#     2. If so, it:
#        a. Removes the most recent set of grids from the memory stack.
#        b. Adds a None placeholder to the end of the memory stack.
#        c. Updates the memory state by shifting all states and adding a zero state at the end.

#     Args:
#     env: The environment object containing the memory stack and state.
#     grid_lists (list of lists): The current grid lists.

#     Returns:
#     list of lists: The original grid lists, unchanged.

#     Side effects:
#     - Modifies env.memory by removing the most recent grids and adding a None placeholder.
#     - Updates env.memory_state to reflect the removed memory.
#     """
#     if env.memory[0] is not None:
#         env.memory = env.memory[1:] + [None]
#         env.memory_state = np.roll(env.memory_state, -1, axis=0)
#         env.memory_state[-1] = np.zeros(env.encoding_dim, dtype=np.float32)
#     return grid_lists

# def paste(env, grid_lists):
#     """
#     Inserts the memorized grids at the beginning of corresponding grid lists,
#     without combining them with existing grids. This increases the number of
#     grids in affected lists by one.

#     Args:
#     env: The environment object containing the memory.
#     grid_lists (list of lists): The current grid lists.

#     Returns:
#     list of lists: Updated grid lists with memorized grids inserted at the beginning of corresponding sublists.
#     """
#     if env.memory[0] is not None:
#         result = []
#         for i, grid_list in enumerate(grid_lists):
#             if i < len(env.memory[0]) and env.memory[0][i] is not None:
#                 # Insert the memorized grid at the beginning without combining
#                 result.append([env.memory[0][i]] + grid_list)
#             else:
#                 result.append(grid_list)  # Keep original if no corresponding memory grid
#         return result
#     return grid_lists
# def paste(env, grid_lists):
#     if env.memory[0] is not None:
#         result = []
#         for i, grid_list in enumerate(grid_lists):
#             if i < len(env.memory[0]) and env.memory[0][i] is not None:
#                 if grid_list:
#                     combined_grid = env.add_grids([env.memory[0][i], grid_list[0]])
#                     result.append([combined_grid] + grid_list)
#                 else:
#                     result.append([env.memory[0][i]])
#             else:
#                 result.append(grid_list)  # Keep original if no corresponding memory grid
#         return result
#     return grid_lists
