import torch
import inspect


def forget(grid_lists):
    """
    Removes the first grid from each list in the grid_lists.
    If a list becomes empty after removal, it is kept as an empty list.
    """

    def process_grid_list(grid_list):
        if len(grid_list) > 1:
            return grid_list[1:]  # Return all grids except the first one
        else:
            return []  # Return an empty list if the input list has 0 or 1 element

    return [process_grid_list(grid_list) for grid_list in grid_lists]


def keep_last(grid_lists):
    """
    Keeps only the last grid in each sublist of grid_lists.
    If a sublist is empty, it remains unchanged.
    """
    return [[sublist[-1]] if sublist else [] for sublist in grid_lists]


def reverse_order(grid_lists):
    """
    Reverses the order of grids in each sublist of grid_lists.
    """
    return [sublist[::-1] for sublist in grid_lists]


def keep_largest(grid_lists):
    """
    Keeps only the largest grid (by total number of elements) in each sublist.
    If multiple grids have the same largest size, it keeps the first one encountered.
    """

    def largest_grid(sublist):
        if not sublist:
            return []
        largest = max(sublist, key=lambda x: x.numel())
        return [largest]

    return [largest_grid(sublist) for sublist in grid_lists]


def merge_grids(grid_lists):
    """
    Merges all grids in each sublist into a single grid.
    The merge is done by element-wise addition, modulo 10.
    If grids in a sublist have different shapes, they are padded with zeros to match the largest grid.
    """

    def merge_sublist(sublist):
        if not sublist:
            return []
        max_h = max(grid.shape[0] for grid in sublist)
        max_w = max(grid.shape[1] for grid in sublist)
        padded = [
            torch.nn.functional.pad(
                grid, (0, max_w - grid.shape[1], 0, max_h - grid.shape[0])
            )
            for grid in sublist
        ]
        merged = sum(padded) % 10
        return [merged]

    return [merge_sublist(sublist) for sublist in grid_lists]


def reset_to_original(env, grid_lists):
    """
    Resets the current grid lists to the initial grids stored in env.initial_grids.
    This function ignores the input grid_lists and returns a copy of the initial grids.

    Args:
    grid_lists: The current grid lists (ignored in this function)
    env: The environment object containing the initial_grids attribute

    """
    return [[grid.clone()] for grid in env.initial_grids]


def forget_zero_grids(grid_lists):
    """
    Removes any grids that are completely filled with zeros from each grid list.

    Args:
    grid_lists (list of lists of torch.Tensor or np.ndarray): The input grid lists.

    Returns:
    list of lists of torch.Tensor or np.ndarray: Grid lists with zero grids removed.
    """

    def is_non_zero_grid(grid):
        if isinstance(grid, torch.Tensor):
            return torch.any(grid != 0).item()
        elif isinstance(grid, np.ndarray):
            return np.any(grid != 0)
        else:
            return any(any(cell != 0 for cell in row) for row in grid)

    return [
        [grid for grid in grid_list if is_non_zero_grid(grid)]
        for grid_list in grid_lists
    ]


def concatenate(grid_lists):
    """
    Concatenates grids within each grid list horizontally (left to right).
    Returns a new list where each sublist contains a single concatenated grid.
    If shapes don't match in a sublist, returns that sublist unchanged.
    """

    def process_grid_list(grid_list):
        if not grid_list:  # Handle empty lists
            return grid_list

        # Check if all grids have the same height
        first_height = grid_list[0].size(0)
        if not all(grid.size(0) == first_height for grid in grid_list):
            return grid_list  # Heights don't match, return unchanged

        # Concatenate all grids horizontally
        try:
            concatenated = torch.cat(
                grid_list, dim=1
            )  # dim=1 for horizontal concatenation
            return [concatenated]  # Return as a list with single grid
        except:
            return grid_list  # If any error occurs, return unchanged

    return [process_grid_list(grid_list) for grid_list in grid_lists]
