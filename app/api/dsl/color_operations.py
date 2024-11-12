import torch
import inspect
import numpy as np

def _select_digit(grid_lists, digit):
    """
    Selects grids containing the specified digit.
    If the digit is not present in any grid, returns the original grids.
    Works with both PyTorch tensors and numpy arrays.
    """
    def _contains_digit(grid):
        if isinstance(grid, torch.Tensor):
            return (grid == digit).any().item()
        elif isinstance(grid, np.ndarray):
            return (grid == digit).any()
        else:
            return digit in grid  # Fallback for other types
    
    result = [[grid for grid in grid_list if _contains_digit(grid)] for grid_list in grid_lists]
    
    # Check if any grids were selected
    if any(len(grid_list) > 0 for grid_list in result):
        return result
    else:
        # If no grids were selected, return the original grid_lists
        return grid_lists

def select_digit_0(grid_lists):
    return _select_digit(grid_lists, 0)

def select_digit_1(grid_lists):
    return _select_digit(grid_lists, 1)

def select_digit_2(grid_lists):
    return _select_digit(grid_lists, 2)

def select_digit_3(grid_lists):
    return _select_digit(grid_lists, 3)

def select_digit_4(grid_lists):
    return _select_digit(grid_lists, 4)

def select_digit_5(grid_lists):
    return _select_digit(grid_lists, 5)

def select_digit_6(grid_lists):
    return _select_digit(grid_lists, 6)

def select_digit_7(grid_lists):
    return _select_digit(grid_lists, 7)

def select_digit_8(grid_lists):
    return _select_digit(grid_lists, 8)

def select_digit_9(grid_lists):
    return _select_digit(grid_lists, 9)

def _fill_digit(grid_lists, digit):
    """
    Fills all non-empty grids with the specified digit.
    Works with both PyTorch tensors and numpy arrays.
    """
    def fill_grid(grid):
        if isinstance(grid, torch.Tensor):
            return torch.full_like(grid, digit) if grid.numel() > 0 else grid
        elif isinstance(grid, np.ndarray):
            return np.full_like(grid, digit) if grid.size > 0 else grid
        else:
            return [[digit for _ in row] for row in grid] if grid else grid  # Fallback for other types
    
    return [[fill_grid(grid) for grid in grid_list] for grid_list in grid_lists]

def fill_0(grid_lists):
    return _fill_digit(grid_lists, 0)

def fill_1(grid_lists):
    return _fill_digit(grid_lists, 1)

def fill_2(grid_lists):
    return _fill_digit(grid_lists, 2)

def fill_3(grid_lists):
    return _fill_digit(grid_lists, 3)

def fill_4(grid_lists):
    return _fill_digit(grid_lists, 4)

def fill_5(grid_lists):
    return _fill_digit(grid_lists, 5)

def fill_6(grid_lists):
    return _fill_digit(grid_lists, 6)

def fill_7(grid_lists):
    return _fill_digit(grid_lists, 7)

def fill_8(grid_lists):
    return _fill_digit(grid_lists, 8)

def fill_9(grid_lists):
    return _fill_digit(grid_lists, 9)

def separate_by_color(grid_lists):
    """
    Separates each grid into multiple grids, one for each unique color/digit.
    """
    def separate_grid(grid):
        unique_colors = torch.unique(grid)
        return [torch.where(grid == color, grid, torch.zeros_like(grid)) for color in unique_colors]
    
    return [sum([separate_grid(grid) for grid in grid_list], []) for grid_list in grid_lists]

def sort_grids_ascending(grid_lists):
    """
    Sorts the grids in each sublist based on the smallest digit present in each grid.
    Grids with smaller digits (e.g., 0, 1) come first.
    """
    def get_min_digit(grid):
        return torch.min(grid).item()
    
    return [sorted(grid_list, key=get_min_digit) for grid_list in grid_lists]

def sort_grids_descending(grid_lists):
    """
    Sorts the grids in each sublist based on the largest digit present in each grid.
    Grids with larger digits (e.g., 8, 9) come first.
    """
    def get_max_digit(grid):
        return torch.max(grid).item()
    
    return [sorted(grid_list, key=get_max_digit, reverse=True) for grid_list in grid_lists]

def select_largest_digit(grid_lists):
    """
    Selects grids containing the largest digit present across all grids in the list.
    """
    def find_largest_digit(grid_list):
        return max(torch.max(grid).item() for grid in grid_list)
    
    def contains_digit(grid, digit):
        return (grid == digit).any()
    
    return [
        [grid for grid in grid_list if contains_digit(grid, find_largest_digit(grid_list))]
        for grid_list in grid_lists
    ]

def select_smallest_digit(grid_lists):
    """
    Selects grids containing the smallest digit present across all grids in the list.
    """
    def find_smallest_digit(grid_list):
        return min(torch.min(grid).item() for grid in grid_list)
    
    def contains_digit(grid, digit):
        return (grid == digit).any()
    
    return [
        [grid for grid in grid_list if contains_digit(grid, find_smallest_digit(grid_list))]
        for grid_list in grid_lists
    ]

def add_1(grid_lists):
    """
    Adds 1 to each digit in all grids, wrapping from 9 to 0.
    """
    def add_one_to_grid(grid):
        return (grid + 1) % 10
    
    return [[add_one_to_grid(grid) for grid in grid_list] for grid_list in grid_lists]

def subtract_1(grid_lists):
    """
    Subtracts 1 from each digit in all grids, wrapping from 0 to 9.
    """
    def subtract_one_from_grid(grid):
        return (grid - 1) % 10
    
    return [[subtract_one_from_grid(grid) for grid in grid_list] for grid_list in grid_lists]

def color_inverse(grid_lists):
    """
    Inverts the colors in each grid by converting all non-zero digits to 0 and all zeros to 1.

    Args:
    grid_lists (list of lists of torch.Tensor): The input grid lists.

    Returns:
    list of lists of torch.Tensor: The grid lists with inverted colors.
    """
    def invert_grid(grid):
        return torch.where(grid == 0, torch.ones_like(grid), torch.zeros_like(grid))
    
    return [[invert_grid(grid) for grid in grid_list] for grid_list in grid_lists]

def fill_dominant_color(grid_lists):
    """
    Fills each grid with its most dominant non-zero color (digit).
    Works with both PyTorch tensors and numpy arrays.
    """
    def process_grid(grid):
        if isinstance(grid, torch.Tensor):
            if grid.numel() == 0:
                return grid
            unique, counts = torch.unique(grid, return_counts=True)
            # Filter out zero and find the dominant non-zero color
            non_zero_mask = unique != 0
            if non_zero_mask.sum() == 0:  # If all elements are zero
                return grid
            non_zero_unique = unique[non_zero_mask]
            non_zero_counts = counts[non_zero_mask]
            dominant_color = non_zero_unique[torch.argmax(non_zero_counts)].item()
            return torch.full_like(grid, dominant_color)
        elif isinstance(grid, np.ndarray):
            if grid.size == 0:
                return grid
            unique, counts = np.unique(grid, return_counts=True)
            # Filter out zero and find the dominant non-zero color
            non_zero_mask = unique != 0
            if non_zero_mask.sum() == 0:  # If all elements are zero
                return grid
            non_zero_unique = unique[non_zero_mask]
            non_zero_counts = counts[non_zero_mask]
            dominant_color = non_zero_unique[np.argmax(non_zero_counts)]
            return np.full_like(grid, dominant_color)
        else:
            # Fallback for other types
            if not grid:
                return grid
            flat_grid = [item for sublist in grid for item in sublist if item != 0]
            if not flat_grid:  # If all elements are zero
                return grid
            dominant_color = max(set(flat_grid), key=flat_grid.count)
            return [[dominant_color for _ in row] for row in grid]

    return [[process_grid(grid) for grid in grid_list] for grid_list in grid_lists]

def paint_dominant_color(grid_lists):
    """
    Paints all non-zero digits in each grid with the most dominant non-zero color (digit).
    Zeros (background) are left unchanged.
    Works with both PyTorch tensors and numpy arrays.
    """
    def process_grid(grid):
        if isinstance(grid, torch.Tensor):
            if grid.numel() == 0:
                return grid
            unique, counts = torch.unique(grid, return_counts=True)
            # Filter out zero and find the dominant non-zero color
            non_zero_mask = unique != 0
            if non_zero_mask.sum() == 0:  # If all elements are zero
                return grid
            non_zero_unique = unique[non_zero_mask]
            non_zero_counts = counts[non_zero_mask]
            dominant_color = non_zero_unique[torch.argmax(non_zero_counts)].item()
            return torch.where(grid != 0, torch.full_like(grid, dominant_color), grid)
        elif isinstance(grid, np.ndarray):
            if grid.size == 0:
                return grid
            unique, counts = np.unique(grid, return_counts=True)
            # Filter out zero and find the dominant non-zero color
            non_zero_mask = unique != 0
            if non_zero_mask.sum() == 0:  # If all elements are zero
                return grid
            non_zero_unique = unique[non_zero_mask]
            non_zero_counts = counts[non_zero_mask]
            dominant_color = non_zero_unique[np.argmax(non_zero_counts)]
            return np.where(grid != 0, np.full_like(grid, dominant_color), grid)
        else:
            # Fallback for other types
            if not grid:
                return grid
            flat_grid = [item for sublist in grid for item in sublist if item != 0]
            if not flat_grid:  # If all elements are zero
                return grid
            dominant_color = max(set(flat_grid), key=flat_grid.count)
            return [[dominant_color if cell != 0 else 0 for cell in row] for row in grid]

    return [[process_grid(grid) for grid in grid_list] for grid_list in grid_lists]

  
def _remove_color(grid_lists, color):
    """
    Helper function to remove a specific color (digit) from the grid.

    Args:
    grid_lists (list of lists of torch.Tensor): The input grid lists.
    color (int): The color (digit) to remove.

    Returns:
    list of lists of torch.Tensor: Grids with the specified color removed.
    """
    def process_grid(grid):
        if isinstance(grid, torch.Tensor):
            return torch.where(grid == color, torch.zeros_like(grid), grid)
        elif isinstance(grid, np.ndarray):
            return np.where(grid == color, np.zeros_like(grid), grid)
        else:
            return [[0 if cell == color else cell for cell in row] for row in grid]

    return [[process_grid(grid) for grid in grid_list] for grid_list in grid_lists]

def remove_color_1(grid_lists):
    """Removes color 1 from the grids."""
    return _remove_color(grid_lists, 1)

def remove_color_2(grid_lists):
    """Removes color 2 from the grids."""
    return _remove_color(grid_lists, 2)

def remove_color_3(grid_lists):
    """Removes color 3 from the grids."""
    return _remove_color(grid_lists, 3)

def remove_color_4(grid_lists):
    """Removes color 4 from the grids."""
    return _remove_color(grid_lists, 4)

def remove_color_5(grid_lists):
    """Removes color 5 from the grids."""
    return _remove_color(grid_lists, 5)

def remove_color_6(grid_lists):
    """Removes color 6 from the grids."""
    return _remove_color(grid_lists, 6)

def remove_color_7(grid_lists):
    """Removes color 7 from the grids."""
    return _remove_color(grid_lists, 7)

def remove_color_8(grid_lists):
    """Removes color 8 from the grids."""
    return _remove_color(grid_lists, 8)

def remove_color_9(grid_lists):
    """Removes color 9 from the grids."""
    return _remove_color(grid_lists, 9)

def color_count(grid_lists):
    """
    Counts the occurrences of each non-zero digit in the grid and creates a single row grid
    for each unique color, with the width equal to the count of that color, up to a maximum of 30.
    If there's more than one color, it returns the original input.

    Args:
    grid_lists (list of lists of torch.Tensor): The input grid lists.

    Returns:
    list of lists of torch.Tensor: For each input grid, either a list with a single row grid
    representing the count of the single color (if only one color is present), or the original grid.
    """
    def process_single_grid(grid):
        # Get unique non-zero values and their counts
        unique, counts = torch.unique(grid[grid != 0], return_counts=True)
        
        # If there's more than one color, return the original grid
        if len(unique) > 1:
            return [grid]
        
        # If there's only one color, create a single row grid with width equal to the count, up to 30
        if len(unique) == 1:
            color = unique[0]
            count = min(counts[0].item(), 30)  # Limit the count to 30
            color_grid = torch.full((1, count), color.item(), dtype=grid.dtype, device=grid.device)
            return [color_grid]
        
        # If there are no non-zero values, return the original grid
        return [grid]

    return [sum([process_single_grid(grid) for grid in grid_list], []) for grid_list in grid_lists]