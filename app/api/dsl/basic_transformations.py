import torch
import inspect
import numpy as np

MAX_SIZE = 30
MIN_SIZE = 1


def identity(grid_lists):
    """
    Returns the input grid lists unchanged.
    This function acts as an identity operation, useful for testing or as a placeholder.
    """
    return grid_lists


def flip_horizontal(grid_lists):
    """
    Flips each grid in the grid lists horizontally.
    This reverses the order of elements along the second dimension (columns) of each grid.
    """
    return [[torch.flip(grid, [1]) for grid in grid_list] for grid_list in grid_lists]


def flip_vertical(grid_lists):
    """
    Flips each grid in the grid lists vertically.
    This reverses the order of elements along the first dimension (rows) of each grid.
    """
    return [[torch.flip(grid, [0]) for grid in grid_list] for grid_list in grid_lists]


def rotate_90(grid_lists):
    """
    Rotates each grid in the grid lists by 90 degrees clockwise.
    """
    return [
        [torch.rot90(grid, 1, [0, 1]) for grid in grid_list] for grid_list in grid_lists
    ]


def rotate_180(grid_lists):
    """
    Rotates each grid in the grid lists by 180 degrees.
    """
    return [
        [torch.rot90(grid, 2, [0, 1]) for grid in grid_list] for grid_list in grid_lists
    ]


def rotate_270(grid_lists):
    """
    Rotates each grid in the grid lists by 270 degrees clockwise (or 90 degrees counter-clockwise).
    """
    return [
        [torch.rot90(grid, 3, [0, 1]) for grid in grid_list] for grid_list in grid_lists
    ]


def inverse(grid_lists):
    """
    Inverts the values in each grid.
    For a grid with values 0-9, this operation will subtract each value from 9.
    """
    return [[9 - grid for grid in grid_list] for grid_list in grid_lists]


def transpose(grid_lists):
    """
    Transposes each grid in the grid lists.
    This operation swaps the row and column indices for all elements in each grid.
    """
    return [[grid.t() for grid in grid_list] for grid_list in grid_lists]


def add(grid_lists):
    """
    Adds all grids within each sublist of grid_lists element-wise, with modulo 10.
    If the grids in a sublist have different shapes, that sublist is returned unchanged.
    """

    def add_grids(grids):
        if not grids:
            return grids
        if not all(grid.shape == grids[0].shape for grid in grids):
            return grids
        result = grids[0].clone()
        for grid in grids[1:]:
            result = (result + grid) % 10
        return [result]

    return [add_grids(grid_list) for grid_list in grid_lists]


def main_diagonal(grid_lists):
    """
    Returns a grid of the same shape as the input, but with all elements set to zero
    except those on the main diagonal (top-left towards bottom-right).
    Works for both square and non-square grids.
    """

    def get_main_diagonal(grid):
        h, w = grid.shape
        min_dim = min(h, w)
        mask = torch.eye(min_dim, device=grid.device).bool()
        result = torch.zeros_like(grid)
        result[:min_dim, :min_dim][mask] = grid[:min_dim, :min_dim][mask]
        return result

    return [[get_main_diagonal(grid) for grid in grid_list] for grid_list in grid_lists]


def diagonal(grid_lists):
    """
    Returns a grid of the same shape as the input, but with all elements below
    the main diagonal set to zero. Elements on and above the main diagonal remain unchanged.
    Works for both square and non-square grids.
    """

    def get_upper_diagonal(grid):
        h, w = grid.shape
        min_dim = min(h, w)
        mask = torch.triu(torch.ones(min_dim, min_dim, device=grid.device)).bool()
        result = torch.zeros_like(grid)
        result[:min_dim, :min_dim][mask] = grid[:min_dim, :min_dim][mask]
        if w > h:
            result[:, min_dim:] = grid[:, min_dim:]
        return result

    return [
        [get_upper_diagonal(grid) for grid in grid_list] for grid_list in grid_lists
    ]


def main_diagonal_through_points(grid_lists):
    """
    Draws a main diagonal (top-left to bottom-right) starting from each 1-by-1 object on the grid.
    The diagonal passes through the digit and extends in both directions (up-left & down-right)
    until it encounters another object or reaches the grid boundary.

    Args:
    grid_lists (list of lists of torch.Tensor): The input grid lists.

    Returns:
    list of lists of torch.Tensor: The grids with diagonals drawn.
    """

    def process_single_grid(grid):
        new_grid = grid.clone()

        # Find all unique non-zero digits in the grid
        unique_digits = torch.unique(grid)
        unique_digits = unique_digits[unique_digits != 0]  # Exclude background (0)

        for digit in unique_digits:
            # Find all positions of the current digit
            object_positions = torch.nonzero(grid == digit, as_tuple=False)

            for pos in object_positions:
                x, y = pos[0].item(), pos[1].item()

                # Draw diagonal in the down-right direction
                cx, cy = x, y
                while cx < grid.size(0) and cy < grid.size(1):
                    # Stop drawing if we encounter another object not equal to the current digit
                    if new_grid[cx, cy] != 0 and new_grid[cx, cy] != digit:
                        break
                    new_grid[cx, cy] = digit
                    cx += 1
                    cy += 1

                # Draw diagonal in the up-left direction
                cx, cy = x, y
                while cx >= 0 and cy >= 0:
                    # Stop drawing if we encounter another object not equal to the current digit
                    if new_grid[cx, cy] != 0 and new_grid[cx, cy] != digit:
                        break
                    new_grid[cx, cy] = digit
                    cx -= 1
                    cy -= 1

        return new_grid

    return [
        [process_single_grid(grid) for grid in grid_list] for grid_list in grid_lists
    ]


def diagonal_through_points(grid_lists):
    """
    Draws a diagonal (top-right to bottom-left) starting from each 1-by-1 object on the grid.
    The diagonal passes through the digit and extends in both directions (up-right & down-left)
    until it encounters another object or reaches the grid boundary.

    Args:
    grid_lists (list of lists of torch.Tensor): The input grid lists.

    Returns:
    list of lists of torch.Tensor: The grids with diagonals drawn.
    """

    def process_single_grid(grid):
        new_grid = grid.clone()

        # Find all unique non-zero digits in the grid
        unique_digits = torch.unique(grid)
        unique_digits = unique_digits[unique_digits != 0]  # Exclude background (0)

        for digit in unique_digits:
            # Find all positions of the current digit
            object_positions = torch.nonzero(grid == digit, as_tuple=False)

            for pos in object_positions:
                x, y = pos[0].item(), pos[1].item()

                # Draw diagonal in the down-left direction
                cx, cy = x, y
                while cx < grid.size(0) and cy >= 0:
                    # Stop drawing if we encounter another object not equal to the current digit
                    if new_grid[cx, cy] != 0 and new_grid[cx, cy] != digit:
                        break
                    new_grid[cx, cy] = digit
                    cx += 1
                    cy -= 1

                # Draw diagonal in the up-right direction
                cx, cy = x, y
                while cx >= 0 and cy < grid.size(1):
                    # Stop drawing if we encounter another object not equal to the current digit
                    if new_grid[cx, cy] != 0 and new_grid[cx, cy] != digit:
                        break
                    new_grid[cx, cy] = digit
                    cx -= 1
                    cy += 1

        return new_grid

    return [
        [process_single_grid(grid) for grid in grid_list] for grid_list in grid_lists
    ]


def pad_up(grid_lists):
    """
    Adds a row of zeros at the top of each grid in the grid lists if height < MAX_SIZE.
    """

    def pad_grid_up(grid):
        if grid.shape[0] < MAX_SIZE:
            pad_row = torch.zeros(
                (1, grid.shape[1]), dtype=grid.dtype, device=grid.device
            )
            return torch.cat([pad_row, grid], dim=0)
        return grid

    return [[pad_grid_up(grid) for grid in grid_list] for grid_list in grid_lists]


def pad_down(grid_lists):
    """
    Adds a row of zeros at the bottom of each grid in the grid lists if height < MAX_SIZE.
    """

    def pad_grid_down(grid):
        if grid.shape[0] < MAX_SIZE:
            pad_row = torch.zeros(
                (1, grid.shape[1]), dtype=grid.dtype, device=grid.device
            )
            return torch.cat([grid, pad_row], dim=0)
        return grid

    return [[pad_grid_down(grid) for grid in grid_list] for grid_list in grid_lists]


def pad_left(grid_lists):
    """
    Adds a column of zeros at the left side of each grid in the grid lists if width < MAX_SIZE.
    """

    def pad_grid_left(grid):
        if grid.shape[1] < MAX_SIZE:
            pad_col = torch.zeros(
                (grid.shape[0], 1), dtype=grid.dtype, device=grid.device
            )
            return torch.cat([pad_col, grid], dim=1)
        return grid

    return [[pad_grid_left(grid) for grid in grid_list] for grid_list in grid_lists]


def pad_right(grid_lists):
    """
    Adds a column of zeros at the right side of each grid in the grid lists if width < MAX_SIZE.
    """

    def pad_grid_right(grid):
        if grid.shape[1] < MAX_SIZE:
            pad_col = torch.zeros(
                (grid.shape[0], 1), dtype=grid.dtype, device=grid.device
            )
            return torch.cat([grid, pad_col], dim=1)
        return grid

    return [[pad_grid_right(grid) for grid in grid_list] for grid_list in grid_lists]


def crop_up(grid_lists):
    """
    Removes the top row from each grid in the grid lists if height > MIN_SIZE.
    """

    def crop_grid_up(grid):
        if grid.shape[0] > MIN_SIZE:
            return grid[1:, :]
        return grid

    return [[crop_grid_up(grid) for grid in grid_list] for grid_list in grid_lists]


def crop_down(grid_lists):
    """
    Removes the bottom row from each grid in the grid lists if height > MIN_SIZE.
    """

    def crop_grid_down(grid):
        if grid.shape[0] > MIN_SIZE:
            return grid[:-1, :]
        return grid

    return [[crop_grid_down(grid) for grid in grid_list] for grid_list in grid_lists]


def crop_left(grid_lists):
    """
    Removes the leftmost column from each grid in the grid lists if width > MIN_SIZE.
    """

    def crop_grid_left(grid):
        if grid.shape[1] > MIN_SIZE:
            return grid[:, 1:]
        return grid

    return [[crop_grid_left(grid) for grid in grid_list] for grid_list in grid_lists]


def crop_right(grid_lists):
    """
    Removes the rightmost column from each grid in the grid lists if width > MIN_SIZE.
    """

    def crop_grid_right(grid):
        if grid.shape[1] > MIN_SIZE:
            return grid[:, :-1]
        return grid

    return [[crop_grid_right(grid) for grid in grid_list] for grid_list in grid_lists]


def _logical_operation(grid_lists, operation):
    """
    Applies a logical operation (OR, AND, XOR, NAND) to the first two grids in each grid list.
    If there are fewer than two grids, it returns the original grid list unchanged.

    Args:
    grid_lists (list of lists of torch.Tensor): The input grid lists.
    operation (str): The logical operation to apply ('OR', 'AND', 'XOR', 'NAND').

    Returns:
    list of lists of torch.Tensor: Result of applying the logical operation to each grid list,
    with the first two grids combined and the rest unchanged.
    """

    def process_grid_list(grid_list):
        if len(grid_list) < 2:
            return grid_list

        # Check if the first two grids have the same shape
        if grid_list[0].shape != grid_list[1].shape:
            return grid_list  # Different shapes, return original grid list

        grid0, grid1 = grid_list[0], grid_list[1]

        if operation == "OR":
            result = torch.where(grid0 != 0, grid0, grid1)
        elif operation == "AND":
            result = torch.where(
                (grid0 != 0) & (grid1 != 0), grid0, torch.zeros_like(grid0)
            )
        elif operation == "XOR":
            result = torch.where(
                (grid0 != 0) & (grid1 == 0),
                grid0,
                torch.where(
                    (grid0 == 0) & (grid1 != 0), grid1, torch.zeros_like(grid0)
                ),
            )
        elif operation == "NAND":
            and_result = torch.where(
                (grid0 != 0) & (grid1 != 0), grid0, torch.zeros_like(grid0)
            )
            result = torch.where(
                and_result == 0, torch.ones_like(grid0), torch.zeros_like(grid0)
            )

        return [result] + grid_list[2:]  # Combine the result with the remaining grids

    return [process_grid_list(grid_list) for grid_list in grid_lists]


def logical_or(grid_lists):
    """Applies logical OR operation to the first two grids in each grid list."""
    return _logical_operation(grid_lists, "OR")


def logical_and(grid_lists):
    """Applies logical AND operation to the first two grids in each grid list."""
    return _logical_operation(grid_lists, "AND")


def logical_xor(grid_lists):
    """Applies logical XOR operation to the first two grids in each grid list."""
    return _logical_operation(grid_lists, "XOR")


def logical_nand(grid_lists):
    """Applies logical NAND operation to the first two grids in each grid list."""
    return _logical_operation(grid_lists, "NAND")


def _resize_grid(grid, scale_factor):
    """Helper function to resize a grid by a given scale factor."""
    if isinstance(grid, torch.Tensor):
        return torch.nn.functional.interpolate(
            grid.unsqueeze(0).unsqueeze(0).float(),
            scale_factor=scale_factor,
            mode="nearest",
        )[0, 0].long()
    elif isinstance(grid, np.ndarray):
        return np.kron(grid, np.ones((scale_factor, scale_factor))).astype(grid.dtype)
    else:
        return [
            [cell for cell in row for _ in range(scale_factor)]
            for row in grid
            for _ in range(scale_factor)
        ]


def _check_size(grid, scale_factor):
    """Check if resizing would exceed the maximum allowed size."""
    if isinstance(grid, torch.Tensor):
        h, w = grid.shape
    elif isinstance(grid, np.ndarray):
        h, w = grid.shape
    else:
        h, w = len(grid), len(grid[0])
    return (h * scale_factor <= 30) and (w * scale_factor <= 30)


def resize_x2(grid_lists):
    """Resize grids by a factor of 2."""

    def process_grid(grid):
        return _resize_grid(grid, 2) if _check_size(grid, 2) else grid

    return [[process_grid(grid) for grid in grid_list] for grid_list in grid_lists]


def resize_x3(grid_lists):
    """Resize grids by a factor of 3."""

    def process_grid(grid):
        return _resize_grid(grid, 3) if _check_size(grid, 3) else grid

    return [[process_grid(grid) for grid in grid_list] for grid_list in grid_lists]


def resize_x4(grid_lists):
    """Resize grids by a factor of 4."""

    def process_grid(grid):
        return _resize_grid(grid, 4) if _check_size(grid, 4) else grid

    return [[process_grid(grid) for grid in grid_list] for grid_list in grid_lists]


def resize_x5(grid_lists):
    """Resize grids by a factor of 5."""

    def process_grid(grid):
        return _resize_grid(grid, 5) if _check_size(grid, 5) else grid

    return [[process_grid(grid) for grid in grid_list] for grid_list in grid_lists]


def resize_by_count(grid_lists):
    """Resize grids based on the count of colored cells."""

    def process_grid(grid):
        if isinstance(grid, torch.Tensor):
            count = torch.sum(grid != 0).item()
        elif isinstance(grid, np.ndarray):
            count = np.sum(grid != 0)
        else:
            count = sum(cell != 0 for row in grid for cell in row)

        scale_factor = max(1, min(count, 30 // max(grid.shape)))
        return (
            _resize_grid(grid, scale_factor)
            if _check_size(grid, scale_factor)
            else grid
        )

    return [[process_grid(grid) for grid in grid_list] for grid_list in grid_lists]
