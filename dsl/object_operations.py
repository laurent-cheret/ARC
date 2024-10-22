import torch
import numpy as np
from scipy import ndimage
from scipy.ndimage import label, binary_dilation, binary_fill_holes, binary_erosion
import inspect


def identify_and_isolate_objects(grid_lists):
    """
    Identifies objects in each grid and creates new grids, each containing a single object.
    An object is defined as a group of adjacent non-zero elements, including diagonal neighbors.

    Args:
    grid_lists (list of lists of torch.Tensor): The input grid lists.

    Returns:
    list of lists of torch.Tensor: For each input grid, a list of grids where each grid
    contains a single isolated object.
    """
    def process_single_grid(grid):
        # Convert PyTorch tensor to numpy array
        np_grid = grid.cpu().numpy()
        
        # Create a binary mask of non-zero elements
        binary_mask = (np_grid != 0).astype(int)
        
        # Define the structure for labeling (including diagonals)
        structure = np.ones((3, 3), dtype=int)
        
        # Use scipy's label function to identify connected components
        labeled_array, num_features = label(binary_mask, structure=structure)
        
        # Create a separate grid for each identified object
        object_grids = []
        for i in range(1, num_features + 1):
            object_mask = (labeled_array == i)
            object_grid = np.where(object_mask, np_grid, 0)
            object_grids.append(torch.tensor(object_grid, dtype=grid.dtype, device=grid.device))
        
        return object_grids

    return [sum([process_single_grid(grid) for grid in grid_list], []) for grid_list in grid_lists]

def crop_objects(grid_lists):
    """
    Crops each grid to the smallest rectangle that contains all non-zero elements.

    Args:
    grid_lists (list of lists of torch.Tensor): The input grid lists.

    Returns:
    list of lists of torch.Tensor: Cropped grids containing all non-zero elements.
    """
    def process_single_grid(grid):
        if isinstance(grid, torch.Tensor):
            device = grid.device
            np_grid = grid.cpu().numpy()
        else:
            np_grid = grid
            device = None

        # Create a binary mask of non-zero elements
        binary_mask = (np_grid != 0)

        if not np.any(binary_mask):
            # If the grid is all zeros, return it as is
            return [grid]

        # Find the bounding box of all non-zero elements
        rows = np.any(binary_mask, axis=1)
        cols = np.any(binary_mask, axis=0)
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]

        # Crop the grid
        cropped_grid = np_grid[ymin:ymax+1, xmin:xmax+1]

        # Convert back to PyTorch tensor if necessary
        if device is not None:
            cropped_grid = torch.tensor(cropped_grid, dtype=grid.dtype, device=device)

        return [cropped_grid]

    return [sum([process_single_grid(grid) for grid in grid_list], []) for grid_list in grid_lists]

def find_border(grid_lists):
    """
    Identifies objects in each grid and creates new grids, each containing a single object
    that directly touches at least one border of the grid.

    Args:
    grid_lists (list of lists of torch.Tensor): The input grid lists.

    Returns:
    list of lists of torch.Tensor: For each input grid, a list of grids where each grid
    contains a single isolated object touching a border.
    """
    def process_single_grid(grid):
        np_grid = grid.cpu().numpy()
        binary_mask = (np_grid != 0).astype(int)
        labeled_array, num_features = ndimage.label(binary_mask)
        
        height, width = np_grid.shape
        border_indices = set(np.concatenate([
            labeled_array[0, :],  # top row
            labeled_array[-1, :],  # bottom row
            labeled_array[:, 0],  # leftmost column
            labeled_array[:, -1]  # rightmost column
        ]))
        border_indices.discard(0)  # Remove background label
        
        object_grids = []
        for label in border_indices:
            object_mask = (labeled_array == label)
            object_grid = np.where(object_mask, np_grid, 0)
            object_grids.append(torch.tensor(object_grid, dtype=grid.dtype, device=grid.device))
        
        return object_grids
    
    return [sum([process_single_grid(grid) for grid in grid_list], []) for grid_list in grid_lists]

def find_center(grid_lists):
    """
    Identifies objects in each grid and creates new grids, each containing a single object
    that does not touch any border of the grid.

    Args:
    grid_lists (list of lists of torch.Tensor): The input grid lists.

    Returns:
    list of lists of torch.Tensor: For each input grid, a list of grids where each grid
    contains a single isolated object not touching any border.
    """
    def process_single_grid(grid):
        np_grid = grid.cpu().numpy()
        binary_mask = (np_grid != 0).astype(int)
        labeled_array, num_features = ndimage.label(binary_mask)
        
        height, width = np_grid.shape
        border_indices = set(np.concatenate([
            labeled_array[0, :],  # top row
            labeled_array[-1, :],  # bottom row
            labeled_array[:, 0],  # leftmost column
            labeled_array[:, -1]  # rightmost column
        ]))
        border_indices.discard(0)  # Remove background label
        
        object_grids = []
        for i in range(1, num_features + 1):
            if i not in border_indices:
                object_mask = (labeled_array == i)
                object_grid = np.where(object_mask, np_grid, 0)
                object_grids.append(torch.tensor(object_grid, dtype=grid.dtype, device=grid.device))
        
        return object_grids
    
    return [sum([process_single_grid(grid) for grid in grid_list], []) for grid_list in grid_lists]

def find_corners(grid_lists):
    """
    Identifies objects in each grid and creates new grids, each containing a single object
    that touches at least one corner of the grid.

    Args:
    grid_lists (list of lists of torch.Tensor): The input grid lists.

    Returns:
    list of lists of torch.Tensor: For each input grid, a list of grids where each grid
    contains a single isolated object touching a corner.
    """
    def process_single_grid(grid):
        np_grid = grid.cpu().numpy()
        binary_mask = (np_grid != 0).astype(int)
        labeled_array, num_features = ndimage.label(binary_mask)
        
        height, width = np_grid.shape
        corner_indices = set([
            labeled_array[0, 0],   # top-left corner
            labeled_array[0, -1],  # top-right corner
            labeled_array[-1, 0],  # bottom-left corner
            labeled_array[-1, -1]  # bottom-right corner
        ])
        corner_indices.discard(0)  # Remove background label
        
        object_grids = []
        for label in corner_indices:
            object_mask = (labeled_array == label)
            object_grid = np.where(object_mask, np_grid, 0)
            object_grids.append(torch.tensor(object_grid, dtype=grid.dtype, device=grid.device))
        
        return object_grids
    
    return [sum([process_single_grid(grid) for grid in grid_list], []) for grid_list in grid_lists]


def _paint_objects(grid_lists, color):
    """
    Helper function to paint all non-zero objects in the grids with the specified color.
    """
    def process_grid(grid):
        np_grid = grid.cpu().numpy()
        binary_mask = (np_grid != 0)
        labeled_array, _ = label(binary_mask)
        
        result = np.zeros_like(np_grid)
        result[labeled_array > 0] = color
        
        return torch.tensor(result, dtype=grid.dtype, device=grid.device)
    
    return [[process_grid(grid) for grid in grid_list] for grid_list in grid_lists]
    
def paint_obj_0(grid_lists):
    """
    Paints all non-zero objects in the grids with the color 0.
    """
    return _paint_objects(grid_lists, 0)

def paint_obj_1(grid_lists):
    """
    Paints all non-zero objects in the grids with the color 1.
    """
    return _paint_objects(grid_lists, 1)

def paint_obj_2(grid_lists):
    """
    Paints all non-zero objects in the grids with the color 2.
    """
    return _paint_objects(grid_lists, 2)

def paint_obj_3(grid_lists):
    """
    Paints all non-zero objects in the grids with the color 3.
    """
    return _paint_objects(grid_lists, 3)

def paint_obj_4(grid_lists):
    """
    Paints all non-zero objects in the grids with the color 4.
    """
    return _paint_objects(grid_lists, 4)

def paint_obj_5(grid_lists):
    """
    Paints all non-zero objects in the grids with the color 5.
    """
    return _paint_objects(grid_lists, 5)

def paint_obj_6(grid_lists):
    """
    Paints all non-zero objects in the grids with the color 6.
    """
    return _paint_objects(grid_lists, 6)

def paint_obj_7(grid_lists):
    """
    Paints all non-zero objects in the grids with the color 7.
    """
    return _paint_objects(grid_lists, 7)

def paint_obj_8(grid_lists):
    """
    Paints all non-zero objects in the grids with the color 8.
    """
    return _paint_objects(grid_lists, 8)

def paint_obj_9(grid_lists):
    """
    Paints all non-zero objects in the grids with the color 9.
    """
    return _paint_objects(grid_lists, 9)

def reorder_by_object_size(grid_lists):
    """
    Reorders the grids in each grid list based on the size of the objects (number of non-zero digits),
    from largest to smallest.

    Args:
    grid_lists (list of lists of torch.Tensor): The input grid lists.

    Returns:
    list of lists of torch.Tensor: Reordered grid lists, with grids sorted by object size.
    """
    def count_nonzero(grid):
        return torch.count_nonzero(grid).item()

    def sort_grid_list(grid_list):
        return sorted(grid_list, key=count_nonzero, reverse=True)

    return [sort_grid_list(grid_list) for grid_list in grid_lists]


def separate_by_digit(grid_lists):
    """
    Separates objects in each grid based on their digit (color) and creates
    separate grids for each unique digit, containing only the objects of that digit.

    Args:
    grid_lists (list of lists of torch.Tensor): The input grid lists.

    Returns:
    list of lists of torch.Tensor: For each input grid, a list of grids where each grid
    contains all objects of a single digit.
    """
    from scipy.ndimage import label
    def process_single_grid(grid):
        np_grid = grid.cpu().numpy()
        
        # Find unique non-zero digits
        unique_digits = np.unique(np_grid)
        unique_digits = unique_digits[unique_digits != 0]
        
        separated_grids = []
        for digit in unique_digits:
            # Create a mask for the current digit
            digit_mask = (np_grid == digit)
            
            # Label connected components
            labeled_array, _ = label(digit_mask)
            
            # Create a grid with only objects of the current digit
            digit_grid = np.where(digit_mask, np_grid, 0)
            
            separated_grids.append(torch.tensor(digit_grid, dtype=grid.dtype, device=grid.device))
        
        return separated_grids

    return [sum([process_single_grid(grid) for grid in grid_list], []) for grid_list in grid_lists]

def keep_smallest_obj(grid_lists):
    """
    Keeps only the grids with the smallest objects (fewest non-zero digits) in each grid list.

    Args:
    grid_lists (list of lists of torch.Tensor): The input grid lists.

    Returns:
    list of lists of torch.Tensor: Modified grid lists, with only the grids containing the smallest objects.
    """
    def count_nonzero(grid):
        return torch.count_nonzero(grid).item()

    def filter_smallest_objects(grid_list):
        if not grid_list:
            return []
        min_size = min(count_nonzero(grid) for grid in grid_list)
        return [grid for grid in grid_list if count_nonzero(grid) == min_size]

    return [filter_smallest_objects(grid_list) for grid_list in grid_lists]

def move_obj_up(grid_lists):
    """
    Moves all non-zero elements in each grid up by one cell, wrapping around to the bottom if necessary.
    Handles both PyTorch tensors and NumPy arrays.
    """
    def shift_up(grid):
        if isinstance(grid, torch.Tensor):
            return torch.roll(grid, shifts=-1, dims=0)
        elif isinstance(grid, np.ndarray):
            return np.roll(grid, shift=-1, axis=0)
        else:
            return grid  # Return unchanged if neither tensor nor array

    return [[shift_up(grid) for grid in grid_list] for grid_list in grid_lists]

def move_obj_down(grid_lists):
    """
    Moves all non-zero elements in each grid down by one cell, wrapping around to the top if necessary.
    Handles both PyTorch tensors and NumPy arrays.
    """
    def shift_down(grid):
        if isinstance(grid, torch.Tensor):
            return torch.roll(grid, shifts=1, dims=0)
        elif isinstance(grid, np.ndarray):
            return np.roll(grid, shift=1, axis=0)
        else:
            return grid  # Return unchanged if neither tensor nor array

    return [[shift_down(grid) for grid in grid_list] for grid_list in grid_lists]

def move_obj_left(grid_lists):
    """
    Moves all non-zero elements in each grid left by one cell, wrapping around to the right if necessary.
    Handles both PyTorch tensors and NumPy arrays.
    """
    def shift_left(grid):
        if isinstance(grid, torch.Tensor):
            return torch.roll(grid, shifts=-1, dims=1)
        elif isinstance(grid, np.ndarray):
            return np.roll(grid, shift=-1, axis=1)
        else:
            return grid  # Return unchanged if neither tensor nor array

    return [[shift_left(grid) for grid in grid_list] for grid_list in grid_lists]

def move_obj_right(grid_lists):
    """
    Moves all non-zero elements in each grid right by one cell, wrapping around to the left if necessary.
    Handles both PyTorch tensors and NumPy arrays.
    """
    def shift_right(grid):
        if isinstance(grid, torch.Tensor):
            return torch.roll(grid, shifts=1, dims=1)
        elif isinstance(grid, np.ndarray):
            return np.roll(grid, shift=1, axis=1)
        else:
            return grid  # Return unchanged if neither tensor nor array

    return [[shift_right(grid) for grid in grid_list] for grid_list in grid_lists]


def drill_hole(grid_lists):
    """
    Removes the inner part of objects in each grid, leaving only the external 1 cell width layer.
    This function handles both individual objects and the entire grid as a single object.

    Args:
    grid_lists (list of lists of torch.Tensor): The input grid lists.

    Returns:
    list of lists of torch.Tensor: Grids with objects drilled (only outer layer remaining).
    """
    def process_grid(grid):
        if isinstance(grid, torch.Tensor):
            device = grid.device
            grid = grid.cpu().numpy()
            is_torch = True
        else:
            is_torch = False

        if grid.size == 0:
            return grid

        # Create a binary mask of non-zero elements
        mask = grid != 0

        # Define structuring element for 4-connectivity
        struct = np.array([[0, 1, 0],
                           [1, 1, 1],
                           [0, 1, 0]], dtype=bool)

        # Erode the mask
        eroded = binary_erosion(mask, structure=struct)

        # The border is the difference between the original mask and the eroded mask
        border = mask & ~eroded

        # Apply the border mask to the original grid
        result = np.where(border, grid, 0)

        if is_torch:
            return torch.from_numpy(result).to(device)
        return result

    return [[process_grid(grid) for grid in grid_list] for grid_list in grid_lists]

def find_holes(grid_lists):
    """
    Finds holes in objects within each grid and paints them with 1.
    Only focuses on completely closed loops within objects.
    If there are no holes or the grid is empty, returns the original input.

    Args:
    grid_lists (list of lists of torch.Tensor): The input grid lists.

    Returns:
    list of lists of torch.Tensor: Grids with holes painted as 1, or original grids if no holes are found.
    """
    def process_grid(grid):
        if isinstance(grid, torch.Tensor):
            device = grid.device
            grid = grid.cpu().numpy()
            is_torch = True
        else:
            is_torch = False

        if grid.size == 0:
            return grid

        # Create a binary mask of non-zero elements
        mask = grid != 0

        # If the grid is empty, all zero, or all non-zero, return the original grid
        if not np.any(mask) or np.all(mask):
            return torch.from_numpy(grid).to(device) if is_torch else grid

        # Label connected components in the inverted mask
        background_labels, num_labels = label(~mask)

        # If there's only one label (i.e., no holes), return the original grid
        if num_labels == 1:
            return torch.from_numpy(grid).to(device) if is_torch else grid

        # Find the label of the largest background component
        background_sizes = np.bincount(background_labels.ravel())[1:]
        if len(background_sizes) == 0:
            return torch.from_numpy(grid).to(device) if is_torch else grid
        largest_background_label = np.argmax(background_sizes) + 1

        # The largest labeled component is the main background
        main_background = background_labels == largest_background_label

        # Fill holes in the objects
        filled_objects = binary_fill_holes(mask)

        # Identify holes: areas that are not part of the main background and not part of the original objects
        holes = ~main_background & ~mask & filled_objects

        # If no holes are found, return the original grid
        if not np.any(holes):
            return torch.from_numpy(grid).to(device) if is_torch else grid

        # Create result grid with holes painted as 1
        result = np.where(holes, 1, grid)

        if is_torch:
            return torch.from_numpy(result).to(device)
        return result

    return [[process_grid(grid) for grid in grid_list] for grid_list in grid_lists]


def draw_object_outlines(grid_lists):
    """
    Draws a 1-cell width outline around objects in each grid without changing the grid size.
    An object is defined as a group of adjacent non-zero elements, including diagonal neighbors.

    Args:
    grid_lists (list of lists of torch.Tensor): The input grid lists.

    Returns:
    list of lists of torch.Tensor: Grids with 1-cell width outlines around objects.
    """
    def process_single_grid(grid):
        # Convert PyTorch tensor to numpy array
        np_grid = grid.cpu().numpy()
        
        # Create a binary mask of non-zero elements
        binary_mask = (np_grid != 0).astype(int)
        
        # Define the structure for labeling (including diagonals)
        structure = np.ones((3, 3), dtype=int)
        
        # Use scipy's label function to identify connected components
        labeled_array, num_features = label(binary_mask, structure=structure)
        
        # Create the outline grid
        outline_grid = np.zeros_like(np_grid)
        
        for i in range(1, num_features + 1):
            object_mask = (labeled_array == i)
            # Dilate the object mask
            dilated_mask = binary_dilation(object_mask, structure=structure)
            # The outline is where the dilated mask is True but the original mask is False
            outline = dilated_mask & ~object_mask
            # Set the outline in the grid
            outline_grid[outline] = 1
        
        # Convert back to PyTorch tensor
        return torch.tensor(outline_grid, dtype=grid.dtype, device=grid.device)

    return [[process_single_grid(grid) for grid in grid_list] for grid_list in grid_lists]

def reverse_object_background(grid_lists):
    """
    Reverses objects and background in each grid.
    Objects become zeros, and the background takes the color of the object.
    If there are more than two unique digits, it uses the most common non-zero digit for the background.

    Args:
    grid_lists (list of lists of torch.Tensor): The input grid lists.

    Returns:
    list of lists of torch.Tensor: Grids with reversed objects and background.
    """
    def process_single_grid(grid):
        # Get unique values and their counts
        unique, counts = torch.unique(grid, return_counts=True)
        
        if len(unique) == 1:
            # If there's only one unique value, return the grid as is
            return grid
        
        if len(unique) == 2 and 0 in unique:
            # If there are only two unique values and one is zero
            background_color = unique[unique != 0].item()
        else:
            # If there are more than two unique values or two non-zero values
            # Find the most common non-zero value
            non_zero_mask = unique != 0
            non_zero_unique = unique[non_zero_mask]
            non_zero_counts = counts[non_zero_mask]
            background_color = non_zero_unique[non_zero_counts.argmax()].item()
        
        # Create the reversed grid
        reversed_grid = torch.full_like(grid, background_color)
        reversed_grid[grid != 0] = 0
        
        return reversed_grid

    return [[process_single_grid(grid) for grid in grid_list] for grid_list in grid_lists]


def _copy_object(grid, direction):
    """Helper function to copy objects in a specified direction while keeping the original."""
    if isinstance(grid, torch.Tensor):
        is_torch = True
        device = grid.device
        np_grid = grid.cpu().numpy()
    elif isinstance(grid, np.ndarray):
        is_torch = False
        np_grid = grid
    else:
        raise TypeError("Input must be either a PyTorch tensor or a NumPy array")

    binary_mask = (np_grid != 0).astype(int)
    structure = np.ones((3, 3), dtype=int)
    labeled_array, num_features = label(binary_mask, structure=structure)
    
    h, w = np_grid.shape
    pad_top, pad_bottom, pad_left, pad_right = 0, 0, 0, 0
    
    if 'up' in direction:
        pad_top = h
    if 'down' in direction:
        pad_bottom = h
    if 'left' in direction:
        pad_left = w
    if 'right' in direction:
        pad_right = w
    
    new_h, new_w = h + pad_top + pad_bottom, w + pad_left + pad_right
    if new_h > 30 or new_w > 30:
        return grid  # Return original grid if new size would exceed 30x30
    
    new_grid = np.zeros((new_h, new_w), dtype=np_grid.dtype)
    new_grid[pad_top:pad_top+h, pad_left:pad_left+w] = np_grid  # Keep the original object
    
    for i in range(1, num_features + 1):
        object_mask = (labeled_array == i)
        object_slice = np_grid * object_mask
        if 'up' in direction:
            new_grid[:h, pad_left:pad_left+w] += object_slice
        if 'down' in direction:
            new_grid[pad_top+h:, pad_left:pad_left+w] += object_slice
        if 'left' in direction:
            new_grid[pad_top:pad_top+h, :w] += object_slice
        if 'right' in direction:
            new_grid[pad_top:pad_top+h, pad_left+w:] += object_slice
    
    if is_torch:
        return torch.tensor(new_grid, dtype=grid.dtype, device=device)
    else:
        return new_grid

def copy_obj_up(grid_lists):
    return [[_copy_object(grid, 'up') for grid in grid_list] for grid_list in grid_lists]

def copy_obj_down(grid_lists):
    return [[_copy_object(grid, 'down') for grid in grid_list] for grid_list in grid_lists]

def copy_obj_left(grid_lists):
    return [[_copy_object(grid, 'left') for grid in grid_list] for grid_list in grid_lists]

def copy_obj_right(grid_lists):
    return [[_copy_object(grid, 'right') for grid in grid_list] for grid_list in grid_lists]

def copy_obj_diag_top_right(grid_lists):
    return [[_copy_object(grid, 'up right') for grid in grid_list] for grid_list in grid_lists]

def copy_obj_diag_top_left(grid_lists):
    return [[_copy_object(grid, 'up left') for grid in grid_list] for grid_list in grid_lists]

def copy_obj_diag_bottom_right(grid_lists):
    return [[_copy_object(grid, 'down right') for grid in grid_list] for grid_list in grid_lists]

def copy_obj_diag_bottom_left(grid_lists):
    return [[_copy_object(grid, 'down left') for grid in grid_list] for grid_list in grid_lists]


def find_common_obj(grid_lists):
    """
    Identifies the most common object(s) across all grid_0s in the different grid lists.
    Returns new grid lists with only the most common object(s) in their original positions for each grid_0.
    If there are no common objects or if any grid_list is empty, it returns the original input.
    """
    def extract_objects(grid):
        if isinstance(grid, torch.Tensor):
            np_grid = grid.cpu().numpy()
        elif isinstance(grid, np.ndarray):
            np_grid = grid
        else:
            raise TypeError("Grid must be either a PyTorch tensor or a NumPy array")
        
        binary_mask = (np_grid != 0).astype(int)
        structure = np.ones((3, 3), dtype=int)
        labeled_array, num_features = label(binary_mask, structure=structure)
        
        objects = []
        for i in range(1, num_features + 1):
            obj_mask = labeled_array == i
            obj = np_grid[obj_mask]
            objects.append((obj.tobytes(), obj.shape))  # Store object content and shape
        
        return objects

    # Extract objects from all grid_0s, skipping empty grid_lists
    all_objects = [extract_objects(grid_list[0]) for grid_list in grid_lists if grid_list]

    # If any grid_list was empty, return the original input
    if len(all_objects) != len(grid_lists):
        return grid_lists

    # Count occurrences of each unique object
    object_counts = {}
    for objects in all_objects:
        unique_objects = set(objects)
        for obj in unique_objects:
            if obj in object_counts:
                object_counts[obj] += 1
            else:
                object_counts[obj] = 1

    # If there are no common objects, return the original input
    if not object_counts:
        return grid_lists

    # Find the maximum count
    max_count = max(object_counts.values())

    # Find objects with the maximum count
    common_objects = [obj for obj, count in object_counts.items() if count == max_count]

    def process_grid(grid):
        if isinstance(grid, torch.Tensor):
            device = grid.device
            np_grid = grid.cpu().numpy()
        elif isinstance(grid, np.ndarray):
            device = None
            np_grid = grid
        else:
            raise TypeError("Grid must be either a PyTorch tensor or a NumPy array")

        binary_mask = (np_grid != 0).astype(int)
        structure = np.ones((3, 3), dtype=int)
        labeled_array, _ = label(binary_mask, structure=structure)

        new_grid = np.zeros_like(np_grid)
        for i in range(1, labeled_array.max() + 1):
            obj_mask = labeled_array == i
            obj = np_grid[obj_mask]
            obj_tuple = (obj.tobytes(), obj.shape)
            if obj_tuple in common_objects:
                new_grid[obj_mask] = np_grid[obj_mask]

        if device is not None:
            return torch.tensor(new_grid, dtype=grid.dtype, device=device)
        else:
            return new_grid

    # Process only grid_0 in each non-empty grid list
    return [[process_grid(grid_list[0])] + grid_list[1:] if grid_list else grid_list for grid_list in grid_lists]

def paint_common_color(grid_lists):
    """
    Paints all non-zero objects in each grid with the most common non-zero color (digit).
    The dominant color is determined by the number of objects, not the number of cells.
    Zeros (background) are left unchanged.
    Works with both PyTorch tensors and numpy arrays.
    """
    def process_grid(grid):
        if isinstance(grid, torch.Tensor):
            np_grid = grid.cpu().numpy()
        elif isinstance(grid, np.ndarray):
            np_grid = grid
        else:
            np_grid = np.array(grid)
        
        if np_grid.size == 0:
            return grid

        # Create a binary mask of non-zero elements
        binary_mask = (np_grid != 0)
        
        # Label connected components
        structure = np.ones((3, 3), dtype=int)  # 8-connectivity
        labeled, num_features = label(binary_mask, structure=structure)
        
        # Count objects of each color
        color_counts = {}
        for i in range(1, num_features + 1):
            object_color = np.unique(np_grid[labeled == i])[0]
            if object_color in color_counts:
                color_counts[object_color] += 1
            else:
                color_counts[object_color] = 1
        
        if not color_counts:  # If no objects found
            return grid
        
        # Find the dominant color
        dominant_color = max(color_counts, key=color_counts.get)
        
        # Paint non-zero elements with the dominant color
        result = np.where(np_grid != 0, dominant_color, np_grid)
        
        if isinstance(grid, torch.Tensor):
            return torch.from_numpy(result).to(grid.device).type(grid.dtype)
        elif isinstance(grid, np.ndarray):
            return result
        else:
            return result.tolist()

    return [[process_grid(grid) for grid in grid_list] for grid_list in grid_lists]