import torch
import numpy as np
from collections import deque
from scipy.ndimage import label, center_of_mass
from scipy import stats

# def collide(grid_lists):
#     """
#     Simulates collision between the first two objects in each grid list.
#     Objects from the first grid move linearly towards objects in the second grid until they touch.
#     The moving object stays entirely within the grid and maintains its shape.
#     If they never touch or can't move, the original list is returned.
#     """
#     def process_grid_list(grid_list):
#         if len(grid_list) < 2:
#             return grid_list  # Return original list if there's nothing to collide

#         original_grid1, original_grid2 = grid_list[0], grid_list[1]
#         grid1, grid2 = original_grid1.clone(), original_grid2.clone()
        
#         if grid1.shape != grid2.shape:
#             return grid_list  # Return original list if shapes don't match

#         def find_closest_points(grid1, grid2):
#             points1 = torch.nonzero(grid1)
#             points2 = torch.nonzero(grid2)
#             if len(points1) == 0 or len(points2) == 0:
#                 return None, None
#             distances = torch.cdist(points1.float(), points2.float(), p=1)  # Manhattan distance
#             min_distance_idx = distances.argmin()
#             closest1 = points1[min_distance_idx // len(points2)]
#             closest2 = points2[min_distance_idx % len(points2)]
#             return closest1, closest2

#         def can_move(grid, direction):
#             if direction[0] != 0:  # Vertical movement
#                 if direction[0] > 0:  # Move down
#                     return not torch.any(grid[-1, :] != 0)
#                 else:  # Move up
#                     return not torch.any(grid[0, :] != 0)
#             else:  # Horizontal movement
#                 if direction[1] > 0:  # Move right
#                     return not torch.any(grid[:, -1] != 0)
#                 else:  # Move left
#                     return not torch.any(grid[:, 0] != 0)

#         def move_object(grid, direction):
#             if not can_move(grid, direction):
#                 return grid  # Can't move, return original grid
#             new_grid = torch.zeros_like(grid)
#             if direction[0] != 0:  # Vertical movement
#                 if direction[0] > 0:  # Move down
#                     new_grid[1:, :] = grid[:-1, :]
#                 else:  # Move up
#                     new_grid[:-1, :] = grid[1:, :]
#             else:  # Horizontal movement
#                 if direction[1] > 0:  # Move right
#                     new_grid[:, 1:] = grid[:, :-1]
#                 else:  # Move left
#                     new_grid[:, :-1] = grid[:, 1:]
#             return new_grid

#         def objects_touching(grid1, grid2):
#             return torch.any((grid1.roll(1, 0) | grid1.roll(-1, 0) | 
#                               grid1.roll(1, 1) | grid1.roll(-1, 1)) & grid2)

#         from_point, to_point = find_closest_points(grid1, grid2)
#         if from_point is None or to_point is None:
#             return grid_list  # No collision possible

#         # Determine movement direction
#         direction = torch.sign(to_point - from_point)
#         if direction[0] != 0 and direction[1] != 0:
#             # If diagonal, choose the dimension with larger difference
#             if abs(to_point[0] - from_point[0]) > abs(to_point[1] - from_point[1]):
#                 direction[1] = 0
#             else:
#                 direction[0] = 0

#         collision_occurred = False
#         while not objects_touching(grid1, grid2):
#             new_grid1 = move_object(grid1, direction)
#             if torch.equal(new_grid1, grid1):
#                 break  # No more movement possible
#             grid1 = new_grid1
#             if objects_touching(grid1, grid2):
#                 collision_occurred = True

#         if collision_occurred:
#             result_grid = grid1 | grid2
#             return [result_grid] + grid_list[2:]  # Combine first two grids and keep the rest
#         else:
#             return grid_list  # Return original list if no collision occurred

#     return [process_grid_list(grid_list) for grid_list in grid_lists]


def collide(grid_lists):
    """
    Simulates collision between the first two objects in each grid list.
    Objects from the first grid move linearly (horizontally or vertically) towards objects in the second grid until they are one step away from touching.
    The moving object stays entirely within the grid and maintains its shape.
    If they never get close enough to touch or can't move, the original list is returned.
    Objects can have different non-zero values (colors).
    """
    def process_grid_list(grid_list):
        if len(grid_list) < 2:
            return grid_list  # Return original list if there's nothing to collide

        original_grid1, original_grid2 = grid_list[0], grid_list[1]
        grid1, grid2 = original_grid1.clone(), original_grid2.clone()
        
        if grid1.shape != grid2.shape:
            return grid_list  # Return original list if shapes don't match

        def find_closest_points(grid1, grid2):
            points1 = torch.nonzero(grid1)
            points2 = torch.nonzero(grid2)
            if len(points1) == 0 or len(points2) == 0:
                return None, None
            distances = torch.cdist(points1.float(), points2.float(), p=1)  # Manhattan distance
            min_distance_idx = distances.argmin()
            closest1 = points1[min_distance_idx // len(points2)]
            closest2 = points2[min_distance_idx % len(points2)]
            return closest1, closest2

        def can_move(grid, direction):
            if direction[0] != 0:  # Vertical movement
                if direction[0] > 0:  # Move down
                    return not torch.any(grid[-1, :] != 0)
                else:  # Move up
                    return not torch.any(grid[0, :] != 0)
            else:  # Horizontal movement
                if direction[1] > 0:  # Move right
                    return not torch.any(grid[:, -1] != 0)
                else:  # Move left
                    return not torch.any(grid[:, 0] != 0)

        def move_object(grid, direction):
            if not can_move(grid, direction):
                return grid  # Can't move, return original grid
            new_grid = torch.zeros_like(grid)
            if direction[0] != 0:  # Vertical movement
                if direction[0] > 0:  # Move down
                    new_grid[1:, :] = grid[:-1, :]
                else:  # Move up
                    new_grid[:-1, :] = grid[1:, :]
            else:  # Horizontal movement
                if direction[1] > 0:  # Move right
                    new_grid[:, 1:] = grid[:, :-1]
                else:  # Move left
                    new_grid[:, :-1] = grid[:, 1:]
            return new_grid

        def objects_touching(grid1, grid2):
            return torch.any((grid1 != 0) & (grid2 != 0))

        from_point, to_point = find_closest_points(grid1, grid2)
        if from_point is None or to_point is None:
            return grid_list  # No collision possible

        # Determine movement direction (only linear movement allowed)
        direction = torch.zeros(2, dtype=torch.int)
        if abs(to_point[0] - from_point[0]) > abs(to_point[1] - from_point[1]):
            direction[0] = torch.sign(to_point[0] - from_point[0]).item()
        else:
            direction[1] = torch.sign(to_point[1] - from_point[1]).item()

        collision_occurred = False
        previous_grid1 = grid1.clone()
        while not objects_touching(grid1, grid2):
            new_grid1 = move_object(grid1, direction)
            if torch.equal(new_grid1, grid1):
                break  # No more movement possible
            previous_grid1 = grid1.clone()
            grid1 = new_grid1
            if objects_touching(grid1, grid2):
                collision_occurred = True
                grid1 = previous_grid1  # Move back one step
                break

        if collision_occurred:
            return [grid1, grid2] + grid_list[2:]  # Return the grids one step before collision
        else:
            return grid_list  # Return original list if no collision occurred

    return [process_grid_list(grid_list) for grid_list in grid_lists]

def connect(grid_lists):
    """
    Connects objects of the same digit within each grid by drawing paths between them.
    """
    def process_single_grid(grid):
        def find_objects(grid):
            objects = {}
            for digit in torch.unique(grid):
                if digit != 0:  # Ignore background
                    objects[digit.item()] = torch.nonzero(grid == digit)
            return objects

        def find_closest_points(points1, points2):
            min_distance = float('inf')
            closest1, closest2 = None, None
            for p1 in points1:
                for p2 in points2:
                    distance = abs(int(p1[0]) - int(p2[0])) + abs(int(p1[1]) - int(p2[1]))
                    if distance < min_distance:
                        min_distance = distance
                        closest1, closest2 = p1, p2
            return closest1, closest2

        def draw_path(grid, from_point, to_point, digit):
            new_grid = grid.clone()
            current = from_point.clone()

            while not torch.all(current == to_point):
                if abs(int(current[0]) - int(to_point[0])) > abs(int(current[1]) - int(to_point[1])):
                    step = torch.tensor([1 if to_point[0] > current[0] else -1, 0], dtype=torch.long)
                else:
                    step = torch.tensor([0, 1 if to_point[1] > current[1] else -1], dtype=torch.long)

                current = current + step
                new_grid[current[0], current[1]] = digit

            return new_grid

        objects = find_objects(grid)
        new_grid = grid.clone()

        for digit, points in objects.items():
            if len(points) < 2:
                continue  # No need to connect if there's only one object of this digit

            # Connect all objects of the same digit
            for i in range(len(points) - 1):
                from_point, to_point = find_closest_points(points[i].unsqueeze(0), points[i+1:])
                new_grid = draw_path(new_grid, from_point, to_point, digit)

        return new_grid

    return [[process_single_grid(grid) for grid in grid_list] for grid_list in grid_lists]

def connect_straight(grid_lists):
    """
    Connects objects of the same digit within each grid by drawing straight paths between them.
    Objects are connected only if they share the same row or column.
    """
    def process_single_grid(grid):
        def find_objects(grid):
            objects = {}
            for digit in torch.unique(grid):
                if digit != 0:  # Ignore background
                    objects[digit.item()] = torch.nonzero(grid == digit)
            return objects

        def can_connect(point1, point2):
            return point1[0] == point2[0] or point1[1] == point2[1]

        def draw_straight_path(from_point, to_point, digit):
            path = []
            if from_point[0] == to_point[0]:  # Same row
                start, end = min(from_point[1], to_point[1]), max(from_point[1], to_point[1])
                path = [(from_point[0], i) for i in range(start + 1, end)]
            else:  # Same column
                start, end = min(from_point[0], to_point[0]), max(from_point[0], to_point[0])
                path = [(i, from_point[1]) for i in range(start + 1, end)]
            return path, digit

        objects = find_objects(grid)
        new_paths = []

        for digit, points in objects.items():
            if len(points) < 2:
                continue  # No need to connect if there's only one object of this digit

            # Try to connect all pairs of points
            for i in range(len(points)):
                for j in range(i + 1, len(points)):
                    if can_connect(points[i], points[j]):
                        path, color = draw_straight_path(points[i], points[j], digit)
                        new_paths.extend([(p, color) for p in path])

        # Apply new paths to the grid
        new_grid = grid.clone()
        for (x, y), color in new_paths:
            if new_grid[x, y] == 0:  # Only draw on empty cells
                new_grid[x, y] = color

        return new_grid

    return [[process_single_grid(grid) for grid in grid_list] for grid_list in grid_lists]


def connect_to_source(grid_lists):
    """
    Connects objects from grid1 to the source object in grid0 using straight lines.
    Connections are made only if they share the same row or column.
    The color of the connection is the color of the starting point in grid1.
    If there's a shape mismatch or any other issue, it returns the original grid lists.
    """
    def process_grid_pair(grid0, grid1):
        if isinstance(grid0, torch.Tensor):
            grid0 = grid0.cpu().numpy()
            grid1 = grid1.cpu().numpy()
            is_torch = True
        else:
            is_torch = False

        # Check if shapes match
        if grid0.shape != grid1.shape:
            return None  # Signal that processing couldn't be done

        # Find the source object in grid0
        source_mask = grid0 != 0
        source_labeled, _ = label(source_mask)
        source_coords = np.argwhere(source_labeled == 1)  # Assuming the first object is the source

        # Find objects in grid1
        grid1_mask = grid1 != 0
        grid1_labeled, num_objects = label(grid1_mask)

        # Create a new grid for the connections
        connection_grid = np.zeros_like(grid1)

        for obj_id in range(1, num_objects + 1):
            obj_coords = np.argwhere(grid1_labeled == obj_id)
            for coord in obj_coords:
                for source_coord in source_coords:
                    if coord[0] == source_coord[0] or coord[1] == source_coord[1]:
                        # They share a row or column, so we can connect
                        color = grid1[tuple(coord)]
                        if coord[0] == source_coord[0]:  # Same row
                            start, end = min(coord[1], source_coord[1]), max(coord[1], source_coord[1])
                            connection_grid[coord[0], start:end+1] = color
                        else:  # Same column
                            start, end = min(coord[0], source_coord[0]), max(coord[0], source_coord[0])
                            connection_grid[start:end+1, coord[1]] = color
                        break  # Connect only once per object point

        # Combine the original grids with the connection grid
        result_grid = np.where(grid1 != 0, grid1, connection_grid)
        result_grid = np.where(grid0 != 0, grid0, result_grid)

        if is_torch:
            return torch.from_numpy(result_grid)
        return result_grid

    # Try to process each grid list
    processed_grid_lists = []
    for grid_list in grid_lists:
        if len(grid_list) < 2:
            processed_grid_lists.append(grid_list)  # Keep original if less than 2 grids
        else:
            try:
                processed_pair = process_grid_pair(grid_list[0], grid_list[1])
                if processed_pair is None:
                    return grid_lists  # Return original if any pair can't be processed
                processed_grid_lists.append([processed_pair] + grid_list[2:])
            except Exception as e:
                # print(f"Error in processing: {e}. Returning original grid lists.")
                return grid_lists  # Return original if any exception occurs

    return processed_grid_lists

def connect_regardless(grid_lists):
    """
    Connects different objects in each grid, starting from the object with the lowest digit.
    Connections are made first horizontally (left or right), then vertically (up or down).

    Args:
    grid_lists (list of lists of torch.Tensor): The input grid lists.

    Returns:
    list of lists of torch.Tensor: Modified grid lists with objects connected.
    """
    def process_grid(grid):
        if isinstance(grid, torch.Tensor):
            grid = grid.cpu().numpy()
            is_torch = True
        else:
            is_torch = False

        # Find objects
        labeled, num_objects = label(grid != 0)
        
        if num_objects < 2:
            return grid  # No need to connect if there are fewer than 2 objects

        # Get object properties
        objects = []
        for i in range(1, num_objects + 1):
            obj_mask = labeled == i
            coords = np.argwhere(obj_mask)
            center = coords.mean(axis=0).astype(int)
            min_value = grid[obj_mask].min()
            objects.append((min_value, center, coords))

        # Sort objects by their minimum value
        objects.sort(key=lambda x: x[0])

        # Connect objects
        result = grid.copy()
        for i in range(len(objects) - 1):
            start_center = objects[i][1]
            end_center = objects[i+1][1]
            color = objects[i][0]  # Use the color of the starting object

            # Horizontal connection
            x_start, x_end = sorted([start_center[1], end_center[1]])
            for x in range(x_start, x_end + 1):
                result[start_center[0], x] = color

            # Vertical connection
            y_start, y_end = sorted([start_center[0], end_center[0]])
            for y in range(y_start, y_end + 1):
                result[y, end_center[1]] = color

        if is_torch:
            return torch.from_numpy(result)
        return result

    return [[process_grid(grid) for grid in grid_list] for grid_list in grid_lists]


def move_obj_to_source(grid_lists):
    """
    Moves the object in grid1 to each source object in grid0. If there's a single object
    in grid1, it will be copied to each source location in grid0. If there are multiple
    objects in grid1, the function does nothing. Other grids in the grid list remain unchanged.

    Args:
    grid_lists (list of lists of torch.Tensor): The input grid lists.

    Returns:
    list of lists of torch.Tensor: Modified grid lists with the object in grid1 moved if applicable.
    """
    def process_grid_pair(grid0, grid1):
        if isinstance(grid0, torch.Tensor):
            grid0 = grid0.cpu().numpy()
            grid1 = grid1.cpu().numpy()
            is_torch = True
        else:
            is_torch = False

        # Find the source objects in grid0
        source_mask = grid0 != 0
        source_labeled, num_sources = label(source_mask)

        # Find objects in grid1
        obj_mask = grid1 != 0
        obj_labeled, num_objects = label(obj_mask)

        # If there's more than one object in grid1, return the original grid1
        if num_objects != 1:
            if is_torch:
                return torch.from_numpy(grid1)
            return grid1

        # Find the bounding box of the object in grid1
        obj_coords = np.argwhere(obj_mask)
        min_y, min_x = obj_coords.min(axis=0)
        max_y, max_x = obj_coords.max(axis=0)

        # Calculate the center of the bounding rectangle
        obj_center = np.array([(min_y + max_y) // 2, (min_x + max_x) // 2])

        # Create a new grid for the moved objects
        new_grid1 = np.zeros_like(grid1)

        # Move the object to each source location
        h, w = grid1.shape
        for source_id in range(1, num_sources + 1):
            source_coords = np.argwhere(source_labeled == source_id)
            source_center = source_coords.mean(axis=0).astype(int)

            # Calculate the shift
            shift = source_center - obj_center

            # Copy the object to the new location
            for i in range(min_y, max_y + 1):
                for j in range(min_x, max_x + 1):
                    if grid1[i, j] != 0:
                        new_i, new_j = i + shift[0], j + shift[1]
                        if 0 <= new_i < h and 0 <= new_j < w:
                            new_grid1[new_i, new_j] = grid1[i, j]

        if is_torch:
            return torch.from_numpy(new_grid1)
        return new_grid1

    # Process only the first two grids in each grid list
    return [[process_grid_pair(grid_list[0], grid_list[1])] + grid_list[2:] if len(grid_list) > 1 else grid_list 
            for grid_list in grid_lists]

def fit_rectangle(grid_lists):
    """
    Fits each group of same-colored pixels into the smallest possible rectangle or square.
    Works with both PyTorch tensors and numpy arrays.
    """
    def process_grid(grid):
        if isinstance(grid, torch.Tensor):
            return _process_torch_grid(grid)
        elif isinstance(grid, np.ndarray):
            return _process_numpy_grid(grid)
        else:
            return _process_list_grid(grid)

    def _process_torch_grid(grid):
        if grid.numel() == 0:
            return grid
        
        unique_colors = torch.unique(grid)
        new_grid = torch.zeros_like(grid)
        
        for color in unique_colors:
            if color == 0:  # Skip background
                continue
            mask = grid == color
            if mask.any():
                rows, cols = torch.where(mask)
                min_row, max_row = rows.min().item(), rows.max().item()
                min_col, max_col = cols.min().item(), cols.max().item()
                new_grid[min_row:max_row+1, min_col:max_col+1] = color
        
        return new_grid

    def _process_numpy_grid(grid):
        if grid.size == 0:
            return grid
        
        unique_colors = np.unique(grid)
        new_grid = np.zeros_like(grid)
        
        for color in unique_colors:
            if color == 0:  # Skip background
                continue
            mask = grid == color
            if mask.any():
                rows, cols = np.where(mask)
                min_row, max_row = rows.min(), rows.max()
                min_col, max_col = cols.min(), cols.max()
                new_grid[min_row:max_row+1, min_col:max_col+1] = color
        
        return new_grid

    def _process_list_grid(grid):
        if not grid:
            return grid
        
        height, width = len(grid), len(grid[0])
        new_grid = [[0 for _ in range(width)] for _ in range(height)]
        
        def find_bounds(r, c, color):
            stack = [(r, c)]
            min_r, max_r, min_c, max_c = r, r, c, c
            while stack:
                r, c = stack.pop()
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < height and 0 <= nc < width and grid[nr][nc] == color:
                            min_r, max_r = min(min_r, nr), max(max_r, nr)
                            min_c, max_c = min(min_c, nc), max(max_c, nc)
                            stack.append((nr, nc))
            return min_r, max_r, min_c, max_c

        for r in range(height):
            for c in range(width):
                color = grid[r][c]
                if color != 0 and new_grid[r][c] == 0:
                    min_r, max_r, min_c, max_c = find_bounds(r, c, color)
                    for i in range(min_r, max_r + 1):
                        for j in range(min_c, max_c + 1):
                            new_grid[i][j] = color
        
        return new_grid

    return [[process_grid(grid) for grid in grid_list] for grid_list in grid_lists]

def fill_pattern(grid_lists):
    """
    Fills each grid by repeating the pattern of the original object throughout the grid in all directions.
    The original object remains in its place, and the pattern extends in all directions.

    Args:
    grid_lists (list of lists of torch.Tensor): The input grid lists.

    Returns:
    list of lists of torch.Tensor: Grids filled with the repeating pattern.
    """
    def process_grid(grid):
        if isinstance(grid, torch.Tensor):
            return _process_torch_grid(grid)
        elif isinstance(grid, np.ndarray):
            return _process_numpy_grid(grid)
        else:
            return _process_list_grid(grid)

    def _process_torch_grid(grid):
        if grid.numel() == 0:
            return grid
        
        # Find the bounding box of the pattern
        mask = grid != 0
        rows, cols = torch.where(mask)
        if len(rows) == 0:  # If grid is empty
            return grid
        
        min_row, max_row = rows.min().item(), rows.max().item()
        min_col, max_col = cols.min().item(), cols.max().item()
        
        # Extract the pattern
        pattern = grid[min_row:max_row+1, min_col:max_col+1]
        pattern_h, pattern_w = pattern.shape
        
        # Create a new grid
        h, w = grid.shape
        new_grid = torch.zeros_like(grid)
        
        # Fill the new grid with the pattern in all directions
        for i in range(min_row, -pattern_h, -pattern_h):
            for j in range(min_col, -pattern_w, -pattern_w):
                new_grid[max(0, i):min(h, i+pattern_h), max(0, j):min(w, j+pattern_w)] = \
                    pattern[max(0, -i):pattern_h-max(0, i+pattern_h-h), 
                            max(0, -j):pattern_w-max(0, j+pattern_w-w)]
        
        for i in range(min_row, -pattern_h, -pattern_h):
            for j in range(min_col, w, pattern_w):
                new_grid[max(0, i):min(h, i+pattern_h), j:min(w, j+pattern_w)] = \
                    pattern[max(0, -i):pattern_h-max(0, i+pattern_h-h), 
                            :min(w-j, pattern_w)]
        
        for i in range(min_row, h, pattern_h):
            for j in range(min_col, -pattern_w, -pattern_w):
                new_grid[i:min(h, i+pattern_h), max(0, j):min(w, j+pattern_w)] = \
                    pattern[:min(h-i, pattern_h), 
                            max(0, -j):pattern_w-max(0, j+pattern_w-w)]
        
        for i in range(min_row, h, pattern_h):
            for j in range(min_col, w, pattern_w):
                new_grid[i:min(h, i+pattern_h), j:min(w, j+pattern_w)] = \
                    pattern[:min(h-i, pattern_h), :min(w-j, pattern_w)]
        
        return new_grid

    def _process_numpy_grid(grid):
        return _process_torch_grid(torch.from_numpy(grid)).numpy()

    def _process_list_grid(grid):
        return _process_torch_grid(torch.tensor(grid)).tolist()

    return [[process_grid(grid) for grid in grid_list] for grid_list in grid_lists]


def draw_spiral(grid_lists):
    """
    Draws a clockwise spiral on each grid, starting from the upper left corner.
    The spiral is a single continuous line that moves inward, maintaining a one-cell gap from previous parts.
    The outermost loop goes along the edges of the grid.
    If the grid is too small to draw a spiral, it returns the original grid.
    
    Args:
    grid_lists (list of lists of torch.Tensor or np.ndarray): The input grid lists.
    
    Returns:
    list of lists: Grids with spirals drawn on them, or original grids if too small.
    """
    def process_grid(grid):
        if isinstance(grid, torch.Tensor):
            device = grid.device
            grid = grid.cpu().numpy()
            is_torch = True
        else:
            is_torch = False
        
        height, width = grid.shape
        
        # If the grid is too small to draw a spiral, return the original grid
        if height < 3 or width < 3:
            return torch.from_numpy(grid).to(device) if is_torch else grid
        
        spiral = np.zeros((height, width), dtype=int)
        
        row, col = 0, 0
        direction = 0  # 0: right, 1: down, 2: left, 3: up
        first_loop = True
        
        while True:
            spiral[row, col] = 1
            
            if direction == 0:  # Moving right
                if first_loop and col + 1 < width:
                    col += 1
                elif not first_loop and col + 2 < width and spiral[row, col + 2] == 0:
                    col += 1
                else:
                    direction = 1
                    if row + 1 >= height or spiral[row + 1, col] == 1:
                        break
            elif direction == 1:  # Moving down
                if first_loop and row + 1 < height:
                    row += 1
                elif not first_loop and row + 2 < height and spiral[row + 2, col] == 0:
                    row += 1
                else:
                    direction = 2
                    if col - 1 < 0 or spiral[row, col - 1] == 1:
                        break
            elif direction == 2:  # Moving left
                if first_loop and col - 1 >= 0:
                    col -= 1
                elif not first_loop and col - 2 >= 0 and spiral[row, col - 2] == 0:
                    col -= 1
                else:
                    direction = 3
                    first_loop = False
                    if row - 1 < 0 or spiral[row - 1, col] == 1:
                        break
            elif direction == 3:  # Moving up
                if first_loop and row - 1 > 0:
                    row -= 1
                elif not first_loop and row - 2 >= 0 and spiral[row - 2, col] == 0:
                    row -= 1
                else:
                    direction = 0
                    if col + 1 >= width or spiral[row, col + 1] == 1:
                        break
        
        # Overlay the spiral on the original grid
        result = np.where(spiral != 0, spiral, grid)
        
        if is_torch:
            return torch.from_numpy(result).to(device)
        return result
    
    return [[process_grid(grid) for grid in grid_list] for grid_list in grid_lists]



def find_object_center(grid_lists):
    """
    Finds the central cell(s) of objects in each grid.
    For odd dimensions, it returns a single central cell.
    For even dimensions, it returns the 2x2 or 1x2 or 2x1 central region.
    
    Args:
    grid_lists (list of lists of torch.Tensor or np.ndarray): The input grid lists.
    
    Returns:
    list of lists of torch.Tensor or np.ndarray: Grids with only the central cell(s) of objects marked.
    """
    def process_single_grid(grid):
        # Check if input is PyTorch tensor or NumPy array
        if isinstance(grid, torch.Tensor):
            np_grid = grid.cpu().numpy()
            is_torch = True
            device = grid.device
        elif isinstance(grid, np.ndarray):
            np_grid = grid
            is_torch = False
        else:
            raise TypeError("Input must be either a PyTorch tensor or a NumPy array")

        # Create a binary mask of non-zero elements
        binary_mask = (np_grid != 0).astype(int)
        
        # Define the structure for labeling (including diagonals)
        structure = np.ones((3, 3), dtype=int)
        
        # Use scipy's label function to identify connected components
        labeled_array, num_features = label(binary_mask, structure=structure)
        
        # Create the output grid
        center_grid = np.zeros_like(np_grid)
        
        for i in range(1, num_features + 1):
            object_mask = (labeled_array == i)
            
            # Find the center of mass of the object
            center = center_of_mass(object_mask)
            
            # Get the object's bounding box
            rows, cols = np.where(object_mask)
            top, bottom, left, right = rows.min(), rows.max(), cols.min(), cols.max()
            height, width = bottom - top + 1, right - left + 1
            
            # Determine the central cell(s)
            if height % 2 == 1 and width % 2 == 1:
                # Odd x Odd: single central cell
                center_y, center_x = int(center[0]), int(center[1])
                center_grid[center_y, center_x] = np_grid[center_y, center_x]
            elif height % 2 == 0 and width % 2 == 0:
                # Even x Even: 2x2 central region
                center_y, center_x = int(center[0] - 0.5), int(center[1] - 0.5)
                center_grid[center_y:center_y+2, center_x:center_x+2] = np_grid[center_y:center_y+2, center_x:center_x+2]
            elif height % 2 == 0:
                # Even x Odd: 2x1 central region
                center_y, center_x = int(center[0] - 0.5), int(center[1])
                center_grid[center_y:center_y+2, center_x] = np_grid[center_y:center_y+2, center_x]
            else:
                # Odd x Even: 1x2 central region
                center_y, center_x = int(center[0]), int(center[1] - 0.5)
                center_grid[center_y, center_x:center_x+2] = np_grid[center_y, center_x:center_x+2]
        
        # Convert back to PyTorch tensor if the input was a tensor
        if is_torch:
            return torch.tensor(center_grid, dtype=grid.dtype, device=device)
        else:
            return center_grid

    return [[process_single_grid(grid) for grid in grid_list] for grid_list in grid_lists]


def denoise(grid_lists):
    """
    Applies a 3x3 denoising filter to each grid. The central pixel of each 3x3 window
    is replaced with the most common non-zero color in the surrounding cells within the grid.
    If the majority of surrounding cells are 0, the central cell becomes 0.
    Border pixels are handled as a special case.

    Args:
    grid_lists (list of lists of torch.Tensor): The input grid lists.

    Returns:
    list of lists of torch.Tensor: Denoised grids.
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

        height, width = grid.shape
        result = np.copy(grid)

        def get_surrounding_cells(i, j):
            surrounding = []
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < height and 0 <= nj < width:
                        surrounding.append(grid[ni, nj])
            return surrounding

        for i in range(height):
            for j in range(width):
                surrounding_cells = get_surrounding_cells(i, j)
                
                zero_count = sum(1 for cell in surrounding_cells if cell == 0)
                non_zero_count = len(surrounding_cells) - zero_count

                if zero_count > non_zero_count:
                    result[i, j] = 0
                elif non_zero_count > 0:
                    non_zero_surrounding = [cell for cell in surrounding_cells if cell != 0]
                    mode = stats.mode(non_zero_surrounding, keepdims=True)[0][0]
                    result[i, j] = mode
                # If all surrounding cells are zero or it's a tie, the central cell remains unchanged

        if is_torch:
            return torch.from_numpy(result).to(device)
        return result

    return [[process_grid(grid) for grid in grid_list] for grid_list in grid_lists]


def paint_with_palette(grid_lists):
    """
    Paints objects in grid0 using colors from grid1 as a palette.
    Objects are painted in order from left to right first, then top to bottom.
    Objects connected only diagonally are considered separate.
    If there's any problem with any grid list, the function returns the original grid lists.

    Args:
    grid_lists (list of lists of torch.Tensor): The input grid lists.

    Returns:
    list of lists of torch.Tensor: Modified grid lists with objects in grid0 painted using the palette from grid1,
    or original grid lists if there's any problem.
    """
    def process_grid_pair(grid0, grid1):
        if isinstance(grid0, torch.Tensor):
            grid0 = grid0.cpu().numpy()
            grid1 = grid1.cpu().numpy()
            is_torch = True
        else:
            is_torch = False

        # Find objects in grid0
        structure = np.array([[0, 1, 0],
                              [1, 1, 1],
                              [0, 1, 0]])  # 4-connectivity
        labeled, num_objects = label(grid0 != 0, structure=structure)

        # Get palette colors from grid1
        palette = np.unique(grid1[grid1 != 0])

        # If there are no colors in the palette, return None to signal a problem
        if len(palette) == 0:
            return None

        # If there are fewer colors than objects, cycle the palette
        if len(palette) < num_objects:
            palette = np.tile(palette, (num_objects // len(palette)) + 1)[:num_objects]

        # Sort objects by their leftmost point, then by their topmost point
        object_positions = []
        for i in range(1, num_objects + 1):
            pos = np.argwhere(labeled == i)
            left = pos[:, 1].min()
            top = pos[:, 0].min()
            object_positions.append((i, left, top))
        object_positions.sort(key=lambda x: (x[1], x[2]))  # Sort by left, then top

        # Paint objects
        result = np.zeros_like(grid0)
        for (obj_id, _, _), color in zip(object_positions, palette):
            result[labeled == obj_id] = color

        if is_torch:
            return torch.from_numpy(result)
        return result

    # Try to process all grid lists
    try:
        processed_grid_lists = []
        for grid_list in grid_lists:
            if len(grid_list) < 2:
                return grid_lists  # Return original if any list has fewer than 2 grids
            processed_pair = process_grid_pair(grid_list[0], grid_list[1])
            if processed_pair is None:
                return grid_lists  # Return original if any pair couldn't be processed
            processed_grid_lists.append([processed_pair] + grid_list[2:])
        return processed_grid_lists
    except Exception as e:
        print(f"Error in paint_with_palette: {e}. Returning original grid lists.")
        return grid_lists
