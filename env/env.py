import gym
from gym import spaces
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import random
import math
import sys
import os
import importlib
import inspect
from scipy.ndimage import label
import logging
import json


# Add the directory to Python's sys.path
# dsl_path = '/content/drive/MyDrive/ARC_CHALLENGE'
dsl_path = ''
if dsl_path not in sys.path:
    sys.path.append(dsl_path)

# Import primitive functions
# from dsl.basic_transformations import *
# from dsl.memory_operations import *
# from dsl.color_operations import *
# from dsl.critical_operations import *
# from dsl.abstract_operations import *


class ARCDataset(Dataset):
    def __init__(self, tasks):
        self.tasks = list(tasks.values())
        self.task_ids = list(tasks.keys())

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        task = self.tasks[idx]
        task_id = self.task_ids[idx]

        train_inputs = [torch.tensor(example['input'], dtype=torch.long) for example in task['train']]
        train_outputs = [torch.tensor(example['output'], dtype=torch.long) for example in task['train']]
        test_inputs = [torch.tensor(example['input'], dtype=torch.long) for example in task['test']]
        test_outputs = [torch.tensor(example['output'], dtype=torch.long) for example in task['test']]

        return {
            'task_id': task_id,
            'train': {
                'inputs': train_inputs,
                'outputs': train_outputs
            },
            'test': {
                'inputs': test_inputs,
                'outputs': test_outputs
            }
        }

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, dropout=0.0):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src):
        src = self.embedding(src)
        src = src.unsqueeze(0)  # Add sequence dimension
        output = self.transformer_encoder(src)
        output = self.norm(output)
        return output.squeeze(0)  # Remove sequence dimension

class DeepAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=128, d_model=1024, nhead=16, num_layers=3, dim_feedforward=1024):
        super(DeepAutoencoder, self).__init__()

        self.encoder = TransformerEncoder(input_dim, d_model, nhead, num_layers, dim_feedforward)

        self.decoder = nn.Sequential(
            self._block(d_model, 1024, dropout=0.0),
            self._block(1024, 10000, dropout=0.0),
            nn.Linear(10000, input_dim),
            nn.Sigmoid()
        )

    def _block(self, in_features, out_features, dropout=0.0):
        return nn.Sequential(
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, out_features),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        latent = self.encoder(x)
        decoded = self.decoder(latent)
        return decoded, latent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class GridTransformationEnv(gym.Env):
    def __init__(self, arc_dataset, memory_capacity=5, max_steps=50, demo_file='env\demonstrations.json'):
        super(GridTransformationEnv, self).__init__()

        self.arc_dataset = arc_dataset
        self.memory_capacity = memory_capacity
        self.memory = [None] * memory_capacity
        self.current_task_id = 'Nothing'

        # Load the autoencoder
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.autoencoder = DeepAutoencoder(input_dim=9000, d_model=256)  # Adjust parameters as needed
        self.autoencoder.load_state_dict(torch.load('intuition_models\deep_arc_autoencoder_256.pth', map_location=self.device))
        self.autoencoder.to(self.device)
        self.autoencoder.eval()
        self.max_steps = max_steps
        self.action_sequence = []

        self.encoding_dim = 256  # d_model value
        self.memory_state = np.zeros((memory_capacity, self.encoding_dim), dtype=np.float32)

        # Load all primitives
        self.primitives, self.primitives_names = self.load_all_primitives()

        # Load demonstrations
        self.demonstrations = self.load_demonstrations(demo_file)
        self.using_demonstration = False
        self.current_demonstration = []
        self.demonstration_step = 0

        # Action space: discrete, with the number of primitives
        self.action_space = spaces.Discrete(len(self.primitives))

        # Observation space: Dict space with current objective, task information, quantity error, color usage, and memory state
        self.observation_space = spaces.Dict({
            'current_objective': spaces.Box(low=-np.inf, high=np.inf, shape=(self.encoding_dim,), dtype=np.float32),
            'current_step': spaces.Discrete(max_steps),  # Assuming a maximum of 100 steps per episode
            'num_train_examples': spaces.Discrete(11),  # Assuming a maximum of 10 training examples
            'quantity_error': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'original_colors': spaces.Box(low=0, high=1, shape=(10,), dtype=np.int32),
            'current_colors': spaces.Box(low=0, high=1, shape=(10,), dtype=np.int32),
            'memory_state': spaces.Box(low=-np.inf, high=np.inf, shape=(memory_capacity, self.encoding_dim), dtype=np.float32),
            'avg_width_difference': spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            'avg_height_difference': spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            'using_demonstration': spaces.Discrete(2),
            'demonstration_action': spaces.Discrete(len(self.primitives))
        })

    def load_demonstrations(self, file_path):
        with open(file_path, 'r') as f:
            demos = json.load(f)
        return demos

    def grid_to_tensor(self, grid):
        h, w = len(grid), len(grid[0])
        one_hot = np.eye(10)[np.array(grid)]
        padded_one_hot = np.zeros((30, 30, 10), dtype=float)
        padded_one_hot[:h, :w, :] = one_hot
        return torch.FloatTensor(padded_one_hot)

    def encode_grid(self, grid):
        if grid is None or len(grid) == 0 or len(grid[0]) == 0:
            return np.zeros(self.encoding_dim, dtype=np.float32)

        grid_tensor = self.grid_to_tensor(grid)
        with torch.no_grad():
            grid_tensor = grid_tensor.view(1, -1).to(self.device)  # Reshape to (1, 9000)
            _, latent = self.autoencoder(grid_tensor)
        return latent.cpu().numpy().flatten()

    def _log_current_grids_state(self):
        logger.info("Current state of self.current_grids:")
        for i, grid_list in enumerate(self.current_grids):
            logger.info(f"Grid list {i}:")
            for j, grid in enumerate(grid_list):
                logger.info(f"  Grid {j} shape: {grid.shape}")
                logger.info(f"  Grid {j} unique values: {torch.unique(grid)}")

    def load_all_primitives(self):
        primitives = []
        primitive_names = []
        dsl_dir = os.path.join(dsl_path, 'dsl')

        # First, collect all primitives and their names
        for filename in os.listdir(dsl_dir):
            if filename.endswith('.py'):
                module_name = f'dsl.{filename[:-3]}'
                module = importlib.import_module(module_name)
                for name, obj in inspect.getmembers(module):
                    if inspect.isfunction(obj):
                        # Check if the function is defined in the current module
                        if inspect.getmodule(obj) == module:
                            if name.startswith('_'):  # Skip private functions
                                continue
                            if 'env' in inspect.signature(obj).parameters:
                                # If the function expects an 'env' parameter, wrap it
                                wrapped_func = lambda x, f=obj: f(self, x)
                                wrapped_func.__name__ = obj.__name__  # Preserve the original name
                                primitives.append(wrapped_func)
                            else:
                                primitives.append(obj)
                            primitive_names.append(name)

        # Now, sort both lists based on the primitive names
        sorted_pairs = sorted(zip(primitive_names, primitives), key=lambda x: x[0])

        # Unzip the sorted pairs
        primitive_names, primitives = zip(*sorted_pairs)

        return list(primitives), list(primitive_names)

    def reset(self, task_id=None):
        self.action_sequence = []

        if task_id is not None and task_id in self.arc_dataset.task_ids:
            # Find the index corresponding to the specified task_id
            task_idx = self.arc_dataset.task_ids.index(task_id)
            self.current_task = self.arc_dataset[task_idx]
            self.current_task_id = task_id
        else:
            # Select a random task from the dataset
            task_idx = random.randint(0, len(self.arc_dataset) - 1)
            self.current_task = self.arc_dataset[task_idx]
            self.current_task_id = self.arc_dataset.task_ids[task_idx]

        # # Check if there's a demonstration for this task
        self.using_demonstration = self.current_task_id in self.demonstrations
        self.current_demonstration = self.demonstrations.get(self.current_task_id, [])
        self.demonstration_step = 0

        self.initial_grids = [grid for grid in self.current_task['train']['inputs']]
        self.current_grids = [[grid] for grid in self.initial_grids]
        self.current_step = 0

        # Compute initial encodings
        self.initial_encodings = [self.encode_grid(grid) for grid in self.initial_grids]
        self.output_encodings = [self.encode_grid(grid) for grid in self.current_task['train']['outputs']]

        # Compute general task intuition
        self.general_task_intuition = self._compute_general_task_intuition()

        # Compute original colors
        self.original_colors = self._compute_color_usage(self.initial_grids + self.current_task['train']['outputs'])

        # Clear memory
        self.memory = [None] * self.memory_capacity
        self.memory_state = np.zeros((self.memory_capacity, self.encoding_dim), dtype=np.float32)

        print(f"Now solving task: {self.current_task_id}")
        if self.using_demonstration:
            print(f"Using demonstration with {len(self.current_demonstration)} steps")
        else:
            print("No demonstration available for this task")

        return self._get_observation()

    def step(self, action):
        if self.using_demonstration and self.demonstration_step < len(self.current_demonstration):
            demonstration_action = self.current_demonstration[self.demonstration_step]
            if 0 <= demonstration_action < len(self.primitives):
                action = demonstration_action
            else:
                print(f"Warning: Action index '{demonstration_action}' from demonstration is out of range")
                # Keep the original action if the demonstration action is invalid
            self.demonstration_step += 1

        action_name = self.primitives_names[action]
        self.action_sequence.append(action_name)

        # Apply the chosen primitive function to all current grids
        self.current_grids = self.primitives[action](self.current_grids)

        # Check if all grid lists are empty and memory state is empty
        all_grids_empty = all(len(grid_list) == 0 for grid_list in self.current_grids)
        memory_empty = np.all(self.memory_state == 0)

        if all_grids_empty and memory_empty:
            done = True
            reward = -0.1
        else:
            self.current_step += 1
            # Check if the transformation matches the output
            reward = self._compute_reward(self._get_observation())

            done = self.current_step >= self.max_steps or reward == 10.0
            if reward == 10.0:
                print(f'Solved the task: {self.current_task_id}!!!!!!!!!!!')
                print(self.action_sequence)
            elif reward >= 0.1 and reward <= 1:
                print(f'Solved some of the {self.current_task_id} examples but not all.')

        return self._get_observation(), reward, done, {}

    def _compute_color_usage(self, grids):
        colors = set()
        for grid in grids:
            if isinstance(grid, torch.Tensor):
                colors.update(grid.unique().cpu().numpy())
            elif isinstance(grid, np.ndarray):
                colors.update(np.unique(grid))
            else:
                colors.update(np.unique(grid))
        return np.array([1 if i in colors else 0 for i in range(10)], dtype=np.int32)

    def _get_observation(self):
        current_intuition = self._compute_current_intuition()
        current_objective = self.general_task_intuition - current_intuition
        quantity_error = self._compute_quantity_error()
        current_colors = self._compute_color_usage([grid for sublist in self.current_grids for grid in sublist])
        avg_width_diff, avg_height_diff = self._compute_size_fitness()
        obs = {
            'current_objective': current_objective,
            'current_step': self.current_step,
            'num_train_examples': len(self.current_task['train']['inputs']),
            'quantity_error': np.array([quantity_error], dtype=np.float32),
            'original_colors': self.original_colors,
            'current_colors': current_colors,
            'memory_state': self.memory_state,
            'avg_width_difference': np.array([avg_width_diff], dtype=np.float32),
            'avg_height_difference': np.array([avg_height_diff], dtype=np.float32),
            'using_demonstration': int(self.using_demonstration),
            'demonstration_action': self.current_demonstration[self.demonstration_step] if self.using_demonstration and self.demonstration_step < len(self.current_demonstration) else len(self.primitives) - 1
        }

        logger.info(f"Generated observation: {obs}")
        return obs

    def has_demonstration(self):
        return self.current_task_id in expert_demonstrations

    def get_demonstration_action(self):
        if not self.has_demonstration():
            return None
        actions = expert_demonstrations[self.current_task_id]
        return actions[min(self.current_step, len(actions) - 1)]

    def print_action_names_and_indexes(self):
        for index, name in enumerate(self.primitives_names):
            print(f"Action {index}: {name}")

    def _compute_memory_state_for_grids(self, grids):
        memory_intuitions = []
        for i, initial_encoding in enumerate(self.initial_encodings):
            if i < len(grids) and grids[i] is not None:
                memory_encoding = self.encode_grid(grids[i])
                intuition = memory_encoding - initial_encoding
                memory_intuitions.append(intuition)

        if memory_intuitions:
            avg_memory_intuition = np.mean(memory_intuitions, axis=0)
        else:
            avg_memory_intuition = np.zeros(self.encoding_dim, dtype=np.float32)

        return avg_memory_intuition

    def _compute_memory_state(self):
        return self.memory_state


    def _compute_general_task_intuition(self):
        intuitions = [out - inp for inp, out in zip(self.initial_encodings, self.output_encodings)]
        return np.mean(intuitions, axis=0)

    def _compute_current_intuition(self):
        all_intuitions = []
        for i, initial_encoding in enumerate(self.initial_encodings):
            current_encodings = [self.encode_grid(grid) for grid in self.current_grids[i]]
            intuitions = [current - initial_encoding for current in current_encodings]
            all_intuitions.extend(intuitions)

        # Compute the average intuition
        if all_intuitions:
            average_intuition = np.mean(all_intuitions, axis=0)
        else:
            average_intuition = np.zeros(self.encoding_dim, dtype=np.float32)

        return average_intuition
    def print_grid_info(self):
        print("Current grids information:")
        for i, grid_list in enumerate(self.current_grids):
            print(f"Input {i + 1}:")
            for j, grid in enumerate(grid_list):
                print(f"  Grid {j + 1} shape: {grid.shape}")
                print(f"  Grid {j + 1} unique values: {torch.unique(grid).tolist()}")

        all_current_grids = [grid for sublist in self.current_grids for grid in sublist]
        current_colors = self._compute_color_usage(all_current_grids)
        print("Computed current colors:", current_colors)


    def _compute_quantity_error(self):
        n_total = sum(len(grid_list) for grid_list in self.current_grids)
        n_pairs = len(self.current_grids)
        return 1 - math.exp(-(((n_total - n_pairs)/n_pairs) ** 2))

    # def _compute_reward(self):
    #     correct_matches = 0
    #     total_matches = len(self.current_task['train']['outputs'])

    #     for i, output in enumerate(self.current_task['train']['outputs']):
    #         output_shape = output.shape
    #         matching_grids = [grid for grid in self.current_grids[i] if grid.shape == output_shape]

    #         if any(torch.all(torch.eq(grid, output)) for grid in matching_grids):
    #             correct_matches += 1

    #     return correct_matches / total_matches

    def _compute_reward(self, obs):
        num_train_examples = obs['num_train_examples']

        # Check if any grid list is empty
        if any(len(grid_list) == 0 for grid_list in self.current_grids):
            return 0.0  # Return zero reward if any grid list is empty

        correct_grids = 0
        all_shapes_match = True

        for i in range(num_train_examples):
            output = self.current_task['train']['outputs'][i]
            current_grid = self.current_grids[i][0]  # Take only the first grid from each list

            # Check if shapes match
            if current_grid.shape != output.shape:
                all_shapes_match = False
                break

            # Check if grids are identical
            if torch.all(current_grid == output):
                correct_grids += 1

        # If any shape doesn't match, return zero reward
        if not all_shapes_match:
            return 0.0

        # If all grids are correct, return maximum reward
        if correct_grids == num_train_examples:
            return 10.0

        # If some grids are correct, return proportional reward
        if correct_grids > 0:
            return correct_grids / (num_train_examples * self.current_step) # We impose a discount on the reward, to focus on

        # If shapes match but no grid is completely correct, return small reward
        return 0.01


    def _compute_size_fitness(self):
        width_differences = []
        height_differences = []

        for i, output_grid in enumerate(self.current_task['train']['outputs']):
            if self.current_grids[i]:  # Check if there's a grid at position 0
                current_grid = self.current_grids[i][0]
                width_diff = current_grid.shape[1] - output_grid.shape[1]
                height_diff = current_grid.shape[0] - output_grid.shape[0]
                width_differences.append(width_diff)
                height_differences.append(height_diff)

        avg_width_diff = np.mean(width_differences) if width_differences else 0
        avg_height_diff = np.mean(height_differences) if height_differences else 0

        return avg_width_diff, avg_height_diff

    def load_demonstrations(self, file_path='env\demonstrations.json'):
        with open(file_path, 'r') as f:
            demo_data = json.load(f)

        demonstrations = {}
        for task_id, action_names in demo_data.items():
            valid_actions = []
            for action_name in action_names:
                if action_name in self.primitives_names:
                    action_index = self.primitives_names.index(action_name)
                    valid_actions.append(action_index)
                else:
                    print(f"Warning: Action '{action_name}' not found in primitives for task {task_id}")
            if valid_actions:
                demonstrations[task_id] = valid_actions

        return demonstrations

    def has_demonstration(self):
        return self.current_task_id in self.demonstrations

    def get_demonstration_action(self):
        if not self.has_demonstration():
            return None
        actions = self.demonstrations[self.current_task_id]
        return actions[min(self.current_step, len(actions) - 1)]




class DummyEncoder:
    def __init__(self, encoding_dim=64):
        self.encoding_dim = encoding_dim

    def __call__(self, grid):
        # This is a placeholder encoder. Replace with your actual encoder.
        return torch.randn(self.encoding_dim)

# Example usage:
# Assuming you've already created your ARC dataset
# arc_dataset = ARCDataset(all_tasks)  # all_tasks should be defined earlier

# encoder = DummyEncoder(encoding_dim=64)
# env = GridTransformationEnv(arc_dataset)
# env.print_action_names_and_indexes()

# Run an episode
# obs = env.reset()
# done = False
# total_reward = 0

# step_count = 0
# while not done:
#     action = env.action_space.sample()  # Replace with your action selection logic
#     action_name = get_action_name(env, action)
#     action_location = find_action_location(env, action)
#     print(f"Step {step_count}: Action {action} - {action_name}")
#     print(f"Action location: {action_location}")

#     try:
#         next_obs, reward, done, _ = env.step(action)
#         total_reward += reward
#         obs = next_obs

#         print(f"Reward: {reward}")
#         print(f"Done: {done}")
#         print("--------------------")

#         visualize_grids(env)  # Visualize state after each step
#     except Exception as e:
#         print(f"Error occurred on step {step_count} with action {action_name}:")
#         print(str(e))
#         print(f"Action location: {action_location}")
#         break  # Exit the loop if an error occurs

#     step_count += 1

# print(f"Episode finished. Total reward: {total_reward}")
