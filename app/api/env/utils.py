import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import os
import json
import requests
import zipfile
from io import BytesIO
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.cuda.amp as amp


def extract_arc_from_local():
    # Get the directory where the current script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    zip_file_path = os.path.join(current_dir, "ARC-AGI-master.zip")
    extract_path = os.path.join(
        current_dir, "ARC-AGI-master", "data"
    )  # Directory for extracted data

    print(f"Looking for ZIP file in: {zip_file_path}")

    # Check if ZIP file exists
    if not os.path.isfile(zip_file_path):
        raise FileNotFoundError(f"ZIP file not found: {zip_file_path}")
    # Check if data is already extracted
    if os.path.isdir(extract_path):
        print(f"Data already extracted at: {extract_path}")
        return extract_path

    # Extract the data if it's not already present
    print(f"Extracting dataset from {zip_file_path}...")
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(current_dir)
    print("Extraction complete.")
    return extract_path


# Download and extract the ARC dataset
def download_and_extract_arc():
    arc_url = "https://github.com/fchollet/ARC/archive/master.zip"
    print("Downloading ARC dataset...")
    response = requests.get(arc_url)
    zip_file = zipfile.ZipFile(BytesIO(response.content))
    print("Extracting dataset...")
    zip_file.extractall("/content/")
    return "/content/ARC-AGI-master/data"


# Function to load ARC tasks
def load_tasks(directory):
    tasks = {}
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            with open(os.path.join(directory, filename), "r") as f:
                task = json.load(f)
                tasks[filename[:-5]] = task
    return tasks


class ARCDataset(Dataset):
    def __init__(self, tasks):
        self.tasks = list(tasks.values())
        self.task_ids = list(tasks.keys())

    def __len__(self):
        return len(self.tasks)

    def getItemWeb(self, idx):
        task = self.tasks[idx]
        task_id = self.task_ids[idx]

        train_inputs = [example["input"] for example in task["train"]]
        train_outputs = [example["output"] for example in task["train"]]
        test_inputs = [example["input"] for example in task["test"]]
        test_outputs = [example["output"] for example in task["test"]]

        return {
            "task_id": task_id,
            "train": {"inputs": train_inputs, "outputs": train_outputs},
            "test": {"inputs": test_inputs, "outputs": test_outputs},
        }

    def __getitem__(self, idx):
        task = self.tasks[idx]
        task_id = self.task_ids[idx]

        train_inputs = [
            torch.tensor(example["input"], dtype=torch.long)
            for example in task["train"]
        ]
        train_outputs = [
            torch.tensor(example["output"], dtype=torch.long)
            for example in task["train"]
        ]
        test_inputs = [
            torch.tensor(example["input"], dtype=torch.long) for example in task["test"]
        ]
        test_outputs = [
            torch.tensor(example["output"], dtype=torch.long)
            for example in task["test"]
        ]

        return {
            "task_id": task_id,
            "train": {"inputs": train_inputs, "outputs": train_outputs},
            "test": {"inputs": test_inputs, "outputs": test_outputs},
        }


def visualize_task(task):
    num_train = len(task["train"]["inputs"])
    num_test = len(task["test"]["inputs"])
    total_examples = num_train + num_test

    fig, axes = plt.subplots(total_examples, 2, figsize=(10, 5 * total_examples))
    if total_examples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_train):
        axes[i, 0].imshow(task["train"]["inputs"][i], cmap="tab20")
        axes[i, 0].set_title(f"Train Input {i+1}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(task["train"]["outputs"][i], cmap="tab20")
        axes[i, 1].set_title(f"Train Output {i+1}")
        axes[i, 1].axis("off")

    for i in range(num_test):
        axes[i + num_train, 0].imshow(task["test"]["inputs"][i], cmap="tab20")
        axes[i + num_train, 0].set_title(f"Test Input {i+1}")
        axes[i + num_train, 0].axis("off")

        axes[i + num_train, 1].imshow(task["test"]["outputs"][i], cmap="tab20")
        axes[i + num_train, 1].set_title(f"Test Output {i+1}")
        axes[i + num_train, 1].axis("off")

    plt.tight_layout()
    plt.show()


def load_tasks(directory):
    tasks = {}
    filenames = sorted(os.listdir(directory))
    for filename in filenames:
        if filename.endswith(".json"):
            with open(os.path.join(directory, filename), "r") as f:
                task = json.load(f)
                tasks[filename[:-5]] = task
    return tasks


def visualize_grids(env, img_index, action_name):
    cmap = colors.ListedColormap(
        [
            "#000000",
            "#0074D9",
            "#FF4136",
            "#2ECC40",
            "#FFDC00",
            "#AAAAAA",
            "#F012BE",
            "#FF851B",
            "#7FDBFF",
            "#870C25",
        ]
    )
    norm = colors.Normalize(vmin=0, vmax=9)

    num_inputs = len(env.current_grids)
    num_memory_slots = sum(1 for m in env.memory if m is not None)
    num_memory_grids = sum(len(m) for m in env.memory if m is not None)
    total_grids = num_inputs + num_memory_grids

    fig, axs = plt.subplots(
        1, max(1, total_grids), figsize=(5 * max(1, total_grids), 5)
    )
    if total_grids == 0:
        plt.text(
            0.5,
            0.5,
            "No grids to display",
            horizontalalignment="center",
            verticalalignment="center",
        )
        plt.axis("off")
    else:
        if total_grids == 1:
            axs = [axs]

        # Visualize current grids
        for i, grid_list in enumerate(env.current_grids):
            if grid_list:
                axs[i].imshow(grid_list[0], cmap=cmap, norm=norm)
                axs[i].set_title(f"Input {i+1} after action :'{action_name}'")
            else:
                axs[i].axis("off")
            axs[i].set_xticks([])
            axs[i].set_yticks([])

        # Visualize memory grids
        memory_index = num_inputs
        for slot, memory_grids in enumerate(env.memory):
            if memory_grids is not None:
                for i, grid in enumerate(memory_grids):
                    if memory_index < len(axs):
                        axs[memory_index].imshow(grid, cmap=cmap, norm=norm)
                        axs[memory_index].set_title(f"Memory {slot+1}.{i+1}")
                        axs[memory_index].set_xticks([])
                        axs[memory_index].set_yticks([])
                        memory_index += 1
            elif memory_index < len(axs):
                axs[memory_index].axis("off")
                memory_index += 1

    plt.tight_layout()

    fig.savefig(
        os.path.join(
            "output", "fig_" + str(img_index) + "_" + str(action_name) + ".png"
        )
    )  # for local dev
    # plt.show() # for Collab


# The rest of your code (GridTransformationEnv class, etc.) remains the same


def print_action_list(env):
    print("List of Actions:")
    for i, action in enumerate(env.primitives):
        print(f"{i}: {action.__name__}")
    print()


def get_action_name(env, action_index):
    return env.primitives[action_index].__name__


def find_action_location(env, action_index):
    action = env.primitives[action_index]

    # Get the source file
    try:
        file_path = inspect.getfile(action)
    except TypeError:
        # This can happen for built-in functions
        return (
            f"Action {action.__name__} is a built-in function and has no file location."
        )

    # Get the line number
    try:
        line_number = inspect.getsourcelines(action)[1]
    except OSError:
        line_number = "unknown"

    # Get relative path if it's in a subdirectory of the current working directory
    cwd = os.getcwd()
    if file_path.startswith(cwd):
        file_path = os.path.relpath(file_path, cwd)

    return (
        f"Action {action.__name__} is defined in file: {file_path}, line: {line_number}"
    )


def visualize_demonstration(env, task_id):
    if task_id not in env.demonstrations:
        print(f"No demonstration found for task: {task_id}")
        return

    # Reset the environment with the specific task
    obs = env.reset(task_id)
    print(f"Demonstrating task: {task_id}")
    print("Initial state:")
    visualize_grids(env, 0, "INIT")

    total_reward = 0

    # Iterate through each action in the demonstration
    for step, action_index in enumerate(env.current_demonstration):
        if 0 <= action_index < len(env.primitives_names):
            action_name = env.primitives_names[action_index]

            print(f"Step {step + 1}: Action {action_index} - {action_name}")

            next_obs, reward, done, _ = env.step(action_index)
            total_reward += reward

            print(f"Reward: {reward}")
            print(f"Done: {done}")
            print("State after action:")
            visualize_grids(env, step, action_name)
            print("--------------------")

            if done:
                break
        else:
            print(f"Warning: Invalid action index {action_index} for task {task_id}")

    print(f"Demonstration finished. Total reward: {total_reward}")
    print(f"Final action sequence: {env.action_sequence}")


# Usage
# env = GridTransformationEnv(arc_dataset)  # Initialize your environment
# task_id_to_visualize = "28e73c20"  # Replace with the task ID you want to visualize
# visualize_demonstration(env, task_id_to_visualize)


def tensors_to_json(tensor_list):
    """
    Convert Tensor list to JSON objects
    """
    # Convert each tensor to a Python list
    json_serializable_list = [tensor.tolist() for tensor in tensor_list]
    # Convert to JSON format (string) or return the list directly for JSON serialization
    return json.dumps(json_serializable_list)


def memory_grids_to_json(memory_grids):
    """
    Converts a list of memory grids, where each item can be a sublist of tensors or None, into a JSON-compatible format.
    If an item is None, it is replaced with an empty list in the returned JSON structure.

    Args:
    memory_grids (list): A list containing None or sublists of torch.Tensor objects.

    Returns:
    str: JSON-formatted string representing the memory grids.
    """
    json_compatible_grids = []

    for grid in memory_grids:
        if grid is None:
            json_compatible_grids.append([])  # Replace None with an empty list
        else:
            # Convert each tensor in the sublist to a list format
            tensor_list = [tensor.tolist() for tensor in grid]
            json_compatible_grids.append(tensor_list)

    return json.dumps(json_compatible_grids)


def set_new_demo_list(env, task_id, data):
    obs = env.reset(task_id)

    new_demonstration_list = [
        env.primitives_names.index(action)
        for action in data
        if action in env.primitives_names
    ]

    env.current_demonstration = new_demonstration_list

    return {
        "current_demonstration": env.current_demonstration,  # [50, 43, 88, 76, 60, 85, 96, 83, 60]
        "primitives_names": env.primitives_names,
        "data": data,
        "NEW_current_demonstration": new_demonstration_list,
    }


def step_all_demonstration(env, task_id):
    if task_id not in env.demonstrations:
        print(f"No demonstration found for task: {task_id}")
        return

    results = []
    while env.current_step < len(env.current_demonstration):

        action_index = env.current_demonstration[env.current_step]
        env.step_without_intuition(action_index)
        action_name = env.primitives_names[action_index]
        grids = [tensors_to_json(gridList) for gridList in env.current_grids]
        mem_grids = memory_grids_to_json(env.memory)

        results.append(
            {
                "step": env.demonstration_step,
                "max_steps": len(env.current_demonstration),
                "action_name": action_name,
                "current_grids": grids,
                "memory_grids": mem_grids,
            }
        )

    return results


def step_demonstration(env, task_id):
    if task_id not in env.demonstrations:
        print(f"No demonstration found for task: {task_id}")
        return

    if env.current_step >= len(env.current_demonstration):
        return "Reached end of demo", 400

    action_index = env.current_demonstration[env.current_step]
    env.step_without_intuition(action_index)
    action_name = env.primitives_names[action_index]
    grids = [tensors_to_json(gridList) for gridList in env.current_grids]

    return {
        "step": env.demonstration_step,
        "max_steps": len(env.current_demonstration),
        "action_name": action_name,
        "current_grids": grids,
    }


# def find_closest_tasks(env, task_id, distance_matrix, n=5):
def find_closest_tasks(env, task_id, n=5):
    all_task_ids = env.arc_dataset.task_ids

    current_dir = os.path.dirname(os.path.abspath(__file__))
    distance_matrix_path = os.path.join(current_dir, "arc_distance_matrix_10.npy")
    print(f"Looking for distance_matrix in: {distance_matrix_path}")

    distance_matrix = np.load(distance_matrix_path)

    task_index = all_task_ids.index(task_id)
    distances = distance_matrix[task_index]
    closest_indices = np.argsort(distances)[
        1 : n + 1
    ]  # Exclude the task itself (index 0)
    closest_tasks = [all_task_ids[i] for i in closest_indices]
    closest_distances = distances[closest_indices]
    return list(zip(closest_tasks, closest_distances))
