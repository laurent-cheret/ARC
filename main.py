
from dsl import *
from env import *


# Example usage:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

arc_path = download_and_extract_arc()  # Make sure this function is defined
training_tasks = load_tasks(os.path.join(arc_path, 'training'))
evaluation_tasks = load_tasks(os.path.join(arc_path, 'evaluation'))
all_tasks = {**training_tasks, **evaluation_tasks}

# Create the dataset
arc_dataset = ARCDataset(all_tasks)

env = GridTransformationEnv(arc_dataset)  # Initialize your environment
task_id_to_visualize = "28e73c20"  # Replace with the task ID you want to visualize
# task_id_to_visualize = ""  # Replace with the task ID you want to visualize

visualize_task(arc_dataset[0])
# visualize_demonstration(env, task_id_to_visualize)
