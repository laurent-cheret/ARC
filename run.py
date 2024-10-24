from flask import Flask
import sys
import os
import torch

# import app.api
from app.api.env import utils, env

# Example usage:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

arc_path = utils.extract_arc_from_local()  # Make sure this function is defined
training_tasks = utils.load_tasks(os.path.join(arc_path, 'training'))
evaluation_tasks = utils.load_tasks(os.path.join(arc_path, 'evaluation'))
all_tasks = {**training_tasks, **evaluation_tasks}

# # Create the dataset
arc_dataset =  utils.ARCDataset(all_tasks)
env = env.GridTransformationEnv(arc_dataset)  # Initialize your environment
 
# ------------------------------------------------------------------------------

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/primitives", methods=['GET'])
def test_model():
    return {
      "primitives": env.primitives_names
    }

@app.route("/dataset/<int:task_id>", methods=['GET'])
def get_task(task_id):
    return arc_dataset.getItemWeb(task_id)

@app.route("/test", methods=['GET'])
def jsonapi():
    return {
      "one": 1,
      "two": 2,
    }




