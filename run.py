from flask import Flask, request, jsonify
import sys
import os
import torch

# import app.api
from app.api.env import utils, env

# Example usage:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

arc_path = utils.extract_arc_from_local()  # Make sure this function is defined
training_tasks = utils.load_tasks(os.path.join(arc_path, "training"))
evaluation_tasks = utils.load_tasks(os.path.join(arc_path, "evaluation"))
all_tasks = {**training_tasks, **evaluation_tasks}

# # Create the dataset
arc_dataset = utils.ARCDataset(all_tasks)
env = env.GridTransformationEnv(arc_dataset)  # Initialize your environment

# ------------------------------------------------------------------------------

app = Flask(__name__)


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route("/task_ids", methods=["GET"])
def get_task_ids():
    return list(training_tasks.keys()) + list(evaluation_tasks.keys())
    # return env.arc_dataset.task_ids


@app.route("/closest_tasks", methods=["GET"])
def get_closest_tasks():
    return utils.find_closest_tasks(env, "48f8583b")


@app.route("/dataset/training/<task_id>", methods=["GET"])
def get_training_task(task_id):
    return training_tasks[task_id]


@app.route("/dataset/evaluation/<task_id>", methods=["GET"])
def get_evaluation_task(task_id):
    return evaluation_tasks[task_id]


@app.route("/demonstration/<task_id>", methods=["GET"])
def demonstration(task_id):
    return utils.step_demonstration(env, task_id)


@app.route("/demonstration/reset/<task_id>", methods=["GET"])
def reset(task_id):
    env.reset_without_intuition(task_id)
    return f"Reset to {task_id}"


@app.route("/demonstration/step/<task_id>", methods=["GET"])
def demo_step(task_id):
    return utils.step_demonstration(env, task_id)


@app.route("/demonstration/set-new-list/<task_id>", methods=["POST"])
def set_new_demo_list(task_id):
    data = request.get_json()
    return utils.set_new_demo_list(env, task_id, data)


# @app.route("/demonstration/step-task", methods=['GET'])
# def ddd(task_id):
#   step env
#   send grids
#   send mem grids


@app.route("/primitives", methods=["GET"])
def test_model():
    return {"primitives": env.primitives_names}
