from flask import Flask, session, request, jsonify, abort
from uuid import uuid4

import sys
import os
import torch
import config

from app.api.env import utils, env

# Example usage:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

arc_path = utils.extract_arc_from_local()  # Make sure this function is defined
training_tasks = utils.load_tasks(os.path.join(arc_path, "training"))
evaluation_tasks = utils.load_tasks(os.path.join(arc_path, "evaluation"))
all_tasks = {**training_tasks, **evaluation_tasks}

# Create the dataset
arc_dataset = utils.ARCDataset(all_tasks)
# env = env.GridTransformationEnv(arc_dataset)  # Initialize your environment


# ------------------------------------------------------------------------------
def get_user_env():
    # Check if the user has an assigned session ID
    if "session_id" not in session:
        session["session_id"] = str(
            uuid4()
        )  # Generate a unique ID and store in session
    session_id = session["session_id"]

    # Check if the user already has an `env` instance
    if session_id not in user_envs:
        user_envs[session_id] = env.GridTransformationEnv(arc_dataset)  # init the env

    return user_envs[session_id]


# ------------------------------------------------------------------------------

app = Flask(__name__)
app.config.from_object(config)  # Load config from the config.py file
app.secret_key = app.config["SECRET_KEY"]
user_envs = {}  # Dictionary to store env objects per user session

# ------------------------------------------------------------------------------


def abort_400_if_unknown_task(task_id):
    if task_id is None or task_id not in list(training_tasks.keys()):
        abort(400, description=f"Unknown task id: {task_id}")


@app.route("/primitives", methods=["GET"])
def primitives():
    return {"primitives": get_user_env().primitives_names}


@app.route("/task_ids", methods=["GET"])
def get_task_ids():
    return list(training_tasks.keys()) + list(evaluation_tasks.keys())


@app.route("/dataset/training/<task_id>", methods=["GET"])
def get_training_task(task_id):
    abort_400_if_unknown_task(task_id)
    return training_tasks[task_id]


@app.route("/dataset/evaluation/<task_id>", methods=["GET"])
def get_evaluation_task(task_id):
    abort_400_if_unknown_task(task_id)
    return evaluation_tasks[task_id]


@app.route("/closest_tasks", methods=["GET"])
def get_closest_tasks():
    return utils.find_closest_tasks(get_user_env(), "48f8583b")


@app.route("/demonstration/reset/<task_id>", methods=["GET"])
def reset(task_id):
    abort_400_if_unknown_task(task_id)
    res = get_user_env().reset_without_intuition(task_id)
    res["train"] = training_tasks[task_id]["train"]
    res["test"] = training_tasks[task_id]["test"]
    return res


@app.route("/demonstration/step/<task_id>", methods=["GET"])
def demo_step(task_id):
    abort_400_if_unknown_task(task_id)
    return utils.step_demonstration(get_user_env(), task_id)


@app.route("/demonstration/step-all/<task_id>", methods=["GET"])
def demo_step_all(task_id):
    abort_400_if_unknown_task(task_id)
    return utils.step_all_demonstration(get_user_env(), task_id)


@app.route("/demonstration/set-new-list/<task_id>", methods=["POST"])
def set_new_demo_list(task_id):
    abort_400_if_unknown_task(task_id)
    data = request.get_json()
    return utils.set_new_demo_list(get_user_env(), task_id, data)


if __name__ == "__main__":
    app.run()
