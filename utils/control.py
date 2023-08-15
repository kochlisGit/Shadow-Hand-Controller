import csv
import json
import numpy as np


# Reads the control transitions for each sign
def read_sign_transitions(json_filepath: str) -> dict[str, list[np.ndarray]]:
    signs = {}
    with open(json_filepath, mode='r', encoding='utf-8') as jsonfile:
        jsonobj = json.load(jsonfile)

        for sign, ctrl_list in jsonobj.items():
            ctrl_transitions = [np.float32(ctrl) for ctrl in ctrl_list]
            signs[sign] = ctrl_transitions
    return signs


# Reads the control limits (low position, high position) for each actuator
def read_ctrl_limits(csv_filepath: str) -> list[np.ndarray]:
    ctrl_limits = []
    with open(csv_filepath, mode='r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)

        for row in reader:
            left_limit = float(row[1])
            right_limit = float(row[2])
            limit = np.float32([left_limit, right_limit])
            ctrl_limits.append(limit)
    return ctrl_limits


# Generates a control trajectory of N steps between start control and end control
def generate_control_trajectory(start_ctrl: np.ndarray, end_ctrl: np.ndarray, n_steps: int) -> list[np.ndarray]:
    trajectory = []
    for i in range(n_steps + 1):
        ctrl = start_ctrl + i*(end_ctrl - start_ctrl)/n_steps
        trajectory.append(ctrl)
    return trajectory
