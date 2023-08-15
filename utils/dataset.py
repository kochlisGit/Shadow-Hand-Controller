import csv
import numpy as np

ONE_HOT_SIGNS = {
    'rest': np.float32([0, 0, 0, 0, 0, 0, 1]),
    'drop': np.float32([0, 0, 0, 0, 0, 1, 0]),
    'middle finger': np.float32([0, 0, 0, 0, 1, 0, 0]),
    'yes': np.float32([0, 0, 0, 1, 0, 0, 0]),
    'no': np.float32([0, 0, 1, 0, 0, 0, 0]),
    'rock': np.float32([0, 1, 0, 0, 0, 0, 0]),
    'circle': np.float32([1, 0, 0, 0, 0, 0, 0])
}
ONE_HOT_ORDERS = {
    0: np.float32([0, 0, 1]),
    1: np.float32([0, 1, 0]),
    2: np.float32([1, 0, 0])
}
NUM_ACTUATORS = 20


# Generates expert dataset by saving expert transitions [sign, order, ctrl]
def generate_expert_dataset(transition_history: list[str, int, np.ndarray], dataset_filepath: str):
    ctrl_features = [f'ctrl_{i + 1}' for i in range(NUM_ACTUATORS)]
    dataset_features = ['sign', 'order'] + ctrl_features

    with open(dataset_filepath, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(dataset_features)

        for transition in transition_history:
            sign, order = transition[0: 2]
            ctrl = transition[2].tolist()
            row = [sign, order] + ctrl
            writer.writerow(row)


# Reads dataset x, y with x --> (sign, order) y --> (ctrl)
def read_dataset(dataset_filepath: str, one_hot: bool) -> (dict[str, np.ndarray], np.ndarray):
    signs = []
    orders = []
    ctrls = []

    with open(dataset_filepath, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)

        for row in reader:
            sign = row[0]
            order = int(row[1])

            if one_hot:
                sign = ONE_HOT_SIGNS[sign]
                order = ONE_HOT_ORDERS[order]

            ctrl = [float(pos) for pos in row[2:]]

            signs.append(sign)
            orders.append(order)
            ctrls.append(ctrl)
    x = {'sign': np.array(signs), 'order': np.float32(orders)}
    y = np.float32(ctrls)
    return x, y
