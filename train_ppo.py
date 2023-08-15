import random
import matplotlib.pyplot as plt
import numpy as np
import ray
import tensorflow as tf
from models.rllib.ppo import PPOAgent
from utils import control, dataset


dataset_filepath = 'data/expert_dataset.csv'
ctrl_limits_filepath = 'data/ctrl_limits.csv'
checkpoint_directory = 'checkpoints/ppo'
one_hot_signs = True
batch_size = 8
epochs = 30
train_iterations = 1000
seed = 0

summary = True
print_test_predictions = True
plot_performance = True


def main():
    ray.shutdown()
    ray.init()

    tf.random.set_seed(seed=seed)
    np.random.seed(seed=seed)
    random.seed(seed)

    x, y = dataset.read_dataset(dataset_filepath=dataset_filepath, one_hot=one_hot_signs)

    print('Train Dataset:', y.shape, x['sign'].shape, x['order'].shape)

    episode_steps = y.shape[0] - 1
    ctrl_limits = control.read_ctrl_limits(csv_filepath=ctrl_limits_filepath)
    env_config = {'x': x, 'y': y, 'ctrl_limits': ctrl_limits}

    agent = PPOAgent(
        env_config=env_config,
        episode_steps=episode_steps,
        batch_size=batch_size,
        epochs=epochs,
        train_iterations=train_iterations,
        seed=seed
    )
    agent.build(summary=summary)
    cumulative_rewards_per_iteration = agent.train(x=x, y=y)
    agent.save(checkpoint_directory=checkpoint_directory)

    if print_test_predictions:
        x_test_signs = x['sign']
        x_test_orders = x['order']

        for i in range(y.shape[0]):
            y_pred = agent.predict_next_control(sign_vector=x_test_signs[i], order_vector=x_test_orders[i])
            print(f'i: {i + 1}, y_pred: {y_pred}\tactual: {y[i]}')

    if plot_performance:
        plt.plot(cumulative_rewards_per_iteration, label='Cumulative Rewards per Iteration')
        plt.title('PPO Performance')
        plt.xlabel('Iterations')
        plt.ylabel('Cumulative Rewards')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    main()
