import numpy as np
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo import PPOConfig
from environments.shadowhand import ShadowHandEnv
from models.model import Model


class PPOAgent(Model):
    def __init__(
            self,
            env_config: dict,
            episode_steps: int,
            batch_size: int = 8,
            epochs: int = 30,
            train_iterations: int = 100,
            gamma: float = 0.99,
            learning_rate: float = 0.0005,
            seed: int = 0
    ):
        self._config = PPOConfig()
        self._config.train_batch_size = episode_steps
        self._config.env_config = env_config
        self._config.sgd_minibatch_size = batch_size
        self._config.num_sgd_iter = epochs
        self._config.gamma = gamma
        self._config.lr = learning_rate
        self._config.seed = seed

        self._env_config = env_config
        self._train_iterations = train_iterations

        self._agent = None

    def save(self, checkpoint_directory: str):
        assert self._agent is not None

        self._agent.save(checkpoint_dir=checkpoint_directory)

    def load(self, checkpoint_directory: str):
        self._agent = Algorithm.from_checkpoint(checkpoint=checkpoint_directory)

    def build(self, summary: bool):
        self._agent = self._config.build(env=ShadowHandEnv)

        if summary:
            self._agent.get_policy().model.base_model.summary(expand_nested=True)

    def train(self, x: dict[str, np.ndarray], y: np.ndarray) -> list[float]:
        assert self._agent is not None

        test_env = ShadowHandEnv(env_config=self._env_config)
        cumulative_rewards_per_iteration = []

        for i in range(self._train_iterations):
            self._agent.train()

            observation, _ = test_env.reset()
            done = False
            cumulative_rewards = 0.0

            while not done:
                action = self._agent.compute_single_action(observation=observation)
                observation, reward, done, _, _ = test_env.step(action=action)
                cumulative_rewards += reward

            cumulative_rewards_per_iteration.append(cumulative_rewards)

            print(f'Iteration: {i + 1}, Cumulative Rewards: {cumulative_rewards}')

        return cumulative_rewards_per_iteration

    def predict_next_control(self, sign_vector: np.ndarray, order_vector: np.ndarray) -> np.ndarray:
        assert self._agent is not None

        observation = {'sign': sign_vector, 'order': order_vector}
        return self._agent.compute_single_action(observation=observation)
