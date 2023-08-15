import gymnasium as gym
import numpy as np


class ShadowHandEnv(gym.Env):
    def __init__(self, env_config: dict):
        x = env_config['x']
        self._target_ctrls = env_config['y']
        ctrl_limits = env_config['ctrl_limits']

        sign_values = x['sign']
        order_values = x['order']
        self._observations = [{'sign': sign, 'order': ctrl} for sign, ctrl in zip(sign_values, order_values)]
        self._total_timesteps = len(self._observations) - 1
        self._timestep = 0

        lows = []
        highs = []
        for (left_limit, right_limit) in ctrl_limits:
            lows.append(left_limit)
            highs.append(right_limit)
        lows = np.float32(lows)
        highs = np.float32(highs)

        sign_shape = sign_values.shape[1:]
        order_shape = order_values.shape[1:]

        self.action_space = gym.spaces.Box(low=lows, high=highs, shape=self._target_ctrls.shape[1:], dtype=np.float32)
        self.observation_space = gym.spaces.Dict({
            'sign': gym.spaces.Box(low=0.0, high=1.0, shape=sign_shape, dtype=np.float32),
            'order': gym.spaces.Box(low=0.0, high=1.0, shape=order_shape, dtype=np.float32)
        })

    # Computes reward as 1/dist, where dist is the Euclidean distance between predicted ctrl and target ctrl
    def _compute_reward(self, pred_ctrl: np.ndarray) -> float:
        target_ctrl = self._target_ctrls[self._timestep]
        dist = np.linalg.norm(target_ctrl - pred_ctrl)

        if dist == 0.0:
            return 100
        else:
            return 1.0/dist

    # Increments environment timestep
    def _increment_timestep(self) -> bool:
        self._timestep = (self._timestep + 1) % self._total_timesteps
        return self._timestep == 0

    # Returns current observation: (sign, ctrl)
    def reset(self, **kwargs) -> (dict[str, np.ndarray], dict):
        info = {}
        return self._observations[self._timestep], info

    # Predicts the target ctrl from current observation and rewards the agent
    def step(self, action: np.ndarray) -> (np.ndarray, float, bool, bool, dict):
        truncated = False
        info = {}

        reward = self._compute_reward(pred_ctrl=action)
        done = self._increment_timestep()
        next_state = self._observations[self._timestep]
        return next_state, reward, done, truncated, info

    def render(self):
        raise NotImplementedError()
