import numpy as np
from utils.dataset import NUM_ACTUATORS
from abc import ABC, abstractmethod


class Controller(ABC):
    def __init__(self, ctrl_limits: list[np.ndarray]):
        assert len(ctrl_limits) == NUM_ACTUATORS

        self._ctrl_limits = ctrl_limits
        self._order = 0
        self._is_done = False

    @property
    def order(self) -> int:
        return self._order

    @property
    def is_done(self) -> bool:
        return self._is_done

    # Sets the behavior controller to specified sign
    def set_sign(self, sign: str):
        self._order = 0
        self._is_done = False

        self._set_sign(sign=sign)

    @abstractmethod
    def _set_sign(self, sign: str):
        pass

    # Returns the next control of the transition. The positions are clipped according to control limits just in case
    def get_next_control(self, sign: str) -> np.ndarray or None:
        if self._is_done:
            return None

        next_ctrl = self._get_next_control(sign=sign, order=self._order)

        if next_ctrl is None:
            self._is_done = True
        else:
            assert next_ctrl.shape[0] == NUM_ACTUATORS

            self._order += 1

            for i, (low, high) in enumerate(self._ctrl_limits):
                next_ctrl[i] = np.clip(a=next_ctrl[i], a_min=low, a_max=high)
        return next_ctrl

    @abstractmethod
    def _get_next_control(self, sign: str, order: int) -> np.ndarray or None:
        pass
