import numpy as np
from controllers.controller import Controller


class ExpertController(Controller):
    def __init__(self, ctrl_limits: list[np.ndarray], signs: dict[str, list[np.ndarray]]):
        super().__init__(ctrl_limits=ctrl_limits)

        self._signs = signs
        self._ctrl_transition_iter = None

    def _set_sign(self, sign: str):
        self._ctrl_transition_iter = iter(self._signs[sign])

    def _get_next_control(self, sign: str, order: int) -> np.ndarray or None:
        return next(self._ctrl_transition_iter, None)
