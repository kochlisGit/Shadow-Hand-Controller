import numpy as np
from abc import ABC, abstractmethod


class Model(ABC):
    @abstractmethod
    def build(self, summary: bool):
        pass

    @abstractmethod
    def load(self, checkpoint_directory: str):
        pass

    @abstractmethod
    def save(self, checkpoint_directory: str):
        pass

    @abstractmethod
    def train(self, x: dict[str, np.ndarray] or None, y: np.ndarray or None) -> list[float]:
        pass

    @abstractmethod
    def predict_next_control(self, sign_vector: np.ndarray, order_vector: np.ndarray) -> np.ndarray:
        pass
