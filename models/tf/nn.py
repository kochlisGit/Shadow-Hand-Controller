import numpy as np
import tensorflow as tf
from models.model import Model


class NeuralNetwork(Model):
    def __init__(
            self,
            input_shapes: dict[str, tuple],
            num_outputs: int,
            learning_rate: float = 0.001,
            loss_fn: str = 'MAE',
            epochs: int = 100,
            batch_size: int = 8
    ):
        self._sign_input_shape = input_shapes['sign']
        self._order_input_shape = input_shapes['order']
        self._num_outputs = num_outputs
        self._learning_rate = learning_rate
        self._loss_fn = loss_fn
        self._epochs = epochs
        self._batch_size = batch_size

        self._model = None

    @property
    def model(self) -> tf.keras.Model or None:
        return self._model

    def load(self, checkpoint_directory: str):
        self._model = tf.keras.models.load_model(checkpoint_directory)

    def save(self, checkpoint_directory: str):
        assert self._model is not None

        self._model.save(checkpoint_directory)

    def build(self, summary: bool):
        i1 = tf.keras.layers.Input(shape=self._sign_input_shape, name='sign')
        h1 = tf.keras.layers.Dense(units=256, activation='relu', name='hidden_1')(i1)

        i2 = tf.keras.layers.Input(shape=self._order_input_shape, name='order')
        h2 = tf.keras.layers.Dense(units=256, activation='relu', name='hidden_2')(i2)

        h = tf.keras.layers.Concatenate(name='h1h2_concat', axis=-1)([h1, h2])
        h = tf.keras.layers.Dense(units=128, activation='relu', name='hidden_final')(h)
        y = tf.keras.layers.Dense(units=self._num_outputs, name='outputs')(h)

        self._model = tf.keras.Model(inputs=[i1, i2], outputs=y, name='neural-network')

        if summary:
            self._model.summary()

    def train(self, x: dict[str, np.ndarray] or None, y: np.ndarray or None) -> list[float]:
        assert self._model is not None

        self._model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self._learning_rate),
            loss=self._loss_fn,
            metrics=[self._loss_fn]
        )
        history = self._model.fit(
            x=x,
            y=y,
            batch_size=self._batch_size,
            epochs=self._epochs
        )
        return history.history['loss']

    def predict_next_control(self, sign_vector: np.ndarray, order_vector: np.ndarray) -> np.ndarray:
        assert self._model is not None
        assert sign_vector.ndim == order_vector.ndim

        if sign_vector.ndim == 1:
            x = {'sign': np.expand_dims(a=sign_vector, axis=0), 'order': np.expand_dims(a=order_vector, axis=0)}
            next_ctrl = self._model.predict(x)[0]
        else:
            x = {'sign': sign_vector, 'order': order_vector}
            return self._model.predict(x)
        return next_ctrl
