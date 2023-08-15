import random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from models.tf.nn import NeuralNetwork
from utils import dataset


dataset_filepath = 'data/expert_dataset.csv'
checkpoint_directory = 'checkpoints/nn'
one_hot_signs = True
num_outputs = dataset.NUM_ACTUATORS
learning_rate = 0.001
loss_fn = 'MAE'
epochs = 1000
batch_size = 8
seed = 0

summary = True
print_test_predictions = True
plot_performance = True


def main():
    tf.random.set_seed(seed=seed)
    np.random.seed(seed=seed)
    random.seed(seed)

    x, y = dataset.read_dataset(dataset_filepath=dataset_filepath, one_hot=one_hot_signs)

    print('Train Dataset:', y.shape, x['sign'].shape, x['order'].shape)

    model = NeuralNetwork(
        input_shapes={name: input_data.shape[1:] for name, input_data in x.items()},
        num_outputs=num_outputs,
        learning_rate=learning_rate,
        loss_fn=loss_fn,
        epochs=epochs,
        batch_size=batch_size
    )
    model.build(summary=summary)
    loss = model.train(x=x, y=y)
    model.save(checkpoint_directory=checkpoint_directory)

    if print_test_predictions:
        y_pred = model.predict_next_control(sign=x['sign'], order=x['order'])

        for i in range(y.shape[0]):
            print(f'i: {i + 1}, y_pred: {y_pred[i]}\tactual: {y[i]}')

    if plot_performance:
        plt.plot(loss, label='Loss')
        plt.title('Neural Network Performance')
        plt.xlabel('Epochs')
        plt.ylabel(loss_fn)
        plt.legend()
        plt.show()


if __name__ == '__main__':
    main()
