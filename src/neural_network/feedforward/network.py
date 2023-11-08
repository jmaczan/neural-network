"""
For educational purposes

Based on 3Blue1Brown Deep Learning series

https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi
"""


from src.activation_function.rectifier import rectifier
from src.activation_function.softmax import softmax
from src.loss_function.mean_squared_error import mean_squared_error
from src.neural_network.feedforward.backpropagation import Backpropagation
from src.neural_network.feedforward.forward_propagation import ForwardPropagation
from src.neural_network.feedforward.initialize import (
    initialize_biases,
    initialize_weights,
)
from src.neural_network.feedforward.self_heal import self_heal_train_input
from src.neural_network.feedforward.validate import validate_train_input
from src.utils.utils import todo


class FeedforwardNeuralNetwork:
    """
    Feedforward neural network (FNN) with backpropagation - a "vanilla" neural network
    """

    def __init__(self, weights=[], biases=[]):
        self.input = []
        self.output = []
        self.weights = weights
        self.biases = biases

    def train(
        self,
        hidden_layers,  # define number of hidden layers and number of neurons in each hidden layer, for instance [8, 6] means that network will have 2 hidden layers, first with 8 neurons and second with 6 neurons
        training_set,
        labels,
        weights=[],
        biases=[],
        activation_function=rectifier,
        output_activation_function=softmax,
        learning_rate=0.01,
        batch_size=10,
        epochs=1,
    ):
        (batch_size, training_set) = self_heal_train_input(
            batch_size=batch_size, training_set=training_set
        )

        validate_train_input(
            training_set=training_set,
            labels=labels,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            hidden_layers=hidden_layers,
        )

        self.weights = initialize_weights(
            weights=weights,
            input_layer_size=len(training_set[0]),
            hidden_layers=hidden_layers,
            output_layer_size=len(labels[0]),
        )

        self.biases = initialize_biases(
            biases=biases,
            hidden_layers=hidden_layers,
            output_layer_size=len(labels[0]),
        )

        epoch_index = 0

        while epoch_index < epochs:
            batch_index = 0

            while batch_index * batch_size < len(training_set):
                batch_start = batch_size * batch_index
                batch_end = min(
                    batch_size * (batch_index + 1),
                    len(training_set),
                )

                mini_batch = training_set[batch_start:batch_end]
                mini_batch_labels = labels[batch_start:batch_end]

                predictions = ForwardPropagation().predict(
                    input_activations=mini_batch,  # TODO: here I assume array of input vectors, but in predict method in the same network I assume it to be just a single input vector
                    weights=self.weights,
                    biases=self.biases,
                    activation_function=activation_function,
                    output_activation_function=output_activation_function,
                )

                loss = Backpropagation().compute_loss(
                    predictions=predictions,
                    labels=mini_batch_labels,
                    loss_function=mean_squared_error,
                )

                gradient_vector = (
                    Backpropagation().compute_cost_function_gradient_vector(
                        loss=loss, weights=self.weights, biases=self.biases
                    )
                )

                (weights, biases) = Backpropagation().update_weights_and_biases(
                    weights=self.weights,
                    biases=self.biases,
                    learning_rate=learning_rate,
                    gradient_vector=gradient_vector,
                )

                self.weights = weights
                self.biases = biases

                batch_index = batch_index + 1

            epoch_index = epoch_index + 1

    def predict(self, input):
        self.input = input

        self.output = ForwardPropagation().predict_single_activation(  # predict was used here before, but I think predict_single_activation is correct in this context
            input_activations=self.input,
            weights=self.weights,
            biases=self.biases,
            activation_function=rectifier,
        )

        print(self.output)


if __name__ == "__main__":
    todo()
