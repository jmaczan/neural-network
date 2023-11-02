"""
For educational purposes

Based on 3Blue1Brown Deep Learning series

https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi
"""


from src.activation_function import rectifier
from src.activation_function.softmax import softmax
from src.neural_network.feedforward.backpropagation import Backpropagation
from src.neural_network.feedforward.forward_propagation import ForwardPropagation
from src.utils.utils import todo
from random import uniform


class FeedforwardNeuralNetwork:
    """
    Feedforward neural network (FNN) with backpropagation - a "vanilla" neural network
    """

    def __init__(self, weights, biases):
        self.input = []
        self.output = []
        self.weights = weights if weights is not None else []
        self.biases = biases if biases is not None else []

    def train(
        self,
        weights,
        biases,
        training_set=[],
        labels=[],
        activation_function=rectifier,
        output_activation_function=softmax,
        learning_rate=0.01,
        batch_size=10,
        epochs=1,
        hidden_layers=[],  # define number of hidden layers and number of neurons in each hidden layer, for instance [8, 6] means that network will have 2 hidden layers, first with 8 neurons and second with 6 neurons
    ):
        self.__validate_train_input(
            weights=weights,
            biases=biases,
            training_set=training_set,
            labels=labels,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            hidden_layers=hidden_layers,
        )

        self.__initialize_weights(
            weights=weights,
            training_set=training_set,
            hidden_layers=hidden_layers,
            labels=labels,
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
                    input=mini_batch,
                    weights=self.weights,
                    biases=self.biases,
                    activation_function=activation_function,
                    output_activation_function=output_activation_function,
                )

                loss = Backpropagation().compute_loss(
                    predictions=predictions,
                    labels=mini_batch_labels,
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

        self.output = ForwardPropagation().predict(
            input=self.input,
            weights=self.weights,
            biases=self.biases,
            activation_function=rectifier,
        )

        print(self.output)

    def __validate_train_input(
        self,
        training_set,
        labels,
        learning_rate,
        batch_size,
        epochs,
        hidden_layers,
    ):
        if len(training_set) != len(labels):
            raise Exception("Labels dimension doesn't match training set dimension")

        if learning_rate < 0:
            raise Exception("Learning rate should be a positive float number")

        if batch_size < 0:
            raise Exception("Batch size should be a positive float number")

        if batch_size > len(training_set):
            raise Exception("Batch size can't be bigger than size of training set")

        if epochs < 0:
            raise Exception("Number of epochs should be a positive float number")

        if len(hidden_layers) == 0:
            raise Exception(
                "Please define number of hidden layers and number of neurons in each hidden layer, for instance [8, 6] means that network will have 2 hidden layers, first with 8 neurons and second with 6 neurons"
            )

    def __initialize_weights(self, weights, training_set, hidden_layers, labels):
        if len(weights) > 0:
            self.weights = weights
            return

        self.weights = [
            (
                [uniform(0.0, 2.0) for _ in range(len(training_set[0]))]
                for _ in range(hidden_layers[0])
            ),
            (
                [uniform(0.0, 2.0) for _ in range(hidden_layer)]
                for index, hidden_layer in enumerate(hidden_layers)
            ),
            [uniform(0.0, 2.0) for _ in range(len(labels[0]))],
        ]


if __name__ == "__main__":
    todo()
