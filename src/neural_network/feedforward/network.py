"""
For educational purposes

Based on 3Blue1Brown Deep Learning series

https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi
"""


from src.activation_function import rectifier
from src.activation_function.softmax import softmax
from src.neural_network.feedforward.backpropagation import Backpropagation
from src.neural_network.feedforward.forward_propagation import ForwardPropagation
from src.neural_network.feedforward.initialize import initialize_random_weights
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
        validate_train_input(
            weights=weights,
            biases=biases,
            training_set=training_set,
            labels=labels,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            hidden_layers=hidden_layers,
        )

        self.weights = initialize_random_weights(
            weights=weights,
            input_layer_size=len(training_set[0]),
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


if __name__ == "__main__":
    todo()
