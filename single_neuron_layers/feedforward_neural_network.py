"""
For educational purposes

Based on 3Blue1Brown Deep Learning series

https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi
"""


from forward_propagation import ForwardPropagation
from utils import todo


class FeedforwardNeuralNetwork:
    """
    Feedforward neural network (FNN) with backpropagation - a "vanilla" neural network

    This implementation has a single input, two hidden layers with single neuron each and a single output
    """

    def __init__(self):
        self.input = []
        self.weights = [[1.5], [-0.5], [0.4], [-0.7]]
        self.biases = [[-0.6], [-0.2], [1.1], [1.2]]
        self.output = []

        # TODO: initialize random weights and biases
        todo()

    def train(self):
        todo()

    def predict(self, input):
        self.input = input

        input_activation = ForwardPropagation().compute_neuron_activation(
            weights=self.weights[0][0],
            prev_layer_activations=self.input,
            bias=self.biases[0][0],
            activation_function=self.rectifier,
        )

        first_hidden_layer_activation = ForwardPropagation().compute_neuron_activation(
            weights=self.weights[1][0],
            prev_layer_activations=input_activation,
            bias=self.biases[1][0],
            activation_function=self.rectifier,
        )

        second_hidden_layer_activation = ForwardPropagation().compute_neuron_activation(
            weights=self.weights[2][0],
            prev_layer_activations=first_hidden_layer_activation,
            bias=self.biases[2][0],
            activation_function=self.rectifier,
        )

        output_activation = ForwardPropagation().compute_neuron_activation(
            weights=self.weights[3][0],
            prev_layer_activations=second_hidden_layer_activation,
            bias=self.biases[3][0],
            activation_function=self.rectifier,
        )

        print(output_activation)

        todo()

    @staticmethod
    def rectifier(x):
        return 0 if x < 0 else x


if __name__ == "__main__":
    print(FeedforwardNeuralNetwork().predict(3))
