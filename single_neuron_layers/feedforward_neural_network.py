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

        self.output = ForwardPropagation().predict(
            input=self.input,
            weights=self.weights,
            biases=self.biases,
            activation_function=self.rectifier,
        )

        print(self.output)

        todo()

    @staticmethod
    def rectifier(x):
        """ReLU"""
        return 0 if x < 0 else x


if __name__ == "__main__":
    FeedforwardNeuralNetwork().predict(3)
