"""
For educational purposes

Based on 3Blue1Brown Deep Learning series

https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi
"""


from src.activation_function import rectifier
from src.loss_function.mean_squared_error import mean_squared_error
from src.neural_network.feedforward.backpropagation import Backpropagation
from src.neural_network.feedforward.forward_propagation import ForwardPropagation
from src.utils.utils import todo


class FeedforwardNeuralNetwork:
    """
    Feedforward neural network (FNN) with backpropagation - a "vanilla" neural network
    """

    def __init__(self):
        self.input = []
        self.weights = [[1.5], [-0.5], [0.4], [-0.7]]
        self.biases = [[-0.6], [-0.2], [1.1], [1.2]]
        self.output = []

        # TODO: initialize random weights and biases
        todo()

    def train(
        self,
        training_set,
        labels,
        loss_function=mean_squared_error,
        activation_function=rectifier,
        output_activation_function=rectifier,  # TODO: create additional functions like softmax
        learning_rate=0.01,
        batch_size=10,
    ):
        # get up to batch_size elements from training_set
        # compute how many elements are still waiting to be processed (len(training_set) - batch_size)
        # compute network outputs for all elements in batch_size
        # compute loss for each of them
        # compute one loss using all losses
        # compute gradients of loss function with respect to all weights and biases using chain rule
        # update weights and biases
        # if there are any elements to be processed in training_set, iterate to first point

        Backpropagation().backpropagate(weights=self.weights, biases=self.biases)

        todo()

    def predict(self, input):
        self.input = input

        self.output = ForwardPropagation().predict(
            input=self.input,
            weights=self.weights,
            biases=self.biases,
            activation_function=rectifier,
        )

        print(self.output)

        todo()


if __name__ == "__main__":
    FeedforwardNeuralNetwork().predict(3)
