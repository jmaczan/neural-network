import unittest
from src.activation_function.rectifier import rectifier

from src.neural_network.feedforward.forward_propagation import ForwardPropagation


class TestForwardPropagation(unittest.TestCase):
    def test_prediction(self):
        # given
        input = 3
        weights = [[1.5], [-0.5], [0.4], [-0.7]]
        biases = [[-0.6], [-0.2], [1.1], [1.2]]
        output = []

        # when
        output = ForwardPropagation().predict(
            input_activations=input,
            weights=weights,
            biases=biases,
            activation_function=rectifier,
        )

        # then
        self.assertEqual(output, 0.42999999999999994)


if "__name__" == "__main__":
    unittest.main()
