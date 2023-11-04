import unittest

from src.neural_network.feedforward.initialize import (
    initialize_biases,
    initialize_weights,
)


class TestWeightsInitialization(unittest.TestCase):
    """
    Important thing to remember:
    unit tests need to start with "test_" in order to be discovered by python -m unittest -v src/neural_network/feedforward/tests/initialize.py
    """

    def test_return_weights_if_not_empty(self):
        # given
        input_layer_size = 5
        hidden_layers = [3, 2]
        output_layer_size = 2
        weights = [[0], [1]]

        # when
        output = initialize_weights(
            input_layer_size=input_layer_size,
            hidden_layers=hidden_layers,
            output_layer_size=output_layer_size,
            weights=weights,
        )

        # then
        self.assertEqual(output, weights)

    def test_calculated_weights(self):
        # given
        input_layer_size = 5
        hidden_layers = [3, 2]
        output_layer_size = 2

        # when
        output = initialize_weights(
            input_layer_size=input_layer_size,
            hidden_layers=hidden_layers,
            output_layer_size=output_layer_size,
        )

        # then
        # input to first hidden, first hidden to second hidden, second hidden to output
        self.assertEqual(len(output), 3)

        # input layer size should be equal to first hidden layer length
        self.assertEqual(len(output[0]), 3)

        # each of 3 neurons in first hidden layer should have 5 weights, one per input layer neuron
        self.assertEqual(len(output[0][0]), 5)
        self.assertEqual(len(output[0][1]), 5)
        self.assertEqual(len(output[0][2]), 5)

        # 2 neurons in second hidden layer
        self.assertEqual(len(output[1]), 2)

        # second hidden layer should have 3 weights from first hidden layer for each of 2 neurons
        self.assertEqual(len(output[1][0]), 3)
        self.assertEqual(len(output[1][1]), 3)

        # last array contains weights from last hidden layer to output layer
        self.assertEqual(len(output[2]), 2)

        # output layer should have 2 weights from last hidden layer for each of 2 output neurons
        self.assertEqual(len(output[2][0]), 2)
        self.assertEqual(len(output[2][1]), 2)


class TestBiasesInitialization(unittest.TestCase):
    """
    Important thing to remember:
    unit tests need to start with "test_" in order to be discovered by python -m unittest -v src/neural_network/feedforward/tests/initialize.py
    """

    def test_return_biases_if_not_empty(self):
        # given
        hidden_layers = [3, 2]
        output_layer_size = 2
        biases = [[0], [1]]

        # when
        output = initialize_biases(
            hidden_layers=hidden_layers,
            output_layer_size=output_layer_size,
            biases=biases,
        )

        # then
        self.assertEqual(output, biases)

    def test_calculated_biases(self):
        # given
        hidden_layers = [3, 5, 2]
        output_layer_size = 2

        # when
        output = initialize_biases(
            hidden_layers=hidden_layers,
            output_layer_size=output_layer_size,
        )

        # then
        # biases of neurons in first hidden, second hidden and output
        self.assertEqual(len(output), 4)

        # number of biases in first hidden layer should be equal to number of neurons in this layer
        self.assertEqual(len(output[0]), 3)

        # number of biases in second hidden layer should be equal to number of neurons in this layer
        self.assertEqual(len(output[1]), 5)

        # number of biases in third layer should be equal to number of neurons in this layer
        self.assertEqual(len(output[2]), 2)

        # number of biases in output layer should be equal to number of neurons in this layer
        self.assertEqual(len(output[3]), 2)


if "__name__" == "__main__":
    unittest.main()
