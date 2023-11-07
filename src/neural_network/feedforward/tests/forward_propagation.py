from random import seed
import unittest
from src.activation_function.rectifier import rectifier
from src.activation_function.softmax import softmax

from src.neural_network.feedforward.forward_propagation import ForwardPropagation
from src.neural_network.feedforward.initialize import (
    initialize_biases,
    initialize_weights,
)


class TestForwardPropagation(unittest.TestCase):
    def test_prediction(self):
        seed(42)  # fixed randomity

        # given
        input_activations = [1, 1.5]
        hidden_layers = [8, 4]
        weights = initialize_weights(
            input_layer_size=len(input_activations),
            hidden_layers=hidden_layers,
            output_layer_size=3,
        )
        biases = initialize_biases(
            hidden_layers=hidden_layers,
            output_layer_size=3,
        )

        # when
        output = ForwardPropagation().predict_single_activation(
            input_activations=input_activations,
            weights=weights,
            biases=biases,
            activation_function=rectifier,
            output_activation_function=softmax,
        )

        # then
        self.assertEqual(output, [1.3991527840897747e-17, 1.0, 1.5500617547033215e-22])


if "__name__" == "__main__":
    unittest.main()
