from random import seed
import unittest

from src.neural_network.feedforward.backpropagation import Backpropagation


class TestBackpropagation(unittest.TestCase):
    def test_compute_cost_function_gradient_vector(self):
        seed(42)  # fixed randomity

        # given
        weights = mock_weights()
        biases = mock_biases()
        activations = mock_activations()
        loss = mock_loss()

        # when
        output = Backpropagation().compute_cost_function_gradient_vector(
            loss=loss, weights=weights, biases=biases, activations=activations
        )

        # then
        self.assertEqual(output, [1])  # TODO


def mock_loss():
    return 0.6622023755842128


def mock_weights():
    return [
        [
            [1.2788535969157675, 0.05002151044533387, 0.5500586367382385],
            [0.4464214762976455, 1.4729424283280248, 1.3533989748458226],
            [1.7843591354096908, 0.1738776652588323, 0.8438436393705409],
        ],
        [
            [0.05959443887614069, 0.4372759496072067, 1.0107105762067248],
            [0.05307193936772725, 0.397675301373297, 1.2997688755590464],
        ],
        [
            [1.0898829612064334, 0.4408812440813934],
            [1.1785313677518174, 1.6188609133556533],
            [0.012997519356122034, 1.6116385036656158],
        ],
    ]


def mock_biases():
    return [
        [1.3962787899764537, 0.6805010330359837, 0.3109589996235631],
        [1.9144261444135624, 0.6731890902252535],
        [0.18549168676029582, 0.19343275366692803, 1.6949887326949196],
    ]


def mock_activations():
    return [
        [
            [3.2752125340757936, 3.953263912507477, 3.113039439662627],
            [6.984659714787194, 6.465358161125991],
            [10.648408449240861, 18.891588938507947, 12.205592134993163],
        ],
        [
            [2.725153897337555, 2.5998649376616543, 2.2691958002920862],
            [5.5071887660820975, 4.8011504390159],
            [8.304420065635915, 14.45622224748828, 9.504287434688989],
        ],
        [
            [3.8896285773110106, 3.440003436492287, 3.9182801747380562],
            [7.610704359408119, 7.3404822424201175],
            [11.716549634054287, 21.046106359350087, 13.624112827278031],
        ],
    ]


if "__name__" == "__main__":
    unittest.main()
