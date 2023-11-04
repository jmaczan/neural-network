import unittest

from src.neural_network.feedforward.initialize import initialize_random_weights


class TestWeightInitialize(unittest.TestCase):
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
        output = initialize_random_weights(
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

        expected_weights = [
            [
                [
                    1.8171921270060511,
                    0.8917305605062487,
                    0.42330929962290154,
                    0.41228386582543575,
                    0.9128895166563964,
                ],
                [
                    0.662208017041616,
                    0.6152839296205794,
                    1.0470483775855672,
                    1.5066555088041018,
                    1.129147040526466,
                ],
                [
                    0.9720204830387191,
                    1.3872236277768013,
                    1.216666497305848,
                    0.04622559191715814,
                    0.7815154751745081,
                ],
            ],
            [
                [0.5860722291297842, 1.2754204546371686, 1.1756673882433955],
                [0.21799442672806757, 1.8766441711236053, 0.8904777009611113],
            ],
            [
                [1.8267175719906732, 1.1994804863252007],
                [1.703879782270657, 0.9874842537391955],
            ],
        ]

        # when
        output = initialize_random_weights(
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


if "__name__" == "__main__":
    unittest.main()
