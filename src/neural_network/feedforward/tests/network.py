from random import seed
import unittest

from src.neural_network.feedforward.network import FeedforwardNeuralNetwork


class TestNetwork(unittest.TestCase):
    def test_train(self):
        seed(42)

        # given
        hidden_layers = [3, 2]
        network = FeedforwardNeuralNetwork()

        # when
        output = network.train(
            hidden_layers=hidden_layers,
            training_set=[[1, 1, 1], [1, 1, 0], [1.5, 0.5, 1]],
            labels=[[0, 1, 0], [1, 0, 0], [0, 0, 1]],
        )

        # then
        self.assertEqual(output, [])


if "__name__" == "__main__":
    unittest.main()
