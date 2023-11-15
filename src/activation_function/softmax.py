from math import pow, e

from src.activation_function.activation_function import ActivationFunction


def softmax(vector):
    exponentials = list(map(lambda element: pow(e, element), vector))
    exponentials_sum = sum(exponentials)
    return list(map(lambda element: element / exponentials_sum, exponentials))


def softmax_derivative(vector):
    return vector


Softmax = ActivationFunction(softmax, softmax_derivative)
