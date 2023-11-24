from math import pow, e

from src.activation_function.activation_function import ActivationFunction
from src.linear_algebra.jacobian import jacobian


def softmax(vector):
    exponentials = list(map(lambda element: pow(e, element), vector))
    exponentials_sum = sum(exponentials)
    return list(map(lambda element: element / exponentials_sum, exponentials))


def softmax_derivative(z_vector, p_vector):
    return jacobian(z_vector, p_vector, single_softmax_derivative)


def single_softmax_derivative(vector_1, vector_2, vector_1_index, vector_2_index):
    z_vector = vector_1
    p_vector = vector_2
    z_vector_index = vector_1_index
    p_vector_index = vector_2_index

    return (
        on_diagonal_derivative(p_vector[p_vector_index])
        if z_vector_index == p_vector_index
        else off_diagonal_derivative(p_vector[p_vector_index], p_vector[z_vector_index])
    )


def on_diagonal_derivative(p_i):
    return p_i * (1 - p_i)


def off_diagonal_derivative(p_i, p_k):
    return -p_i * p_k


Softmax = ActivationFunction(softmax, softmax_derivative)
