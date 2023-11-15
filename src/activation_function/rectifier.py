from src.activation_function.activation_function import ActivationFunction


def rectifier(x):
    """ReLU"""
    return 0 if x < 0 else x


def rectifier_derivative(x):
    return 0 if x <= 0 else 1


Rectifier = ActivationFunction(rectifier, rectifier_derivative)
