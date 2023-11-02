def mean_squared_error(received, expected):
    """
    Square of subtraction of neuron activation (an output of neural network) from desired output (called y)

    C0 = (a(L) - y)^2

    Often used as a cost function
    """
    return (received - expected) ** 2
