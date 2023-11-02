def mean_squared_error(received, expected):
    """
    Square of subtraction of neuron activation (an output of neural network) from desired output (called y)

    C0 = (a(L) - y)^2

    Often used as a cost function
    """
    if len(received) != len(expected):
        raise Exception(
            "Mean Squared Error cannot be computed - incompatibile dimensions of received and expected values"
        )
    return (1 / len(received)) * sum((received - expected) ** 2)
