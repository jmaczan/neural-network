def mean_squared_error(predictions, labels):
    """
    Square of subtraction of neuron activation (an output of neural network) from desired output (called y)

    C0 = (a(L) - y)^2

    Often used as a cost function
    """
    if len(predictions) != len(labels):
        raise Exception(
            "Mean Squared Error cannot be computed - incompatibile dimensions of received and expected values"
        )
    return (1 / len(predictions)) * sum(
        list(
            map(
                compute_single_prediction(labels),
                enumerate(predictions),
            )
        )
    )


def compute_single_prediction(labels):
    return lambda index_prediction: (
        1
        / len(index_prediction[1])
        * sum(
            list(
                map(
                    lambda index_prediction: (
                        index_prediction[1]
                        - labels[index_prediction[0]][index_prediction[0]]
                    )
                    ** 2,
                    enumerate(index_prediction[1]),
                )
            )
        )
    )
