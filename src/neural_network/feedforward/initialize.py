from random import uniform


def initialize_weights(
    input_layer_size,
    hidden_layers,
    output_layer_size,
    weights=[],
):
    if len(weights) > 0:
        return weights

    return [
        [
            [uniform(0.0, 2.0) for _ in range(input_layer_size)]
            for _ in range(hidden_layers[0])
        ],
        *[
            [
                [uniform(0.0, 2.0) for _ in range(hidden_layers[index])]
                for _ in range(hidden_layers[index + 1])
            ]
            for index in range(len(hidden_layers) - 1)
        ],
        [
            [uniform(0.0, 2.0) for _ in range(output_layer_size)]
            for _ in range(hidden_layers[-1])
        ],
    ]


def initialize_biases(
    hidden_layers,
    output_layer_size,
    biases=[],
):
    if len(biases) > 0:
        return biases

    return [
        *[
            [uniform(0.0, 2.0) for _ in range(hidden_layer)]
            for hidden_layer in hidden_layers
        ],
        [uniform(0.0, 2.0) for _ in range(output_layer_size)],
    ]
