from random import uniform


@staticmethod
def initialize_random_weights(
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
