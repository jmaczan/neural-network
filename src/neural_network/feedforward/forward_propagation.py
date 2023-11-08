from src.linear_algebra.dot_product import dot_product


class ForwardPropagation:
    @staticmethod
    def compute_neuron_activation(
        weights, prev_layer_activations, bias, activation_function
    ):
        """
        How to influence neuron activation:
        - increase bias
        - increase weights in proportion to corresponding prev_layer_activations - Hebbian theory "neurons that fire together wire together"
        - (for this one we don't have direct control) change prev_layer_activations in proportion to corresponding weights

        [?] This is what a single neuron for a single input wants
        """
        return activation_function(dot_product(weights, prev_layer_activations) + bias)

    @staticmethod
    def forward_pass(
        input_activations,
        weights,
        biases,
        activation_function,
        output_activation_function,
    ):
        """
        Similar to prediction (inference), but with storing activations. I can also apply some techniques like dropout and batch normalization here,
        unlike prediction
        """
        activations = []
        predictions = []

        for input_vector in input_activations:
            (
                input_vector_activations,
                input_vector_predictions,
            ) = ForwardPropagation().forward_pass_single_input_activation(
                input_activations=input_vector,
                weights=weights,
                biases=biases,
                activation_function=activation_function,
                output_activation_function=output_activation_function,
            )

            activations.append(input_vector_activations)
            predictions.append(input_vector_predictions)

        return (activations, predictions)

    @staticmethod
    def forward_pass_single_input_activation(
        input_activations,
        weights,
        biases,
        activation_function,
        output_activation_function,
    ):
        activations = []

        first_hidden_layer_activations = list(
            map(
                lambda index_weight: ForwardPropagation().compute_neuron_activation(
                    weights=index_weight[1],
                    prev_layer_activations=input_activations,
                    bias=biases[0][index_weight[0]],
                    activation_function=activation_function,
                ),
                enumerate(weights[0]),
            )
        )

        activations.append(first_hidden_layer_activations)

        current_hidden_layer_activations = first_hidden_layer_activations

        for hidden_layer_index in range(len(weights) - 2):
            current_hidden_layer_activations = list(
                map(
                    lambda index_weight: ForwardPropagation().compute_neuron_activation(
                        weights=index_weight[1],
                        prev_layer_activations=current_hidden_layer_activations,
                        bias=biases[hidden_layer_index + 1][index_weight[0]],
                        activation_function=activation_function,
                    ),
                    enumerate(weights[hidden_layer_index + 1]),
                )
            )
            activations.append(current_hidden_layer_activations)

        output_layer_index = len(weights) - 1

        output_activations = list(
            map(
                lambda index_weight: ForwardPropagation().compute_neuron_activation(
                    weights=index_weight[1],
                    prev_layer_activations=current_hidden_layer_activations,
                    bias=biases[output_layer_index][index_weight[0]],
                    activation_function=activation_function,
                ),
                enumerate(weights[output_layer_index]),
            )
        )

        activations.append(output_activations)

        return (output_activations, output_activation_function(output_activations))

    @staticmethod
    def predict(
        input_activations,
        weights,
        biases,
        activation_function,
        output_activation_function,
    ):
        return [
            ForwardPropagation().predict_single_activation(
                input_activations=input_vector,
                weights=weights,
                biases=biases,
                activation_function=activation_function,
                output_activation_function=output_activation_function,
            )
            for input_vector in input_activations
        ]

    @staticmethod
    def predict_single_activation(
        input_activations,
        weights,
        biases,
        activation_function,
        output_activation_function,
    ):
        first_hidden_layer_activations = list(
            map(
                lambda index_weight: ForwardPropagation().compute_neuron_activation(
                    weights=index_weight[1],
                    prev_layer_activations=input_activations,
                    bias=biases[0][index_weight[0]],
                    activation_function=activation_function,
                ),
                enumerate(weights[0]),
            )
        )

        current_hidden_layer_activations = first_hidden_layer_activations

        for hidden_layer_index in range(len(weights) - 2):
            current_hidden_layer_activations = list(
                map(
                    lambda index_weight: ForwardPropagation().compute_neuron_activation(
                        weights=index_weight[1],
                        prev_layer_activations=current_hidden_layer_activations,
                        bias=biases[hidden_layer_index + 1][index_weight[0]],
                        activation_function=activation_function,
                    ),
                    enumerate(weights[hidden_layer_index + 1]),
                )
            )

        output_layer_index = len(weights) - 1

        output_activations = list(
            map(
                lambda index_weight: ForwardPropagation().compute_neuron_activation(
                    weights=index_weight[1],
                    prev_layer_activations=current_hidden_layer_activations,
                    bias=biases[output_layer_index][index_weight[0]],
                    activation_function=activation_function,
                ),
                enumerate(weights[output_layer_index]),
            )
        )

        return output_activation_function(output_activations)
