from src.utils.utils import todo


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
        return activation_function(weights * prev_layer_activations + bias)

    @staticmethod
    def predict(input, weights, biases, activation_function):
        input_activation = ForwardPropagation().compute_neuron_activation(
            weights=weights[0][0],
            prev_layer_activations=input,
            bias=biases[0][0],
            activation_function=activation_function,
        )

        first_hidden_layer_activation = ForwardPropagation().compute_neuron_activation(
            weights=weights[1][0],
            prev_layer_activations=input_activation,
            bias=biases[1][0],
            activation_function=activation_function,
        )

        second_hidden_layer_activation = ForwardPropagation().compute_neuron_activation(
            weights=weights[2][0],
            prev_layer_activations=first_hidden_layer_activation,
            bias=biases[2][0],
            activation_function=activation_function,
        )

        output_activation = ForwardPropagation().compute_neuron_activation(
            weights=weights[3][0],
            prev_layer_activations=second_hidden_layer_activation,
            bias=biases[3][0],
            activation_function=activation_function,
        )

        todo()

        return output_activation
