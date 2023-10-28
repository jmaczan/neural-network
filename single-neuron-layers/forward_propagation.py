class ForwardPropagation:
    def compute_neuron_activation(
        self, weights, prev_layer_activations, bias, activation_function
    ):
        """
        How to influence neuron activation:
        - increase bias
        - increase weights in proportion to corresponding prev_layer_activations - Hebbian theory "neurons that fire together wire together"
        - (for this one we don't have direct control) change prev_layer_activations in proportion to corresponding weights

        [?] This is what a single neuron for a single input wants
        """
        return activation_function(weights * prev_layer_activations + bias)
