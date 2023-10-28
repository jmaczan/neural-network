"""
For educational purposes

Based on 3Blue1Brown Deep Learning series

https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi
"""


class NeuralNetwork:
    def __init__(self):
        self.weights = []
        self.biases = []
        self.input = []
        self.output = []

    def compute_neuron(self, weights, prev_layer_activations, bias, activation_function):
        """
        How to influence neuron activation:
        - increase bias
        - increase weights in proportion to corresponding prev_layer_activations - Hebbian theory "neurons that fire together wire together"
        - (for this one we don't have direct control) change prev_layer_activations in proportion to corresponding weights

        [?] This is what a single neuron for a single input wants
        """
        return activation_function(weights * prev_layer_activations + bias)

    def compute_cost_function_wrt_weight(self):
        """
        How sensitive the cost function is to small changes in our weight (a particular weight from particular layer)

        a.k.a.

        derivative of cost function with respect to that weight (dC0/dw(L))

        we apply chain rule here, so to compute this derivative ☝️, we:

        1. compute derivative of z (z is value of neuron activation but before activation function applied to) with respect to weight on a given layer (dz(L)w(L)), which is
        w(L) * a (L - 1) + b (L)
        where:
        w(L) is a weight on a given layer
        a (L - 1) is a neuron activation from a previous layer
        and b (L) is a bias on a given layer
        2. multiply the result by
        3. derivative of neuron activation with respect to z (da(L)/dz(L))
        4. multiply the result by
        5. derivative of cost function with respect to neuron activation (dC0/da(L))
        """
        todo()

    def compute_cost_function(self, neuron_activation, desired_output):
        """
        Square of subtraction of neuron activation ([?] on output of neural network) from desired output (called y)

        C0 = (a(L) - y)^2
        """
        return (neuron_activation - desired_output) ** 2

    def compute_derivative_of_loss_function_wrt_neuron_activation(
        self, neuron_activation, desired_output
    ):
        """
        dC0/da(L) = 2(a(L) - y)
        """
        return 2 * (neuron_activation - desired_output)
    
    def compute_derivative_of_activation_function(self, argument):
        """
        TODO: Maybe it should be set when creating NeuralNetwork, along with activation function?
        So user passes an enum with activation function to NeuralNetwork and the relevant derivative computation function is being retrieved for it?
        """
        todo()
        return

    def compute_derivative_of_neuron_activation_wrt_z(self, neuron_activation, z):
        """
        It's a derivative of activation function (whichever one we use) with z(L) as a parameter

        da(L)/dz(L) = activation_function'(z(L))
        """
        return self.compute_derivative_of_activation_function(z)
    



def todo():
    """Use it when something is not yet implemented"""


if __name__ == "__main__":
    print(NeuralNetwork())
