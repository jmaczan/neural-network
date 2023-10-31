from src.utils.utils import todo

default_learning_rate = 0.01


class Backpropagation:
    def backpropagate(self, weights, biases, learning_rate):
        self.update_weights_and_biases(
            weights=weights,
            biases=biases,
            gradient_vector=self.compute_cost_function_gradient_vector(),
            learning_rate=learning_rate,
        )

    def update_weights_and_biases(
        self, weights, biases, gradient_vector, learning_rate=default_learning_rate
    ):
        todo()

    def compute_cost_function_gradient_vector(self):
        """
        compute_derivative_of_cost_function_all_training_examples for all weights and biases in a single vector
        """
        todo()

    def compute_derivative_of_cost_function_all_training_examples(self):
        """
        Note: This is a single component of a gradient vector, computed for a specific weight

        Averaging together all costs across many training examples

        dC/dw(L) = 1/n * [sum from k=0 to k=n-1 of dC(k)/dw(L)]
        """
        todo()

    def compute_derivative_of_cost_function_wrt_weight_single_training_example(self):
        """
        Note: This is for a single specific training example

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

    def compute_derivative_of_cost_function_wrt_bias_single_training_example(self):
        """
        Same as for weight but dz/db(L) = 1
        because z(L) = w(L) * a(L - 1) + b(L) | db(L)

        1. derivative of neuron activation with respect to z (da(L)/dz(L))
        2. multiply the result by
        3. derivative of cost function with respect to neuron activation (dC0/da(L))
        """
        todo()

    def compute_derivative_of_cost_function_wrt_prev_layer_activation_single_training_example(
        self,
    ):
        """
        Same as for weight but dz/da(L-1) = w(L)
        because z(L) = w(L) * a(L - 1) + b(L) | da(L - 1)

        1. multiply w(L) by
        2. derivative of neuron activation with respect to z (da(L)/dz(L))
        3. multiply the result by
        4. derivative of cost function with respect to neuron activation (dC0/da(L))
        """
        todo()

    def compute_cost_function(self, neuron_activation, desired_output):
        """
        Square of subtraction of neuron activation ([?] on output of neural network) from desired output (called y)

        C0 = (a(L) - y)^2

        Cost function is a mean squared error
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

    def compute_derivative_of_neuron_activation_wrt_z(self, z):
        """
        It's a derivative of activation function (whichever one we use) with z(L) as a parameter

        da(L)/dz(L) = activation_function'(z(L))
        """
        return self.compute_derivative_of_activation_function(z)

    def compute_derivative_of_z_wrt_weight(self, prev_layer_activation):
        """
        dz(L)/dw(L)=a(L - 1)
        because
        z(L) = w(L) * a(L - 1) + b(L)
        """

        return prev_layer_activation
