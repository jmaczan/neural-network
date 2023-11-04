@staticmethod
def validate_train_input(
    training_set,
    labels,
    learning_rate,
    batch_size,
    epochs,
    hidden_layers,
):
    if len(training_set) != len(labels):
        raise Exception("Labels dimension doesn't match training set dimension")

    if learning_rate < 0:
        raise Exception("Learning rate should be a positive float number")

    if batch_size < 0:
        raise Exception("Batch size should be a positive float number")

    if batch_size > len(training_set):
        raise Exception("Batch size can't be bigger than size of training set")

    if epochs < 0:
        raise Exception("Number of epochs should be a positive float number")

    if len(hidden_layers) == 0:
        raise Exception(
            "Please define number of hidden layers and number of neurons in each hidden layer, for instance [8, 6] means that network will have 2 hidden layers, first with 8 neurons and second with 6 neurons"
        )
