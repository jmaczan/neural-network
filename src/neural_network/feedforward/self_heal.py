def self_heal_train_input(
    batch_size,
    training_set,
):
    if batch_size > len(training_set):
        batch_size = len(training_set)
        print("Self-heal applied: batch_size aligned with training_set size")

    return (batch_size, training_set)
