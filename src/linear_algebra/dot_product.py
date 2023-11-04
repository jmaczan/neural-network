def dot_product(vector_1, vector_2):
    if len(vector_1) != len(vector_2):
        raise Exception(
            f"Dimensions of vectors for dot product should be the same, but they are not. Vector 1: {vector_1}, vector 2: {vector_2}"
        )

    result = 0

    for index in range(len(vector_1)):
        result = result + vector_1[index] * vector_2[index]

    return result
