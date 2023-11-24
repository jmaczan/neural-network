def jacobian(vector_1, vector_2, derivative_function):
    jacobian_matrix = [[0 for _ in range(len(vector_1))] for _ in range(len(vector_2))]
    for i in vector_1:
        for j in vector_2:
            jacobian_matrix[i][j] = derivative_function(
                vector_1, vector_2, vector_1_index=i, vector_2_index=j
            )

    return jacobian_matrix
