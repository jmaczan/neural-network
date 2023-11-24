def jacobian(vector_1, vector_2, derivative_function):
    for i in vector_1:
        for j in vector_2:
            derivative_function(vector_1, vector_2, vector_1_index=i, vector_2_index=j)
