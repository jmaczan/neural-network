from math import pow, e


def softmax(vector):
    exponentials = list(map(lambda element: pow(e, element), vector))
    exponentials_sum = sum(exponentials)
    return list(map(lambda element: element / exponentials_sum, exponentials))
