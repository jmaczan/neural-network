from math import pow, e


def softmax(vector):
    exponentials = map(lambda element: pow(e, element), vector)
    exponentials_sum = sum(exponentials)
    return map(lambda element: element / exponentials_sum, exponentials)
