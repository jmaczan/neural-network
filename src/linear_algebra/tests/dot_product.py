import unittest

from src.linear_algebra.dot_product import dot_product


class TestDotProduct(unittest.TestCase):
    def test_return_weights_if_not_empty(self):
        # given
        vector_1 = [1, 2, 3]
        vector_2 = [4, 5, 6]

        # when
        output = dot_product(vector_1=vector_1, vector_2=vector_2)

        # then
        self.assertEqual(output, 32)


if "__name__" == "__main__":
    unittest.main()
