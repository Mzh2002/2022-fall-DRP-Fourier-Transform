import matplotlib.pyplot as plt
import numpy as np


def basis_expr(dimension, numbers):
    x = []
    for i in range(dimension):
        x.append(i)
    plt.xlim(-.5, dimension-.5)
    plt.xticks(np.linspace(0, dimension-1, dimension, endpoint=True))
    plt.xlabel('Index of basis')
    y_max = int(max(numbers)) + 1
    y_min = int(min(numbers)) - 1
    plt.ylim(min(0, y_min), max(0, y_max))
    plt.ylabel('Magnitude of coefficient')
    plt.plot(x, numbers, 'ro')
    plt.show()


if __name__ == '__main__':
    N = int(input("Please enter the dimension of the vector space: "))
    coefficients = []
    for i in range(N):
        coefficients.append(float(input(f'a{i}: ')))
    basis_expr(N, coefficients)