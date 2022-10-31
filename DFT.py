#from math import pi
import numpy as np
import matplotlib.pyplot as plt

# This function turn two vectors into the graph
def basis_expr(dimension, numbers, numbers_hat):
    x = []
    numbers_hat_real = np.array([numbers_hat[i].real for i in range(dimension)])
    numbers_hat_imag = np.array([numbers_hat[i].imag for i in range(dimension)])
    magnitude = []
    for i in range(dimension):
        magnitude.append(np.sqrt(numbers_hat_real[i]**2 + numbers_hat_imag[i]**2))
    magnitude = np.array(magnitude)
    figure, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2,2)
    for i in range(dimension):
        x.append(i)
    ax1.plot(x, numbers, 'ro')
    ax2.plot(x, numbers_hat_real, 'ro')
    ax3.plot(x, numbers_hat_imag, 'ro')
    ax4.plot(x, magnitude, 'ro')
    plt.show()

# This function returns the primitive Nth root of unity w = e^(2pi*i/N)
def prim_Nth_root(N):
    Re = np.cos(2*np.pi/N)
    Im = np.sin(2*np.pi/N)
    return Re+Im*1j

# This function returns the Fourier matrix of dimension N with decimal precision p
def Fourier_matrix(N):
    w = prim_Nth_root(N)
    return [[w**(m*n) for m in range(N)] for n in range(N)]

def Conjugate(matrix, N):
    for i in range(N):
        for j in range(N):
            matrix[i][j] = matrix[i][j] - 2 * matrix[i][j].imag * 1j
    return matrix

# Main
if __name__ == "__main__":
    N = int(input("\nPlease input the desired dimension of the C-vector space: "))
    # p = int(input("\nPlease choose a desired printing precision: "))

    # Pre-compute w, F_N, and the "normalized" Fourier matrix norm_F_N here (since we might use them a lot below)
    w = prim_Nth_root(N)
    F_N = Fourier_matrix(N)
    norm_F_N = Conjugate(np.multiply(1/np.sqrt(N), F_N), N)

    # Print them to the desired precision
    # print ("\nThe primitive {}th root of unity is w = {}+{}j, so the {}th Fourier matrix is\n\nF_{} =\n".format(N,np.round_(np.real(w),p),np.round_(np.imag(w),p),N,N))
    # print (np.round_(np.matrix(F_N),3))

    random_vector_f = np.random.rand(N)
    f_hat = norm_F_N.dot(random_vector_f)
    basis_expr(N, random_vector_f, f_hat)

    # Try this:
    # 1) Use a random number generator to create and plot a random vector
    #    in R^N (which can be understood as a function f : Z/NZ --> C)
    #    using your vector_graph.py code (we don't really need multiple files;
    #    you could just copy its code into this program).
    # 2) Compute the Fourier transform of f (call it f_hat). This amounts
    #    to transposing the array f, then multiplying it by the normalized
    #    Fourier matrix using np.matmul.
    # 3) The f_hat you get will be a vector in C^N (probably not in R^N
    #    like f was). For each n in range(N), compute the absolute value
    #    (aka magnitude) of the nth entries of f and f_hat, namely
    #    |f(n)| and |f_hat(n)| and plot the arrays [|f(0)|, |f(1)|, ... , |f(N-1)|]
    #    and [|f_hat(0)|, |f_hat(1)|, ... , |f_hat(N-1)|].
    # 4) Think back to what we've read/discussed. What does the array of |f_hat(n)|'s
    #    "say" about the f you started with? Does it have a relationship with the
    #    array of |f(n)|'s? 
    # 5) Repeat everything with the array/function f : Z/NZ --> C defined by
    #    f(n) = sin(2pi*k*n/N) for various choices of constant integer k.
    #    At the very least, try this with k = 0, 1, 2, and 3.
    # 6) Bring your computer and we can do more coding/experimenting on Monday! 