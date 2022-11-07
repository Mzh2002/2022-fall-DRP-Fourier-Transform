#from math import pi
from functools import lru_cache
import numpy as np
import matplotlib.pyplot as plt

# This function returns the primitive Nth root of unity w = e^(2pi*i/N)
def prim_Nth_root(N):
    Re = np.cos(2*np.pi/N)
    Im = np.sin(2*np.pi/N)
    return Re+Im*1j

# This function computes the N by N Fourier matrix
def Fourier_matrix(N):
    w = prim_Nth_root(N)
    return [[w**(m*n) for m in range(N)] for n in range(N)]

# This function computes the DFT of a complex vector x of length N
def DFT(x, N):
    normalized_F_N_bar = np.conjugate(np.multiply(1/np.sqrt(N), Fourier_matrix(N)))
    return normalized_F_N_bar.dot(x)

# This function computes the "twiddle factors" as an array (to make the FFT even faster)
def twiddle(N):
    return np.conjugate(twiddle_inverse(N))

# Use lru_cache so that we don't have to compute it twice
@lru_cache(None)
def twiddle_inverse(N):
    return [prim_Nth_root(N) ** n for n in range(N)]

# This is *only the recursive step* for the FFT of a complex vector x of length N with the "twiddle factor" twid
def FFT_step(twid, x, N):
    # N = 1 the Fourier matrix is [[1]], so we can directly return the number
    if N == 1:
        return x

    # Split the entries into even and odd ones
    N_half = N//2
    x1, x2 = [], []
    for i in range(N_half):
        x1.append(x[2 * i])
        x2.append(x[2 * i + 1])

    # recursively do FFT on even and odd entries
    evenFFT = FFT_step(twid, x1, N_half)
    oddFFT = FFT_step(twid, x2, N_half)

    y = evenFFT
    y.extend(oddFFT)

    # Get the resulting array
    for k in range(N_half):
        p = y[k]
        q = twid[k * (len(twid) // N)] * y[k + N_half]
        #q = np.e ** (-2 * np.pi * 1j * k / (2*N_half)) * y[k + N_half]
        y[k] = (p + q)
        y[k + N_half] = (p - q)
    return y

# This function does the (whole) FFT of a complex vector x with length N

def FFT(x, N):
    twid = twiddle(N)
    return np.multiply(1/np.sqrt(N), FFT_step(twid, x, N))

def IFFT(x, N):
    twid = twiddle_inverse(N)
    return np.multiply(1/np.sqrt(N), IFFT_step(twid, x, N))

def IFFT_step(twid, x, N):
    # N = 1 the Fourier matrix is [[1]], so we can directly return the number
    if N == 1:
        return x

    # Split the entries into even and odd ones
    N_half = N // 2
    x1, x2 = [], []
    for i in range(N_half):
        x1.append(x[2 * i])
        x2.append(x[2 * i + 1])

    # recursively do FFT on even and odd entries
    evenFFT = IFFT_step(twid, x1, N_half)
    oddFFT = IFFT_step(twid, x2, N_half)

    y = evenFFT
    y.extend(oddFFT)

    # Get the resulting array
    for k in range(N_half):
        p = y[k]
        q = twid[k * (len(twid) // N)] * y[k + N_half]
        # q = np.e ** (-2 * np.pi * 1j * k / (2*N_half)) * y[k + N_half]
        y[k] = (p + q)
        y[k + N_half] = (p - q)
    return y
    
# This function turn two vectors into the graph
def basis_expr(N, numbers, DFT_numbers):
    x = [i for i in range(N)]
    #FFT_numbers_magnitude = np.array([np.abs(FFT_numbers[i]) for i in range(N)])
    DFT_numbers_magnitude = np.array([np.abs(DFT_numbers[i]) for i in range(N)])
    figure, ((ax1, ax2)) = plt.subplots(1,2)
    ax1.plot(x, numbers, 'ro')
    ax2.plot(x, DFT_numbers_magnitude, 'ro')
    plt.show()

# Main
if __name__ == "__main__":
    n = int(input("\nPlease input a number n (we'll study Fourier Transform in dimension 2 ** n): "))
    while n > 10:
        n = int(input("\nThe input is too big, please enter a number smaller than or equal to 10: "))
    N = 2 ** n
    f = np.random.rand(N)
    #f = [np.cos(np.pi*k/N) for k in range(N)]
    basis_expr(N, f, IFFT(FFT(f, N), N))