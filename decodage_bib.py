import numpy as np
from itertools import product
import matplotlib.pyplot as plt

def qpsk_alphabet():
    # Alphabet QPSK : {±1 ± j} / sqrt(2)
    A = np.array([
        (1 + 1j),
        (1 - 1j),
        (-1 + 1j),
        (-1 - 1j)
    ], dtype=complex) / np.sqrt(2)
    return A

def generate_channel(N=2, M=2):
    H = (np.random.randn(M, N) + 1j * np.random.randn(M, N)) / np.sqrt(2)
    return H

def generate_codeword(A):
    
    idx = np.random.randint(0, len(A), size=4)
    s = A[idx]
    X = np.array([[s[0], s[2]],
                  [s[1], s[3]]], dtype=complex)
    return X

def generate_noise(M=2, L=2, snr_db=10):
    sigma2 = 10 ** (-snr_db / 10)
    sigma = np.sqrt(sigma2)
    V = sigma / np.sqrt(2) * (np.random.randn(M, L) + 1j * np.random.randn(M, L))
    return V



def new_gen_codeword(A, L=2, M=2):

    codeword_matrix = np.random.choice(A, size=(L, M), replace=True)
    
    return codeword_matrix
