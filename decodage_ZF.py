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

A = qpsk_alphabet()
H = generate_channel()
H_pinv = np.linalg.pinv(H)
X = generate_codeword(A)
V = generate_noise()
Y = H @ X + V
Z = H_pinv @ Y

def decodage_zf(H, Y, A):
    H_pinv = np.linalg.pinv(H)
    Z = H_pinv @ Y
    X_hat = np.zeros_like(Z, dtype=complex)

    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            z_ij = Z[i, j]

            best_dist = np.inf
            best_symbol = None

            for a in A:
                d = np.abs(z_ij - a)**2
                if d < best_dist:
                    best_dist = d
                    best_symbol = a

            X_hat[i, j] = best_symbol

    return X_hat

def simulate_pe_zf(snr_db, n_trials=1000):
    n_symbol_errors = 0
    n_total_symbols = n_trials * 2 * 2 

    for _ in range(n_trials):
        H = generate_channel(N=2, M=2)
        X = generate_codeword(A)
        V = generate_noise(M=2, L=2, snr_db=snr_db)
        Y = H @ X + V

        X_hat = decodage_zf(H, Y, A)

        n_symbol_errors += np.sum(X != X_hat)

    pe = n_symbol_errors / n_total_symbols
    return pe

pe_10 = simulate_pe_zf(snr_db=10, n_trials=1000)
print("P_e (ML) à 10 dB ≈", pe_10)

snr_dbs = np.arange(0, 21, 2)
pes = []

for snr in snr_dbs:
    pe = simulate_pe_zf(snr_db=snr, n_trials=1000)
    print(f"SNR = {snr} dB, Pe_ZF ≈ {pe}")
    pes.append(pe)

plt.figure()
plt.semilogy(snr_dbs, pes)
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.xlabel("SNR (dB)")
plt.ylabel("Probabilité d'erreur symbole $P_e$")
plt.title("Performance du décodeur ZF pour V-BLAST 2x2 (QPSK)")
plt.show()