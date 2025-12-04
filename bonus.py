import numpy as np
from itertools import product
import matplotlib.pyplot as plt

from decodage_bib import *
from decodage_ML import decodage_ml
from decodage_ML import simulate_pe_ml

A = qpsk_alphabet()

def generate_S(A, L, N=2):
    idx = np.random.randint(0, len(A), size=(L, N))
    S = A[idx]
    return S

def decodage_bonus(S, Y):
    S_conj = S.conj().T
    Z = Y.conj().T

    G_hat = np.linalg.pinv(S_conj @ S) @ S_conj @ Z

    H_hat = G_hat.conj().T
    return H_hat

def simulate_bonus(snr_db, L, n_trials=1000):
    n_symbol_errors = 0
    n_total_symbols = n_trials * 2 * 2 

    for _ in range(n_trials):
        H = generate_channel(N=2, M=2)
        S = generate_S(A, L, N=2)

        S_conj = S.conj().T 
        V_train = generate_noise(M=2, L=L, snr_db=snr_db)
        Y_train = H @ S_conj + V_train 

        H_hat = decodage_bonus(S, Y_train)

        X = generate_codeword(A)
        V = generate_noise(M=2, L=2, snr_db=snr_db)
        Y = H @ X + V

        X_hat = decodage_ml(H_hat, Y, A)

        n_symbol_errors += np.sum(X != X_hat)

    pe = n_symbol_errors / n_total_symbols
    return pe

snr_dbs = np.arange(0, 21, 2)
L_list = [3, 5, 10, 20]

# Courbe référence : canal parfait
pes_perfect = []
for snr in snr_dbs:
    pe = simulate_pe_ml(snr_db=snr, n_trials=1000)
    pes_perfect.append(pe)

plt.figure()

for L in L_list:
    pes_L = []
    for snr in snr_dbs:
        pe = simulate_bonus(snr_db=snr, L=L, n_trials=1000)
        pes_L.append(pe)
    plt.semilogy(snr_dbs, pes_L, marker='o', label=f"L={L}")

plt.semilogy(snr_dbs, pes_perfect, linestyle='--', color='k', label="Canal parfait")
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.xlabel("SNR (dB)")
plt.ylabel("Probabilité d'erreur symbole $P_e$")
plt.title("V-BLAST – ML avec canal estimé vs canal parfait")
plt.legend()
plt.show()
