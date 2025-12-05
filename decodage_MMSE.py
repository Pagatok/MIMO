import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from decodage_ZF import simulate_pe_zf
from decodage_bib import *
from tqdm import tqdm

A = qpsk_alphabet()
H = generate_channel()
X = generate_codeword(A)
V = generate_noise()
Y = H @ X + V

def decodage_mmse(H, Y, A, snr_db):
    N = H.shape[1]
    sigma2 = 10 ** (-snr_db / 10)

    H_herm = H.conj().T
    F_mmse = np.linalg.inv(H_herm @ H + sigma2 * np.eye(N)) @ H_herm

    Z = F_mmse @ Y
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

def simulate_pe_mmse(snr_db, n_trials=1000):
    n_symbol_errors = 0
    n_total_symbols = n_trials * 2 * 2 

    for _ in range(n_trials):
        H = generate_channel(N=2, M=2)
        X = generate_codeword(A)
        V = generate_noise(M=2, L=2, snr_db=snr_db)
        Y = H @ X + V

        X_hat = decodage_mmse(H, Y, A, snr_db)

        n_symbol_errors += np.sum(X != X_hat)

    pe = n_symbol_errors / n_total_symbols
    return pe


def vsZF():
    
    snr_dbs = np.linspace(-5, 20, 20)
    
    pes1 = []
    pes2 = []
    for snr in tqdm(snr_dbs):
        pe = simulate_pe_zf(snr_db=snr, n_trials=1000)
        pes1.append(pe)
        pe = simulate_pe_mmse(snr_db=snr, n_trials=1000)
        pes2.append(pe)
        
    plt.figure()
    plt.semilogy(snr_dbs, pes1, label='ZF')
    plt.semilogy(snr_dbs, pes2, label='MMSE')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.xlabel("SNR (dB)")
    plt.ylabel("Probabilité d'erreur symbole $P_e$")
    plt.title("Comparaison des codes MMSE et ZF")
    plt.legend()
    plt.show()
        
    
def classic_sim_MMSE():
    pe_10 = simulate_pe_mmse(snr_db=10, n_trials=1000)
    print("P_e (ML) à 10 dB ≈", pe_10)

    snr_dbs = np.linspace(-5, 20, 20)
    pes = []

    for snr in snr_dbs:
        pe = simulate_pe_mmse(snr_db=snr, n_trials=1000)
        print(f"SNR = {snr} dB, Pe_MMSE ≈ {pe}")
        pes.append(pe)

    plt.figure()
    plt.semilogy(snr_dbs, pes)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.xlabel("SNR (dB)")
    plt.ylabel("Probabilité d'erreur symbole $P_e$")
    plt.title("Performance du décodeur MMSE pour V-BLAST 2x2 (QPSK)")
    plt.show()




if __name__ == "__main__":

    vsZF()