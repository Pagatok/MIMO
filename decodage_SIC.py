import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from tqdm import tqdm
from decodage_bib import *



A = qpsk_alphabet()


def get_QconjR(H):

    Q, R = np.linalg.qr(H)
    
    return Q.conj().T, R



def decodage_SIC(Z, R, A):
    
    N, _ = R.shape
    
    def decode_first_item(l):
        best_norme = np.inf
        xnl_hat = None
        
        for z in A:
            dist = Z[-1, l] - R[-1, -1]*z
            norme = np.linalg.norm(dist)**2
            
            if norme < best_norme:
                best_norme = norme
                xnl_hat = z
        
        return xnl_hat


    def decode_col(l):
    
        Xl_hat = np.zeros((N, 1), dtype=complex)
        
        for n in range(N-1, -1, -1):
            
            best_norme = np.inf
            xnl_hat = None
            
            Interference_I = 0
            for k in range(n + 1, N): 
                Interference_I += R[n, k] * Xl_hat[k]
            
            
            for z in A:
                
                terme_Rnn_z = R[n, n] * z
                
                dist = Z[n, l] - (Interference_I + terme_Rnn_z) 
                norme = np.linalg.norm(dist)**2
                    
                if norme < best_norme:
                    best_norme = norme
                    xnl_hat = z
                    
            Xl_hat[n] = xnl_hat
            
        return Xl_hat

    X = np.zeros((2, 2), dtype=complex)
    
    for l in range(2):
        X[:, l] = decode_col(l).flatten()
    
    return X
    


def simulate_pe_sic(snr_db, n_trials=5000):
    n_symbol_errors = 0
    n_total_symbols = n_trials * 2 * 2 

    for _ in range(n_trials):
        H = generate_channel(N=2, M=2)
        Qstar, R = get_QconjR(H)
        X = generate_codeword(A)
        V = generate_noise(snr_db=snr_db)
        Y = H @ X + V
        Z = Qstar @ Y

        X_hat = decodage_SIC(Z, R, A)

        n_symbol_errors += np.sum(X != X_hat)

    pe = n_symbol_errors / n_total_symbols
    return pe
            
    
if __name__ == "__main__":
    
    # H = generate_channel(N=2, M=2)
    # Qstar, R = get_QconjR(H)
    # X = generate_codeword(A)
    # V = generate_noise()
    # Y = H @ X 
    # Z = Qstar @ Y
    # X_hat = decodage_SIC(Z, R, A)
    
    # print(X)
    # print(X_hat)

    snr_dbs = np.linspace(0, 21, 20)
    pes = []

    for snr in tqdm(snr_dbs):
        pe = simulate_pe_sic(snr_db=snr, n_trials=5000)
        pes.append(pe)

    plt.figure()
    plt.semilogy(snr_dbs, pes)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.xlabel("SNR (dB)")
    plt.ylabel("Probabilité d'erreur symbole $P_e$")
    plt.title("Performance du décodeur SIC (QPSK)")
    plt.show()