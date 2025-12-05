import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from tqdm import tqdm
from decodage_bib import *
from decodage_MMSE import simulate_pe_mmse
from decodage_ZF import simulate_pe_zf



A = qpsk_alphabet()


def get_QconjR(H):

    Q, R = np.linalg.qr(H)
    
    return Q.conj().T, R



def decodage_SIC(Z, R, A, N, M):
    
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

    X = np.zeros((N, M), dtype=complex)
    
    for l in range(M):
        X[:, l] = decode_col(l).flatten()
    
    return X



def new_SIC(Z, R, A, N, M):
    
    X = np.zeros((N, M), dtype=complex)
    
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
    
    def decode_item(n, l, X):
        best_norme = np.inf
        xnl_hat = None
        
        for z in A:
            somme = 0
            for k in range(n+1, N):
                somme += R[n, k] * X[k, l] 
            dist = Z[n, l] - somme - R[n, n]*z
            norme = np.linalg.norm(dist)**2
            if norme < best_norme:
                best_norme = norme
                xnl_hat = z
        
        return xnl_hat
        
    
    
    for l in range(M):
        X[-1, l] = decode_first_item(l)
        for n in range(N-2, -1, -1):
            X[n, l] = decode_item(n, l, X)

        
    return X
            



 


def simulate_pe_sic(snr_db, n_trials=5000, L=2, M=2):
    n_symbol_errors = 0
    n_total_symbols = n_trials * L * M 

    for _ in range(n_trials):
        H = generate_channel(N=L, M=M)
        Qstar, R = get_QconjR(H)
        X = new_gen_codeword(A, L=L, M=M)
        V = generate_noise(snr_db=snr_db, M=M, L=L)
        Y = H @ X + V
        Z = Qstar @ Y

        X_hat = new_SIC(Z, R, A, L, M)

        n_symbol_errors += np.sum(X != X_hat)

    pe = n_symbol_errors / n_total_symbols
    return pe


def sim_classic():
    snr_dbs = np.linspace(-5, 21, 20)
    pes = []

    for snr in tqdm(snr_dbs):
        pe = simulate_pe_sic(snr_db=snr, n_trials=5000, L=4, M=4)
        pes.append(pe)

    plt.figure()
    plt.semilogy(snr_dbs, pes)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.xlabel("SNR (dB)")
    plt.ylabel("Probabilité d'erreur symbole $P_e$")
    plt.title("Performance du décodeur SIC (QPSK)")
    plt.legend()
    plt.show()
    
    
def vsZF_MMSE(n_trials=10000):
    snr_dbs = np.linspace(-5, 20, 20)
    
    pes1 = []
    pes2 = []
    pes3 = []
    for snr in tqdm(snr_dbs):
        pe = simulate_pe_zf(snr_db=snr, n_trials=n_trials)
        pes1.append(pe)
        pe = simulate_pe_mmse(snr_db=snr, n_trials=n_trials)
        pes2.append(pe)
        pe = simulate_pe_sic(snr_db=snr, n_trials=n_trials)
        pes3.append(pe)
        
    plt.figure()
    plt.semilogy(snr_dbs, pes1, label='ZF')
    plt.semilogy(snr_dbs, pes2, label='MMSE')
    plt.semilogy(snr_dbs, pes3, label='SIC')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.xlabel("SNR (dB)")
    plt.ylabel("Probabilité d'erreur symbole $P_e$")
    plt.title("Performances du code SIC")
    plt.legend()
    plt.show()



# Etude de l'impact de L sur les performances de SIC
def sim_L_impact():
    snr_dbs = np.linspace(-5, 21, 20)
    plt.figure()
    
    for L in [2, 5, 10, 20]:
        
        pes = []

        for snr in tqdm(snr_dbs):
            pe = simulate_pe_sic(snr_db=snr, n_trials=5000, L=4, M=4)
            pes.append(pe)

        plt.semilogy(snr_dbs, pes, label=f"L={L}")
        
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.xlabel("SNR (dB)")
    plt.ylabel("Probabilité d'erreur symbole $P_e$")
    plt.title("Performance du décodeur SIC (QPSK)")
    plt.legend()
    plt.show()

  
    
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

    vsZF_MMSE()