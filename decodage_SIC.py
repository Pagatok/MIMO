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
        
        return z


    def decode_col(l):
    
        Xl_hat = np.zeros((N, 1), dtype=complex)
        Xl_hat[-1] = decode_first_item(l)
        
        for n in range(N-1, -1, -1):
            
            best_norme = np.inf
            xnl_hat = None
            
            for z in A:
            
                somme = 0
                for k in range(n, N):
                    temp = R[n, k]*Xl_hat[k] - R[n, n]*z
                    somme += temp
                    
                dist = Z[n, l] - somme
                norme = np.linalg.norm(dist)**2
                    
                if norme < best_norme:
                    best_norme = norme
                    xnl_hat = z
                    
            Xl_hat[n] = xnl_hat
            
        return Xl_hat
    
    X = np.zeros(())
    
    return decode_col(0)
    


def simulate_pe_sic(snr_db, n_trials=1000):
    n_symbol_errors = 0
    n_total_symbols = n_trials * 2 * 2 

    for _ in range(n_trials):
        H = generate_channel(N=2, M=2)
        Qstar, R = get_QconjR(H)
        X = generate_codeword(A)
        V = generate_noise()
        Y = H @ X + V
        Z = Qstar @ Y

        X_hat = decodage_SIC(Z, R, A)

        n_symbol_errors += np.sum(X != X_hat)

    pe = n_symbol_errors / n_total_symbols
    return pe
            
    






X_hat_SIC = decodage_SIC(Z, R, A)

print(f"X = {X}\n\nX_hat_SIC = {X_hat_SIC}")