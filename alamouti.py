import matplotlib.pyplot as plt
import numpy as np
from decodage_ML import *

it_per_SNR = 10000
nbr_val_SNR = 40

# utilisation de l'alphabet QPSK
A = qpsk_alphabet()


def encode_alamouti(x1, x2):
    X = np.array([[x1, -(x2.conj())],
                [x2, x1.conj()]])
    return X




def decode_alamouti(Y, H):
    """
    Décodage ML exact pour l’Alamouti 2xM.
    Y : matrice Mx2
    H : matrice Mx2
    """

    # Séparer Y et H
    y1 = Y[:, 0].T
    y2 = Y[:, 1].T
    h1 = H[:, 0].T
    h2 = H[:, 1].T

    # Calculs intermédiaires du décodage ML
    z1 = np.vdot(h1, y1) + np.vdot(y2, h2)  # h1* · y1 + y2* · h2
    z2 = np.vdot(h2, y1) - np.vdot(y2, h1)  # h2* · y1 - y2* · h1   

    # norme Frobenius de H
    norm_H_sq = np.sum(np.abs(H)**2)


    def ml_decision(z):
        best = A[0]
        best_val = 1e9
        for a in A:
            val = np.abs(z - norm_H_sq * a)**2
            if val < best_val:
                best_val = val
                best = a
        return best

    x1_hat = ml_decision(z1)
    x2_hat = ml_decision(z2)

    return [x1_hat, x2_hat]


def main():
    
    snrs = np.linspace(-5, 22, nbr_val_SNR)
    plt.figure()
    
    # Simulation des Alamoutis
    for M in [2, 4, 8]:
        print(f"M={M}")
        H = generate_channel(N=2, M=M)
        Pe = np.zeros_like(snrs)
        
        for snr in range(len(snrs)):
            
            success = 0
            
            for _ in range(it_per_SNR):
            
                # Génération du signal reçu
                symboles = np.random.choice(A, size=2, replace=True)
                X = encode_alamouti(symboles[0], symboles[1])
                V = generate_noise(M=M, L=2, snr_db=snrs[snr])
                Y = H @ X + V
                
                # Décodage par max de vraissemblance simplifie
                symboles_recus = decode_alamouti(Y, H)
                
                # Verifier si symboles recus == symboles
                for i in range(len(symboles_recus)):
                    if symboles_recus[i] == symboles[i]:
                        success += 1
                   
            pe = (2*it_per_SNR - success)/(2*it_per_SNR)
            Pe[snr] = pe
            
            if pe < 10**(-4):
                break
            
        plt.semilogy(snrs, Pe, label=f"Alamouti (M={M})")
    
    
    
    # Simulation des V-Blasts
    for M in [2, 4, 8]:
        print(f"M={M}")
        Pe = np.zeros_like(snrs)
        
        for snr in range(len(snrs)):
            pe = simulate_pe_ml(snrs[snr], M=M)
            Pe[snr] = pe
            if pe < 10**(-5):
                break
            
        plt.semilogy(snrs, Pe, label=f'V-BLAST (M={M})')
    
    
    
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.xlabel("SNR (dB)")
    plt.ylabel("Probabilité d'erreur symbole $P_e$")
    plt.title("Comparaison des performances des codes Alamouti et V-BLAST")
    plt.legend()
    plt.show()
    






if __name__ == "__main__":
    main()