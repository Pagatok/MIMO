import matplotlib.pyplot as plt
import numpy as np
from decodage_ML import *

it_per_SNR = 1000
nbr_val_SNR = 10

# utilisation de l'alphabet QPSK
A = qpsk_alphabet()


def encode_alamouti(x1, x2):
        X = np.array([[x1, -(x2.conj())],
                    [x2, x1.conj()]])
        return X




def decode_aloumati(Y, H):
    
    def max_likely(H, zk):
        likely = 2000.0
        s = A[0]
        for a in A:
            temp = zk - (np.linalg.norm(H, 'fro')**2)*a
            val = np.sum(np.abs(temp)**2)
            if val < likely:
                likely = val
                s = a
        return s
    
    y1 = Y[:, 0:1]  
    y2 = Y[:, 1:2]  
    h1 = H[:, 0:1]  
    h2 = H[:, 1:2]  
    
    z1 = h1.conj() * y1 + y2.conj() * h2
    z2 = h2.conj() * y1 - y2.conj() * h1
    
    x1 = max_likely(H, z1)
    x2 = max_likely(H, z2)
    
    return [x1, x2]
    




def main():
    
    plt.figure()
    
    for M in [2, 4, 8]:
        print(f"M={M}")
        
        Pe = []
        
        for snr in np.linspace(1, 21, nbr_val_SNR):
            
            success = 0
            
            for j in range(it_per_SNR):
            
                # Génétaion du signal reçu
                symboles = np.random.choice(A, size=2, replace=True)
                X = encode_alamouti(symboles[0], symboles[1])
                H = generate_channel(N=2, M=M)
                V = generate_noise(M=M, L=2, snr_db=snr)
                Y = H @ X + V
                
                # Décodage par max de vraissemblance simplifie
                symboles_recus = decode_aloumati(Y, H)
                
                # Verifier si symboles recus == symboles
                for i in range(len(symboles_recus)):
                    if symboles_recus[i] == symboles[i]:
                        success += 1
                        
            Pe.append((2*j - success)/j)
            
        plt.plot(np.linspace(1, 21, nbr_val_SNR), Pe, label=f"M={M}")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()