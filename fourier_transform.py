import numpy as np
from scipy.integrate import quad
import matplotlib as plt

def integrand(n,t):
    return np.exp(-2*np.pi*n*t)

def compute_fourier_coeffs(n):
    Cn = np.zeros([n,2],dtype=float)
    for i in range(n):
        integral = quad(integrand,0,0.5,args=(i))
        print(integral)
        Cn[i] = integral
    


if __name__=="__main__":
    
    # Compute Fourier Coefficients
    compute_fourier_coeffs(9)
    
