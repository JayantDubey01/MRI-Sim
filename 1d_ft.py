import numpy as np
from matplotlib import pyplot as plt


def rect2exp(vec):

    # Extract real and imaginary parts
    R = vec.real
    I = vec.imag

    # Convert to exponential form: Ae^jtheta
    A = np.sqrt((R**2 + I**2))
    theta = np.arctan2(R,I)

    return A, theta

def dft(data,K):
    DFT_coeffs = np.zeros(K,dtype=complex)
    N = np.size(data)
    exponent = -1j * 2*np.pi   
    for k in range(K):
        for x in range(N):
            DFT_coeffs[k] += (1/np.sqrt(N))*data[x]*np.exp(exponent * (k*x)/N)

    return DFT_coeffs

def ift(coeffs,N):
    signal = np.zeros(N,dtype=complex)
    K = np.size(coeffs)
    exponent = 1j * 2 * np.pi
    for x in range(N):
        for k in range(K):
            signal[x] += (1/np.sqrt(N))*coeffs[k]*np.exp(exponent * (k*x)/N)
    
    return signal

def generate_squarewave(N,freq):
    data = np.empty(N)
    flag = False

    for i in range(N):
        if i%freq == 0:
            flag = ~flag
        
        if flag:
            data[i] = 0
        else:
            data[i] = 1
    
    return data

def generate_wave(N,phase_add):
    data = np.empty(N)
    A = 1
    for i in range(N):
        theta = (2*np.pi/N)*i
        data[i] = A*(np.cos(theta + phase_add) + 1j * np.sin(theta + phase_add))

    return np.abs(data)

if __name__=="__main__":
    
    N = 100
    K = 10
    flag = 0
    #data_complex = np.empty(N)
    data_complex = np.empty(N,dtype=np.float64)
    
    # Generate square wave
    #data = generate_squarewave(N,freq=25)

    data_complex = generate_wave(N,(np.pi))
    plt.figure()
    plt.scatter(np.linspace(0,N,N),data_complex)
    plt.show()

    f = 0
    for k in range(2,K+2,1):
        f = ~f
        if f:
            data_complex+=generate_wave(N,(np.pi/k))
        else:
            data_complex-=generate_wave(N,(np.pi/k))

        plt.scatter(np.linspace(0,N,N),data_complex)
        plt.show()

    
    # Compute DFT from signal
    dft = dft(data_complex,50)
    recon = ift(dft,N)

    x = np.linspace(0,N,N)
    #plt.plot(x,data)
    plt.plot(x,recon)
    plt.scatter(x,data_complex)
    plt.show()
    
