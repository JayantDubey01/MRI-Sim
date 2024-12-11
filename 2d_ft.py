import numpy as np
from matplotlib import pyplot as plt

# NOTE: linspace is (start, end, number of points inbetween) 

def rect2exp(vec):

    # Extract real and imaginary parts
    R = vec.real
    I = vec.imag

    # Convert to exponential form: Ae^jtheta
    A = np.sqrt((R**2 + I**2))
    theta = np.arctan2(R,I)

    return A, theta

def exp2rect(A,theta):
    R = A*(np.cos(theta))
    I = A*(np.sin(theta))

    return R + (1j * I)

def dft(data,K,L):
    DFT_coeffs = np.zeros((K,L),dtype=complex)
    exponent = -1j * 2*np.pi
    N, M = data.shape  
    for k in range(K):
        for l in range(L):
            for x in range(N):
                for y in range(M):
                    DFT_coeffs[k,l] += data[x,y]*np.exp(exponent * ((k*x)/N + (l*y)/M))
    
    return DFT_coeffs

def ift(coeffs,shape):
    N, M = shape
    flag = 2

    # Check if its 1D or 2D IFT
    try:
        K = coeffs.shape[0]
        L = coeffs.shape[1]
        flag = 2
    except:
        K = coeffs.shape[0]
        flag = 1
    
        
    # 2D IFT
    if flag==2:
        print("Detected 2D dataset, performing 2D IFT")
        exponent = 1j * 2 * np.pi
        signal = np.zeros((N,M),dtype=complex)
        for x in range(N):
            for y in range(M):
                for k in range(K):
                    for l in range(L):
                        signal[x,y] += coeffs[k,l]*np.exp(exponent * (((k/K)*x) + ((l/L)*y)))
        
        return np.real(signal) * 1/(K * L)
    
    # 1D IFT
    elif flag==1:
        print("Detected 1D dataset, performing 1D IFT")
        signal = np.zeros((N,M),dtype=complex)
        exponent = 1j * 2 * np.pi
        for x in range(N):
            for y in range(M):
                for k in range(K):
                    signal[x,y] += (1/np.sqrt(N))*coeffs[k]*np.exp(exponent * (k*x)/N)
        
        return signal


# Simulate water density varied by position
def generate_image(N,freq_x,freq_y):
    image = np.zeros((N,N),dtype=np.float32)
    flag_x = False
    flag_y = False

    # Generate 2D Square Wave
    for x in range(N):
        flag_y = False
        
        if x%freq_x==0:
            flag_x = ~flag_x
            
        if flag_x:
            for y in range(N):
                if y%freq_y == 0:
                    flag_y = ~flag_y
                
                if flag_y:
                    image[x,y] = 1
                else:
                    image[x,y] = 0.5
        else:    
            for y in range(N):
                if y%freq_y == 0:
                    flag_y = ~flag_y
                
                if flag_y:
                    image[x,y] = 0.5
                else:
                    image[x,y] = 1
    
    return image

# Apply a gradient over water density map, but the gradient would shift the phase, not the magnitude of the signal, 
# so the image has to be complex
# What would happen if we took the IFT of this complex image, with and without a gradient? 

def apply_x_intensity_gradient(image,gradmag,freq):
    N, M = image.shape

    for y in range(0,M,freq):
        image[:,y:y+freq] = image[:,y:y+freq] - ((gradmag/N)*y)
    
    return image

def apply_y_intensity_gradient(image,gradmag,freq):
    N, M = image.shape

    for x in range(0,N,freq):
        #image[:,y:y+freq] = image[:,y:y+freq] - (grad*y)
        image[x:x+freq,:] = image[x:x+freq,:] - ((gradmag/N)*x)   
    
    return image


# NOTE: The added phase angle can be calculated by calculating the time-varying Bloch equation over some interval time. 
# My grad value 

def apply_x_gradient(image,grad,period):
    N, M = image.shape 
    complex_image = image.astype(complex)

    for y in range(0,M,period):

        # Convert original image data to exponential form: Ae^jtheta
        A, theta = rect2exp(complex_image[:,y:y+period])

        # Calculate phase increment, starts at 0 -> +y
        phase_add = (2*np.pi)*y*(grad/N)
        theta_new = theta + phase_add   # add angle 

        # Rect form signal with new added phase
        shifted_signal = exp2rect(A,theta_new)
        
        # Replace orignial phase with new phase
        complex_image[:,y:y+period] = shifted_signal
    
    return complex_image

def apply_y_gradient(image,grad,period):
    N, M = image.shape 
    complex_image = image.astype(complex)

    for x in range(0,N,period):

        # Convert original image data to exponential form: Ae^jtheta
        A, theta = rect2exp(complex_image[x:x+period,:])

        # Calculate phase increment
        phase_add = (2*np.pi)*x*(grad/N)
        theta_new = theta + phase_add   # add angle 


        # Rect form signal with new added phase
        shifted_signal = exp2rect(A,theta_new)
        
        # Replace orignial phase with new phase
        complex_image[x:x+period,:] = shifted_signal
    
    return complex_image


def acquire_signal_interval_gradient(data,interval,gradrange):

    N, M = data.shape

    # Specifies range 
    Gx = np.ceil(N*gradrange)
    Gy = np.ceil(M*gradrange)

    print("Grad range: " + "[-" + str(Gx) + ", " + str(Gx) + "]")

    gradient_trajectory_x = np.linspace(-Gx,Gx,N)
    gradient_trajectory_y = np.linspace(-Gy,Gy,M)

    print("Num of samples: " + str(gradient_trajectory_x.shape[0]))

    kspace = np.zeros((N, M),dtype=complex)

    for k in range(0,len(gradient_trajectory_y),interval):
        for l in range(0,len(gradient_trajectory_x),interval):

            # Apply gradient, 
            complex_y_data = apply_y_gradient(data,grad=gradient_trajectory_y[k],period=1)
            complex_data = apply_x_gradient(complex_y_data,grad=gradient_trajectory_x[l],period=1)
            
            # Integral over all the spins after gradient is applied, stored as kspace data point
            kspace[k,l] = np.sum(complex_data)
    
    return kspace


def acquire_signal(data):

    N, M = data.shape
    #data = apply_1D_intensity_gradient(data,grad=15,freq=15)

    # k space range is [-N/2, N/2] so there is a total of N points
    Gx = int(N/2)
    Gy = int(M/2)

    # Bandwidth of signal 
    bandwidth_x = np.linspace(-Gx, Gx,N)
    bandwidth_y = np.linspace(-Gy, Gy,M)

    kspace = np.zeros((N, M),dtype=complex)

    # Apply gradient on image in x, y direction then sum the whole image and store the complex signal as a k-space coefficient
    for k,grad_y in enumerate(bandwidth_y):
        for l,grad_x in enumerate(bandwidth_x):

            # Apply gradient, 
            complex_y_data = apply_y_gradient(data,grad=grad_y,period=1)
            complex_data = apply_x_gradient(complex_y_data,grad=grad_x,period=1)
            
            # Integral over all the spins after gradient is applied, stored as kspace data point
            kspace[k,l] = np.sum(complex_data)
    
    return kspace

if __name__=="__main__":
    
    # Generate square wave
    data = generate_image(N=60,freq_x=45,freq_y=45)
    N = data.shape[0]
    
    # Fill-in kspace
    #kspace = acquire_signal
    grad_max = 1/2
    print(grad_max)
    kspace = acquire_signal_interval_gradient(data,interval=1,gradrange=grad_max)

    # Compute DFT from raw signal
    #dft = dft(data,data.shape[0],data.shape[1])

    #recon_signal = ift(kspace,kspace.shape)
    recon_signal = np.fft.ifft2(kspace)
    recon_signal = np.abs(recon_signal)

    #recon_dft = np.fft.ifft2(dft)
    #recon_dft = np.abs(recon_dft)

    
    plt.figure("IFT over gradient vs. DFT")

    plt.subplot(131)
    plt.title("Original")
    plt.imshow(data,cmap='gray')

    plt.subplot(132)
    plt.title("K space")
    plt.imshow(np.log(1+np.abs(kspace)),cmap='gray')

    plt.subplot(133)
    plt.title("K space Recon")
    plt.imshow(recon_signal,cmap='gray')
    plt.show()
    
    '''
    plt.figure()
    plt.subplot(121)
    plt.title("DFT over image")
    #plt.imshow(np.log(1+np.abs(dft)),cmap='gray')
    
    plt.subplot(122)
    plt.title("DFT Recon")
    #plt.imshow(recon_dft,cmap='gray')
    plt.show()
    

    plt.figure("Gradient with intensity")    
    data_x = apply_x_intensity_gradient(data.copy(),gradmag=10,freq=2)
    data_y = apply_y_intensity_gradient(data.copy(),gradmag=10,freq=2)
    plt.subplot(131)
    plt.imshow(data_x,cmap='gray')
    plt.subplot(132)
    plt.imshow(data_y,cmap='gray')
    plt.subplot(133)
    plt.imshow(data,cmap='gray')
    plt.show()
    
    '''

        

    
