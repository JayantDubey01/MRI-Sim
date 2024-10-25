import numpy as np
from numpy import linalg as LA

# Start with an initial condition of M and B0, then calculate the time-evolution of M over a range 0->T

T1, T2, gamma = [0.33, 0.04, 42000000]
Bx, By, Bz = [0,0,1]

A = np.array([[-1/T2, gamma*Bz, -gamma*By], 
              [-gamma*Bz, -1/T2, gamma*Bz], 
              [gamma*By, -gamma*Bx, -1/T1]])



eigenvalues, eigenvectors = LA.eig(A)

Mx0, My0, Mz0 = [1,0,0.1]
M0 = np.array([Mx0, My0, Mz0])
C = LA.solve(eigenvectors.T, M0)


'''
print(A)
print(eigenvalues)
print(eigenvectors)
print(C)
'''

# Iterate over the eigenvector list
time=np.linspace(0,2,1000)

def M_t(t):
    M_t = np.zeros(3, dtype=complex)  # Initialize the solution vector M(t)
    for i in range(3):
        eig_real = np.real(eigenvalues[i])
        eig_imag = np.imag(eigenvalues[i])
        exp_term = np.exp(eig_real * t) * (np.cos(eig_imag * t) + 1j * np.sin(eig_imag * t))
        M_t += C[i] * exp_term * eigenvectors[:, i]
    print(f"{t:.3f}s: [{M_t[0]:.4g}, {M_t[1]:.4g}, {M_t[2]:.4g}]")
    return M_t

M_at_t = []

for t in time:
    M_at_t.append(M_t(t))
    #print(M_at_t)

'''
T1, T2, gamma = sp.symbols('T1 T2 gamma')
Bx, By, Bz = sp.symbols('Bx By Bz')
lambda_ = sp.symbols('lambda')

A = sp.Matrix([[-1/T2, gamma*Bz, -gamma*By], 
              [-gamma*Bz, -1/T2, gamma*Bz], 
              [gamma*By, -gamma*Bx, -1/T1]])

char_poly = A.charpoly(lambda_)

eigenvalues = sp.solve(char_poly.as_expr(), lambda_)

numerical_eigenvalues = [ev.subs({T1: 0.33, T2: 0.04, Bx: 0, By: 0, Bz: 3, gamma:128000000}).evalf() for ev in eigenvalues]
print(numerical_eigenvalues)

#print("Numerical Eigenvalues:", numerical_eigenvalues)

# Step 1: Calculate eigenvectors
eigenvectors = A.eigenvects()

# Step 2: Extract eigenvectors and substitute parameter values
eigenvector_list = []
for val, mult, vecs in eigenvectors:
    for vec in vecs:
        # Debug: Print the original eigenvector
        print(f"Original Eigenvector: {vec}")
        
        # Substitute the values for each eigenvector
        numeric_vec = vec.subs({T1: 0.33, T2: 0.04, Bx: 0, By: 0, Bz: 3, gamma: 128000000})
        
        # Debug: Print the substituted eigenvector before evaluation
        print(f"Substituted Eigenvector: {numeric_vec}")
        
        numeric_vec_eval = numeric_vec.evalf()  # Evaluate numerically
        eigenvector_list.append(numeric_vec_eval)

# Print final numerical eigenvectors
print("Numerical Eigenvectors:")
for ev in eigenvector_list:
    print(ev)

# Assuming t has been initialized and C1, C2, C3 are defined
M_t = np.array([0, 0, 0], dtype=complex)  # Initialize the resulting magnetization vector
C1, C2, C3 = M_t

# Iterate over the eigenvector list
time=np.linspace(0,30,30)

# Create an array to hold results
M_t_list = []  # Store the time-evolved magnetization vectors

# Iterate over the time steps
for t in time:
    M_t = np.array([0, 0, 0], dtype=complex)  # Reset for each time step
    
    # Iterate over the eigenvector list
    for eigval, multi, eigvecs in eigenvector_list:
        
        
        # Accumulate contributions from each eigenvector
        M_t += C1 * np.exp(numerical_eigenvalues[0] * t) * eigvec_numeric[0]  # First eigenvalue and vector
        M_t += C2 * np.exp(numerical_eigenvalues[1] * t) * eigvec_numeric[1]  # Second eigenvalue and vector
        M_t += C3 * np.exp(numerical_eigenvalues[2] * t) * eigvec_numeric[2]  # Third eigenvalue and vector

    M_t_list.append(M_t)  # Store the result for this time step

# Convert to a numpy array for easier manipulation if needed
M_t_array = np.array(M_t_list)

# Print or analyze the time-evolved magnetization vectors
print("Time-evolved magnetization vectors:", M_t_array)
'''





    