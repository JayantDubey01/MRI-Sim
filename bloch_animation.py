import numpy as np
from numpy import linalg as LA

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# ---------------------- Calculate time evolution ------------------

def compute_eigen(molecular_parameters, B, M):
    # Biological parameters
    T1, T2, gamma = molecular_parameters

    # B_eff unpacked to x,y,z components
    Bx, By, Bz = B

    # The magnetic moment's current position
    Mx, My, Mz = M

    A = np.array([[-1/T2, gamma*Bz, -gamma*By], 
                [-gamma*Bz, -1/T2, gamma*Bx], 
                [gamma*By, -gamma*Bx, -1/T1]])
    
    
    print(A)

    eigenvalues, eigenvectors = LA.eig(A)
    M0 = np.array([Mx, My, Mz])
    C = LA.solve(eigenvectors.T, M0)

    # Compute M0 from eigenvectors and C
    a = C[0] * eigenvectors[0]
    b = C[1] * eigenvectors[1]
    c = C[2] * eigenvectors[2]

    # Verify the correct C values were computed by reconstructing initial M
    arr = np.array(a + b + c).real.astype(float)
    #print(arr)
    #print(M)

    # normalize eigenvectors
    eigenvectors[:, 0] = eigenvectors[:, 0] / np.linalg.norm(eigenvectors[:, 0])
    eigenvectors[:, 1] = eigenvectors[:, 1] / np.linalg.norm(eigenvectors[:, 1])
    eigenvectors[:, 2] = B

    #print("val: ", eigenvalues)
    #print("vector: ", eigenvectors)

    return eigenvalues, eigenvectors, C

    
def compute_trajectory(time,molecular_parameters, B, M):
    # Iterate over the eigenvector list
    M_at_t = np.zeros((len(time), 3), dtype=float)

    eigenvalues, eigenvectors, C = compute_eigen(molecular_parameters, B, M) # Compute eigenvalue and vectors

    for idx, t in enumerate(time):
        M_t = np.zeros(3, dtype=complex)  # Initialize the solution vector M(t)
        for i in range(3):
            exp_term = np.exp(eigenvalues[i] * t)
            #print("exp term: ", exp_term)
            M_t += C[i] * exp_term * eigenvectors[:, i]
            #print("M_t: ", M_t)

            #print(f"{t:.3f}s: exp term = {exp_term}, M_t: [{M_t[0]:.4g}, {M_t[1]:.4g}, {M_t[2]:.4g}]")
        #print(f"{t:.3f}s: [{M_t[0]:.4g}, {M_t[1]:.4g}, {M_t[2]:.4g}]")
        M_at_t[idx] = np.real(M_t)
    
    return M_at_t

# --------------- Parameters -----------------------

# MRI
B0 = np.array([0,0,1])
molecular_parameters = np.array([0.8,0.5,42000000])
gamma = molecular_parameters[2]

## RF, Lab frame B_RF = B1*cos(w_rf*t)*x_hat - B1*sin(w_rf*t)*y_hat

## On-resonance B_eff = [B1,0, B0 - w_0/gamma] = [B1,0,B0 - gamma*B0/gamma] = [B1,0,0]
## In order for full precession around B1 strictly, a transition time is needed to fully transition. A 90 degree transition takes tau = pi/2*gamma*B1
# w_0 = gamma*B0
# w_1 = gamma*B1
# Flip-angle: alpha = w_1*tau = gamma*B1*tau

w_rf = gamma * B0[2]
f_fict = w_rf/gamma
#print("f_fict: ", f_fict)
B1 = 1

w_1 = gamma*B1
tau = (1/4*gamma*B1)/100000000  # RF Pulse Duration for 90deg flip angle
#print("tau: ", tau)
alpha = w_1 * tau

B_RF = np.array([B1,1,-f_fict])  
print("B_RF: ", B_RF)

# Effective B-field
B_eff = B_RF + B0
print("B_eff: ", B_eff)

M_initial = [0.1,0.1,-1]

time_equilibrium=np.linspace(0,0.1,100)
time_pulse = np.linspace(0,tau,100)

print("Thermal Equilibrium: ")
M_at_t = compute_trajectory(time_equilibrium,molecular_parameters, B0, M_initial)
#print("New M: ", M_at_t[len(time_equilibrium)-1])

print()
print("RF turned on")
M_rf_pulse = compute_trajectory(time_pulse,molecular_parameters, B_eff, M_at_t[len(M_at_t)-1])

M_after = compute_trajectory(time_equilibrium,molecular_parameters,B0,M_rf_pulse[len(M_rf_pulse)-1])

total_trajectory = np.concatenate((M_at_t, M_rf_pulse,M_after))

# ----------------- Animate ----------------------

def get_arrow(frame):
    x,y,z = 0,0,0
    #u,v,w = M_at_t[frame]
    #u,v,w = M_rf_pulse[frame]
    u,v,w = total_trajectory[frame]

    # Add if-conditional where the range t0->t1 is thermal equilibrium 
    # Then t1-te is the excitation time
    # Then te->tr is the relaxation time

    return x,y,z,u,v,w

# Update function for animation
def update(frame):
    global quiver, trail_x, trail_y, trail_z

    # Get new arrow coordinates
    x, y, z, u, v, w = get_arrow(frame)
    
    # Remove the old arrow
    quiver.remove()

    # Create a new arrow at the current position
    quiver = ax.quiver(x, y, z, u, v, w, color='b')

    # Append the new arrow head coordinates to the trail
    trail_x.append(x + u)
    trail_y.append(y + v)
    trail_z.append(z + w)

    # Plot the trail (previous arrow head positions)
    ax.plot(trail_x, trail_y, trail_z, color='r')


if __name__=="__main__":

    fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-2, 2) 

    # Lists to store the arrow head coordinates (for the trail)
    trail_x, trail_y, trail_z = [], [], []
    quiver = ax.quiver(*get_arrow(0))
    B0 = ax.quiver(0,0,0,0,0,-4,color='g')

    # Concatenate the two time ranges
    total_time = np.concatenate((time_equilibrium, time_equilibrium[-1] + time_pulse))

    # The update argument takes elements of frame as argument, then plots
    ani = FuncAnimation(fig, update, len(total_time), interval=20)
    plt.show()

    ## Plot x,y,z values individually
    #plt.plot(time_, M_at_t[:,1])
    #plt.show()
    