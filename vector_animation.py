import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


def get_arrow(theta):
    x = 0
    y = 0
    z = -2
    u = np.cos(theta)
    v = np.sin(theta)
    w = 2
    return x,y,z,u,v,w

# Update function for animation
def update(frame, angular_frequency):
    global quiver, trail_x, trail_y, trail_z

    theta = frame * angular_frequency

    # Get new arrow coordinates
    x, y, z, u, v, w = get_arrow(theta)
    
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
    B0 = ax.quiver(0,0,-2,0,0,4,color='g')

    angular_frequency = 3
    frames=np.linspace(0,2*np.pi,100)   # Vector of 0-2pi, step of 2pi/100

    # The update argument takes elements of frame as argument, then plots
    ani = FuncAnimation(fig, update, frames, fargs=(angular_frequency,), interval=20)
    plt.show()


