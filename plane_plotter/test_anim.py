import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from plane_animation import PlaneAnimation

save_vid = False

# Set some text properties for nice labels
plt.rc('font', **{'family': 'serif', 'sans-serif': ['Computer Modern Roman']})
plt.rc('text', usetex=True)
fig = plt.figure(figsize=[8,8])
ax = Axes3D(fig)
ax.set_xlim(-2, 10)
ax.set_ylim(-5, 5)
ax.set_zlim(-9.5, 0.5)

ax.invert_zaxis()
ax.invert_yaxis()

ax.set_xlabel(r'$X_e$')
ax.set_ylabel(r'$Y_e$')
ax.set_zlabel(r'$Z_e$')

# Remember: X = [u, v, w, p, q, r, phi, theta, psi, x, y, z]
full_X = np.zeros((50, 12))
full_X[:15,6] = np.linspace(0, np.pi/4, 15)
full_X[15:35,6] = np.linspace(np.pi/4, -np.pi/4, 20)
full_X[35:,6] = np.linspace(-np.pi/4, 0, 15)
full_X[:,9] = np.linspace(0, 10, full_X.shape[0])
full_X[:,10] = 0.5*np.sin(np.linspace(0, 2*np.pi, full_X.shape[0]))

# Remember: U = [lamba_left, lambda_right, delta_rudder]
full_U = np.zeros((50, 3))
full_U[:,0] = np.pi/6*np.sin(np.linspace(0, 2*np.pi, full_U.shape[0]))
full_U[:,1] = np.pi/6*np.sin(np.linspace(0, 2*np.pi, full_U.shape[0]))
full_U[:,2] = np.pi/8*np.sin(np.linspace(0, 2*np.pi, full_U.shape[0]))

my_plane = PlaneAnimation(ax, full_X, full_U, keep_centred=False)

# Creating the Animation object
plane_animation = animation.FuncAnimation(fig, func=my_plane.animate, frames=50, init_func=my_plane.first_frame,
                                          interval=50, blit=False, repeat=False)
if save_vid:
    writer = animation.FFMpegWriter(fps=30, codec="h264", extra_args=["-crf","23"])
    plane_animation.save('vid/test.mp4', writer=writer)

plt.show()