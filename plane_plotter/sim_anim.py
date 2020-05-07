import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from tools.plane_animation import PlaneAnimation

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
sim_X = np.load('state_array_1.npy')
sim_U = np.load('ctrl_array_1.npy')*np.pi/180.0
save_vid = True

my_plane = PlaneAnimation(ax, sim_X, sim_U, keep_centred=False)

# Creating the Animation object
plane_animation = animation.FuncAnimation(fig, func=my_plane.animate, frames=sim_X.shape[0], init_func=my_plane.first_frame,
                                          interval=50, blit=False, repeat=False)
if save_vid:
    writer = animation.FFMpegWriter(fps=30, codec="h264", extra_args=["-crf","23"])
    plane_animation.save('vid/sim_anim_fullframe.mp4', writer=writer)

plt.show()