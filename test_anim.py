import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from plane import Plane


class PlaneAnimation(object):
    path_line = None

    def __init__(self, ax, X, U=None):
        self.ax = ax
        self.X = X
        if U is None:
            U = np.zeros((X.shape[0], 3))
        self.U = U
        self.plane = Plane(self.ax, 'models/default_plane.yaml')

    def first_frame(self):
        self.plane.update(self.X[0], self.U[0])
        if self.path_line is None:
            self.path_line, = self.ax.plot([self.X[0,9]], [self.X[0,10]], [self.X[0,11]], color='firebrick')

    def animate(self, i):
        self.plane.update(self.X[i], self.U[i])
        self.path_line.set_data(self.X[:i, 9:11].T)
        self.path_line.set_3d_properties(self.X[:i, 11])


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

full_X = np.zeros((50, 12))
full_X[:15,6] = np.linspace(0, np.pi/4, 15)
full_X[15:35,6] = np.linspace(np.pi/4, -np.pi/4, 20)
full_X[35:,6] = np.linspace(-np.pi/4, 0, 15)
full_X[:,9] = np.linspace(0, 10, full_X.shape[0])
full_X[:,10] = 0.5*np.sin(np.linspace(0, 2*np.pi, full_X.shape[0]))

full_U = np.zeros((50, 3))
full_U[:,0] = np.pi/4*np.sin(np.linspace(0, 2*np.pi, full_U.shape[0]))
full_U[:,1] = np.pi/4*np.sin(np.linspace(0, 2*np.pi, full_U.shape[0]))
full_U[:,2] = np.pi/8*np.sin(np.linspace(0, 2*np.pi, full_U.shape[0]))

my_plane = PlaneAnimation(ax, full_X, full_U)

# Creating the Animation object
plane_animation = animation.FuncAnimation(fig, func=my_plane.animate, frames=50, init_func=my_plane.first_frame,
                                          interval=50, blit=False, repeat=False)
# writer = animation.FFMpegWriter(fps=30, codec="libx264", extra_args=["-crf","23"])
# plane_animation.save('vid/test.mp4', writer=writer)

plt.show()