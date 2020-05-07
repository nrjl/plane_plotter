import numpy as np
from tools.plane import Plane

class PlaneAnimation(object):
    path_line = None

    def __init__(self, ax, X, U=None, keep_centred=False, box_radius=5.0, model='models/default_plane.yaml'):
        self.ax = ax
        self.X = X
        if U is None:
            U = np.zeros((X.shape[0], 3))
        self.U = U
        self.plane = Plane(self.ax, model)
        self.keep_centred = keep_centred
        self.box_radius = box_radius

    def centre_frame(self, i):
        self.ax.set_xlim(self.X[i, 9]-self.box_radius, self.X[i,9]+self.box_radius)
        self.ax.set_ylim(self.X[i, 10] - self.box_radius, self.X[i, 10] + self.box_radius)
        self.ax.set_zlim(self.X[i, 11] - self.box_radius, self.X[i, 11] + self.box_radius)
        self.ax.invert_zaxis()
        self.ax.invert_yaxis()

    def full_flight_frame(self):

        max_range = np.array([self.X[:, 9].max() - self.X[:, 9].min(),
                              self.X[:, 10].max() - self.X[:, 10].min(),
                              self.X[:, 11].max() - self.X[:, 11].min()]).max() / 2.0

        mid_x = (self.X[:, 9].max() + self.X[:, 9].min()) * 0.5
        mid_y = (self.X[:, 10].max() + self.X[:, 10].min()) * 0.5
        mid_z = (self.X[:, 11].max() + self.X[:, 11].min()) * 0.5
        self.ax.set_xlim(mid_x - max_range, mid_x + max_range)
        self.ax.set_ylim(mid_y - max_range, mid_y + max_range)
        self.ax.set_zlim(mid_z - max_range, mid_z + max_range)

        # xlim = (min(self.X[:, 9]), max(self.X[:, 9]))
        # ylim = (min(self.X[:, 10]), max(self.X[:, 10]))
        # zlim = (min(self.X[:, 11]), max(self.X[:, 11]))
        # dl = [l[1]-l[0] for l in [xlim, ylim, zlim]]
        # if
        #
        # self.ax.set_xlim(xlim)
        # self.ax.set_ylim(ylim)
        # self.ax.set_zlim(zlim)
        self.ax.invert_zaxis()
        self.ax.invert_yaxis()

    def first_frame(self):
        self.plane.update(self.X[0], self.U[0])
        if self.path_line is None:
            self.path_line, = self.ax.plot(self.X[:1, 9], self.X[:1, 10], self.X[:1, 11], color='firebrick')
        if self.keep_centred:
            self.centre_frame(0)
        else:
            self.full_flight_frame()

    def animate(self, i):
        self.plane.update(self.X[i], self.U[i])
        self.path_line.set_data(self.X[:i+1, 9:11].T)
        self.path_line.set_3d_properties(self.X[:i+1, 11])
        if self.keep_centred:
            self.centre_frame(i)

