import numpy as np
from rotation_matrices import L_eb, Lz
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import yaml

class SurfacePolygon(object):
    def __init__(self, root_position, root_chord, tip_chord, span, vector, *args, **kwargs):
        self._root_position = np.array(root_position)

        # Normalise vector
        vector = np.array(vector)
        vector = vector / np.linalg.norm(vector)

        # Create polygon vertices
        polygon = np.zeros((4, 3))
        polygon[0] = np.array([0.5*root_chord, 0, 0])
        polygon[1] = vector * span + np.array([0.5*tip_chord, 0, 0])
        polygon[2] = vector * span + np.array([-0.5*tip_chord, 0, 0])
        polygon[3] = np.array([-0.5*root_chord, 0, 0])
        polygon += self._root_position
        self._base_polygon = polygon
        self._actuated_polygon = polygon.copy()

    def set_sweep(self, angle):
        self._actuated_polygon = np.matmul(Lz(angle), (self._base_polygon-self._root_position).T).T + self._root_position

    def rotate_and_translate(self, orientation, translation):
        # Orientation in body angles (phi, theta, psi)
        L = L_eb(*orientation)
        return (np.matmul(L, self._actuated_polygon.T) + np.atleast_2d(translation).T).T


class Plane(object):
    _polygon_collection = None

    def __init__(self, plot_axis, geometry_file):
        self._axis = plot_axis
        self._geometry_file = geometry_file
        self._surface_defs = self._load_surfaces()
        # build polygon definitions
        self._surface_polygons = {}
        for name, defs in self._surface_defs.items():
            self._surface_polygons[name] = SurfacePolygon(**defs)

    def _load_surfaces(self):
        print("Loading YAML surface files: {0}".format(self._geometry_file))
        with open(self._geometry_file, 'rt') as fh:
            surfaces = yaml.safe_load(fh)
        return surfaces

    def set_wingsweeps(self, lambda_left, lambda_right):
        # Note that we add the minus because positive wing sweep is always towards the tail, but we're rotating
        # around the z-axis (z-down in body axes)
        self._surface_polygons['wing_left'].set_sweep(-lambda_left)
        self._surface_polygons['wing_right'].set_sweep(lambda_right)

    def set_rudder(self, d_r):
        # This just rotates the whole rudder about its base point
        self._surface_polygons['vstab'].set_sweep(d_r)

    def plot(self, X=None, U=None):
        # X = [u, v, w, p, q, r, phi, theta, psi, x, y, z]
        if X is None:
            X = np.zeros(12)
        if U is not None:
            self.set_wingsweeps(U[0], U[1])
            self.set_rudder(U[2])

        all_polygons = [poly.rotate_and_translate(X[6:9], X[9:12]) for poly in self._surface_polygons.values()]
        self._polygon_collection = Poly3DCollection(all_polygons)
        self._axis.add_collection3d(self._polygon_collection, zs='z')

    def update(self, X, U=None):
        if self._polygon_collection is None:
            self.plot(X, U)
            return
        if U is not None:
            self.set_wingsweeps(U[0], U[1])
            self.set_rudder(U[2])
        all_polygons = [poly.rotate_and_translate(X[6:9], X[9:12]) for poly in self._surface_polygons.values()]
        self._polygon_collection.set_verts(all_polygons)

if __name__ == "__main__":
    plt.rc('font', **{'family': 'serif', 'sans-serif': ['Computer Modern Roman']})
    plt.rc('text', usetex=True)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlim(-2, 10)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-9.5, 0.5)

    ax.invert_zaxis()
    ax.invert_yaxis()
    my_plane = Plane(ax, 'models/default_plane.yaml')
    my_plane.set_wingsweeps(np.pi/6, -np.pi/8)
    my_plane.set_rudder(np.pi/8)
    my_plane.plot(X=[0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0])

    ax.set_xlabel(r'$X_e$')
    ax.set_ylabel(r'$Y_e$')
    ax.set_zlabel(r'$Z_e$')

    plt.show(block=False)