import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation
import csv
import argparse


class WingTransformer(object):
    _scale = 1.0
    _position = (0.0, 0.0)
    _rotation = 0.0

    def __init__(self, data_file, le_offset=0.25, position=None, rotation=None, scale=None):
        wing_data = []
        with open(data_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=' ', skipinitialspace=True)
            for i, row in enumerate(csv_reader):
                if i == 0:
                    self.wing_name = " ".join(row)
                else:
                    wing_data.append([float(row[0]), float(row[1])])
        self.wing_data = np.array(wing_data) - np.array([le_offset, 0.0])
        self.n_points = self.wing_data.shape[0]
        print('Read wing data file {0}, wing name {1} with {2} data points.'.format(
            data_file, self.wing_name, self.n_points))

        if position is not None:
            self.set_position(position)
        if rotation is not None:
            self.set_rotation(rotation)
        if scale is not None:
            self.set_scale(scale)

    def set_scale(self, scale):
        self._scale = scale

    def set_position(self, position):
        assert len(position) == 2
        self._position = position

    def set_rotation(self, rotation, in_degrees=False):
        if in_degrees:
            rotation = rotation*np.pi/180.0
        self._rotation = rotation

    def get_points(self):
        # Use affine transformation to get transformed points
        # Return x and y vectors

        # Scale and rotation
        transform = np.identity(2)
        transform[[0, 1], [0, 1]] = np.cos(self._rotation)*self._scale
        transform[0, 1] = np.sin(self._rotation)
        transform[1, 0] = -transform[0, 1]
        new = (transform @ self.wing_data.T).T

        # Translation
        new += np.array(self._position)

        return new[:, 0], new[:, 1]


class AlphaSweepPlotter(object):
    # Class for visualising differential pressure and wing alpha
    artists = []
    legend_entries = []

    def __init__(self, alpha_data, pressure_data, wing_plotter, frameskip=1, dp_offset=0):
        assert len(alpha_data) == pressure_data.shape[0]
        assert isinstance(wing_plotter, WingTransformer)
        self.wing = wing_plotter
        self.alpha_data = alpha_data
        self.pressure_data = pressure_data
        self.n_max = pressure_data.shape[0]
        self.frameskip = frameskip
        self.max_frames = self.n_max//self.frameskip
        self.dp_offset = dp_offset

        self.fh, self.ax = plt.subplots()
        self.fh.set_size_inches([11, 6])

        self.wing_axes = self.fh.add_axes([.6, .6, .25, .25])
        self.wing_axes.set_facecolor('white')

        # Annoying thing to find max point
        self.a_max_index = np.where(self.alpha_data == self.alpha_data.max())[0][0]

        # self.colours = cm.get_cmap()(np.linspace(0, 1.0, (self.x0)+2))[1:-1]
        self.init()

    def init(self):
        self.ax.cla()
        self.ax.set_xlim(self.alpha_data.min(), self.alpha_data.max())
        self.ax.set_ylim(self.pressure_data.min(), self.pressure_data.max())

        self.wing_axes.cla()
        self.wing_axes.set_xlim(-1, 1)
        self.wing_axes.set_ylim(-1, 1)
        # self.wing_axes.set_title(r'Wing $\alpha$')
        self.wing_axes.set_aspect('equal')
        self.wing_axes.get_xaxis().set_ticks([])
        self.wing_axes.get_yaxis().set_ticks([])
        # self.wing_axes.axis('off')

        self.forward_lines = []
        self.backward_lines = []
        self.points = []
        self.legend_entries = []

        for i, dp in enumerate(self.pressure_data.T):
            l, = self.ax.plot(self.alpha_data[0], dp[0], lw=1.0)
            lb, = self.ax.plot(self.alpha_data[self.a_max_index], dp[self.a_max_index], '--', c=l.get_color(), lw=1.0)
            p, = self.ax.plot(self.alpha_data[0], dp[0], 'o', c=l.get_color())
            self.forward_lines.append(l)
            self.backward_lines.append(lb)
            self.points.append(p)
            self.legend_entries.append('$dp_{0}$'.format(i+self.dp_offset))
        self.ax.grid(True)
        self.ax.set_xlabel(r'Angle of attack $\alpha$ (deg)')
        self.ax.set_ylabel('Diff. pressure (Pa)')

        self.ax.legend(self.forward_lines, self.legend_entries, loc=4)

        # Plot wing
        self.wing.set_rotation(self.alpha_data[0], in_degrees=True)
        wing_x, wing_y = self.wing.get_points()
        self.wing_artist, = self.wing_axes.plot(wing_x, wing_y, c='black')

        return [*self.forward_lines, *self.backward_lines, *self.points, self.wing_artist]

    def animate(self, i):
        i = i*self.frameskip

        assert i < self.n_max

        # Update wing orientation
        self.wing.set_rotation(self.alpha_data[i], in_degrees=True)
        wing_x, wing_y = self.wing.get_points()
        self.wing_artist.set_data(wing_x, wing_y)

        for j, a in enumerate(self.points):
            a.set_data(self.alpha_data[i], self.pressure_data[i, j])

        self.points
        if i <= self.a_max_index:
            for j, a in enumerate(self.forward_lines):
                a.set_data(self.alpha_data[:i], self.pressure_data[:i, j])
        else:
            for j, a in enumerate(self.backward_lines):
                a.set_data(self.alpha_data[self.a_max_index:i], self.pressure_data[self.a_max_index:i, j])

        return [*self.forward_lines, *self.backward_lines, *self.points, self.wing_artist]


def load_pressure_data(pressure_file):
    # Return alpha data and pressure data
    pressure_data = []
    with open(pressure_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            row_data = [float(x) for x in row]
            pressure_data.append(row_data)

    pressure_data = np.array(pressure_data)

    # Remove rows with nans
    good_rows = np.all(~np.isnan(pressure_data), axis=1)
    print("Removed {0}/{1} rows containing nan values".format(pressure_data.shape[0]-good_rows.sum(), pressure_data.shape[0]))
    pressure_data = pressure_data[good_rows, :]

    # Every second column should be alpha data, and these should be the same
    n_data = pressure_data.shape[1]//2
    for i in range(n_data-1):
        if (not np.all(pressure_data[:, 0] == pressure_data[:, 2*(i+1)])):
            print("Warning: col. {0} didn't match col. 0, check input data".format(i))

    # Return alpha data and pressure data
    return pressure_data[:, 0], pressure_data[:, 1::2]


if __name__ == '__main__':
    plt.rc('text', usetex=True)

    parser = argparse.ArgumentParser(description='Plot some wing data')
    parser.add_argument('-p', '--pressure-file', required=True, help='CSV file containing pressure/alpha data')
    parser.add_argument('-w', '--wing-file', required=True, help='File containing wing coordinates (UIUC/XFOIL format)')
    parser.add_argument('-t', '--total-time', type=float, default=10.0, help='Duration of animation (s)')
    parser.add_argument('-f', '--frameskip', type=int, default=1, help='Frameskip (only plot every nth frame)')
    parser.add_argument('-s', '--save-video', default='', help="Save video to this file (default don't save)")
    args = parser.parse_args()

    wing = WingTransformer(args.wing_file)
    alpha_data, pressure_data = load_pressure_data(args.pressure_file)

    with plt.style.context('ggplot'):

        animator = AlphaSweepPlotter(alpha_data, pressure_data, wing, frameskip=args.frameskip, dp_offset=3)

        delta_t = (args.total_time * 1000.0 / animator.max_frames)
        animation = FuncAnimation(animator.fh, animator.animate, init_func=animator.init, frames=animator.max_frames,
                                  interval=delta_t, blit=True)
        if args.save_video:
            animation.save(args.save_video, writer='ffmpeg', fps=int(1000.0/delta_t), extra_args=["-crf","10", "-profile:v", "main"])
    plt.show()
