import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse
import pandas


class SensorDataPlotter(object):
    # Class for visualising sensor data for video overlays
    artists = []
    legend_entries = []

    def __init__(self, data, variables, frequency, max_time=100.0, ax_colour='white'):
        self.data = data
        self.variables = variables
        self.n_max = self.data.shape[0]
        self.max_time = max_time
        self.frequency = frequency
        self.ax_colour = ax_colour

        self.fh, self.ax = plt.subplots(len(self.variables), 1)
        self.fh.set_size_inches([5, 5])
        self.fh.set_facecolor(self.ax_colour)

        self.time_vector = np.arange(0.0, self.n_max/self.frequency, 1.0/self.frequency)

        self.init()

    def init(self):

        for ax, var in zip(self.ax, self.variables):
            ax.cla()
            ax.set_facecolor(self.ax_colour)
            ax.set_xlim(0, min(self.time_vector[-1], self.max_time))
            ax.set_ylim(self.data[var].min(), self.data[var].max())

        self.lines = []
        self.points = []

        for ax, var in zip(self.ax, self.variables):
            l, = ax.plot(self.time_vector[:1], self.data[var][:1], lw=2.0, c='cornflowerblue')
            p, = ax.plot(self.time_vector[:1], self.data[var][:1], 'o', c='firebrick')
            self.lines.append(l)
            self.points.append(p)
            ax.grid(True)
            ax.set_xlabel(r'Time (s)')
            ax.set_ylabel(var)

        return [*self.lines, *self.points]

    def animate(self, i):
        assert i < self.n_max

        for l, p, var in zip(self.lines, self.points, self.variables):
            l.set_data(self.time_vector[:i + 1], self.data[var][:i + 1])
            p.set_data(self.time_vector[i], self.data[var][i])

        # Rolling axis limits (10% buffer on right end)
        if self.time_vector[i] > self.max_time*0.9:
            for j, ax in enumerate(self.ax):
                ax.set_xlim(self.time_vector[i]-0.9*self.max_time, self.time_vector[i]+0.1*self.max_time)

        return [*self.lines, *self.points]


def read_sensor_data(file):
    dd = pandas.read_csv(file, skip_blank_lines=True, comment='@')
    print('Loaded data file {0}, found {1} fields, {2} total records.'.format(file, dd.shape[1], dd.shape[0]))
    return dd


if __name__ == '__main__':
    # plt.rc('text', usetex=True)

    parser = argparse.ArgumentParser(description='Plot some sensor data')
    parser.add_argument('-t', '--max-time', type=float, default=100.0, help='Maximum time range (s)')
    parser.add_argument('-f', '--frequency', type=float, default=20.0, help='Data frequency (Hz)')
    parser.add_argument('-vf', '--video-file', default='', help="Output video file (default/empty str don't save)")
    parser.add_argument('-ac', '--ax-colour', default='white', help="Axis colour")
    parser.add_argument('data_file', type=str, help='CSV file containing sensor data')
    parser.add_argument('variables', nargs='*', type=str, default=['all'],
                        help="Data fields to plot, leave blank or set to 'all' for all")
    args = parser.parse_args()

    sensor_data = read_sensor_data(args.data_file)

    if args.variables[0].lower() == 'all':
        accepted_columns = sensor_data.columns

    else:
        accepted_columns = []
        for c in args.variables:
            if c in sensor_data.columns:
                accepted_columns.append(c)
            else:
                print('Requested variable {0} not found, ignoring.'.format(c))

    print('Found {0} variables.'.format(len(accepted_columns)))

    # with plt.style.context('ggplot'):

    animator = SensorDataPlotter(sensor_data, accepted_columns, ax_colour=args.ax_colour, max_time=args.max_time, frequency=args.frequency)

    delta_t = (1000.0 / args.frequency)
    animation = FuncAnimation(animator.fh, animator.animate, init_func=animator.init, frames=sensor_data.shape[0],
                              interval=delta_t, blit=False)
    if args.video_file:
        animation.save(args.video_file, writer='ffmpeg', codec='libx264', fps=int(1000.0/delta_t), dpi=200,
                       extra_args=["-crf", "23", "-profile:v", "main", "-tune", "animation", "-pix_fmt", "yuv420p"],
                       savefig_kwargs={'facecolor': args.ax_colour}
                       )
    plt.show()
