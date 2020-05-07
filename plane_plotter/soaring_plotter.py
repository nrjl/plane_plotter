import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from tools.plane_animation import PlaneAnimation
import argparse

# Set some text properties for nice labels
plt.rc('font', **{'family': 'serif', 'sans-serif': ['Computer Modern Roman']})
plt.rc('text', usetex=True)

parser = argparse.ArgumentParser(description='Basic animation of an aircraft')
parser.add_argument('-p', '--plane', default='models/default_plane.yaml', help='Plane definition file')
parser.add_argument('-vf', '--video-file', type=str, default='', help='Save animation to file')
parser.add_argument('-dt', type=float, default=0.3, help='Time step (s)')
parser.add_argument('--max-steps', type=int, default=-1, help='Max time steps (default -1 is all)')
parser.add_argument('data_file', type=str, help='State history npz file')
args = parser.parse_args()

# Load data file
state_data = np.load(args.data_file)

observations = state_data['obs'][:args.max_steps, :]
actions = state_data['actions'][:args.max_steps, :]*30*np.pi/180
print('Loaded data file {0}, using {1} steps ({2:0.2f} sec)'.format(
    args.data_file, actions.shape[0], actions.shape[0]*args.dt))

# Remember: X = [u, v, w, p, q, r, phi, theta, psi, x, y, z]
full_X = np.zeros((actions.shape[0], 12))
full_X[:, -1] = observations[:, 0]  # Altitude
full_X[:, 6] = actions[:, 1]        # Roll
full_X[:, 7] = actions[:, 0]        # Pitch
full_X[:, 8] = observations[:, 2]   # Heading

full_U = np.zeros((actions.shape[0], 3))

fig = plt.figure(figsize=[8, 8])
ax = Axes3D(fig)

ax.invert_zaxis()
ax.invert_yaxis()

ax.set_xlabel(r'$X_e$')
ax.set_ylabel(r'$Y_e$')
ax.set_zlabel(r'$Z_e$')

glider = PlaneAnimation(ax, full_X, full_U, keep_centred=True)

# Creating the Animation object
glider_animation = animation.FuncAnimation(fig, func=glider.animate, frames=actions.shape[0], blit=False,
                                           init_func=glider.first_frame, interval=1000.0*args.dt, repeat=False)
if args.video_file is not '':
    print('Saving video to {0} ...'.format(args.video_file), end='')
    glider_animation.save(args.video_file, writer='ffmpeg', fps=1.0/args.dt,
                          extra_args=["-crf", "23", "-profile:v", "main", "-tune", "animation", "-pix_fmt", "yuv420p"])
    print('done.')
plt.show()
