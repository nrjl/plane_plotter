import numpy as np
from pyulog.core import ULog
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from plane_animation import PlaneAnimation

def quaternion_rotation_matrix(q):
    qr, qi, qj, qk = q[0], q[1], q[2], q[3]
    R = np.array([[1-2*(qj**2+qk**2), 2*(qi*qj-qk*qr), 2*(qi*qk + qj*qr)],
                  [2*(qi*qj + qk*qr), 1-2*(qi**2+qk**2),  2*(qk*qj-qi*qr)],
                  [2*(qi*qk - qj*qr), 2*(qj*qk + qi*qr), 1-2*(qi**2+qj**2)]])
    return R

def slerp(v0, v1, t_array):
    # This is a quaternion interpolation method
    # >>> slerp([1,0,0,0],[0,0,0,1],np.arange(0,1,0.001))
    # From Wikipedia: https://en.wikipedia.org/wiki/Slerp
    t_array = np.array(t_array)
    v0 = np.array(v0)
    v1 = np.array(v1)
    dot = np.sum(v0 * v1)
    if (dot < 0.0):
        v1 = -v1
        dot = -dot
    DOT_THRESHOLD = 0.9995
    if (dot > DOT_THRESHOLD):
        result = v0[np.newaxis, :] + t_array[:, np.newaxis] * (v1 - v0)[np.newaxis, :]
        result = result / np.linalg.norm(result)
        return result
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * t_array
    sin_theta = np.sin(theta)
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return (s0[:, np.newaxis] * v0[np.newaxis, :]) + (s1[:, np.newaxis] * v1[np.newaxis, :])


def get_quat(data, index):
    return(np.array([data['q[0]'][index], data['q[1]'][index], data['q[2]'][index], data['q[3]'][index]]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Make simple video from ulog file')
    parser.add_argument('input_ulog', help='Input ulog file')
    parser.add_argument('--start', dest='start_time', default=-1.0, help='Start time (s)')
    parser.add_argument('--stop', dest='stop_time', default=-1.0, help='Stop time (s)')
    parser.add_argument('--fps', default=25, help='Frames per second (will be resampled to this time)')
    args = parser.parse_args()

    # Set some text properties for nice labels
    plt.rc('font', **{'family': 'serif', 'sans-serif': ['Computer Modern Roman']})
    plt.rc('text', usetex=True)
    fig = plt.figure(figsize=[8,8])
    ax = Axes3D(fig)

    ax.set_xlabel(r'$X_e$')
    ax.set_ylabel(r'$Y_e$')
    ax.set_zlabel(r'$Z_e$')

    log_data = ULog(args.input_log).data_list
    all_names = [log_data[i].name for i in range(len(log_data))]
    ulog_attitude = log_data[all_names.index('vehicle_attitude')].data
    ulog_controls = log_data[3].data

    full_X = np.zeros((50, 12))
    full_X[:15,6] = np.linspace(0, np.pi/4, 15)
    full_X[15:35,6] = np.linspace(np.pi/4, -np.pi/4, 20)
    full_X[35:,6] = np.linspace(-np.pi/4, 0, 15)
    full_X[:,9] = np.linspace(0, 10, full_X.shape[0])
    full_X[:,10] = 0.5*np.sin(np.linspace(0, 2*np.pi, full_X.shape[0]))

    full_U = np.zeros((50, 3))
    full_U[:,0] = np.pi/6*np.sin(np.linspace(0, 2*np.pi, full_U.shape[0]))
    full_U[:,1] = np.pi/6*np.sin(np.linspace(0, 2*np.pi, full_U.shape[0]))
    full_U[:,2] = np.pi/8*np.sin(np.linspace(0, 2*np.pi, full_U.shape[0]))

    my_plane = PlaneAnimation(ax, full_X, full_U)

    # Creating the Animation object
    plane_animation = animation.FuncAnimation(fig, func=my_plane.animate, frames=50, init_func=my_plane.first_frame,
                                              interval=50, blit=False, repeat=False)
    writer = animation.FFMpegWriter(fps=30, codec="h264", extra_args=["-crf","23"])
    plane_animation.save('vid/test.mp4', writer=writer)

    plt.show()

