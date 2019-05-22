import numpy as np

def Lz(psi):
    return np.array([[np.cos(psi), -np.sin(psi), 0],
                     [np.sin(psi),  np.cos(psi), 0],
                     [0, 0, 1]])


def Ly(theta):
    return np.array([[ np.cos(theta), 0, np.sin(theta)],
                     [0, 1, 0],
                     [-np.sin(theta), 0, np.cos(theta)]])


def Lx(phi):
    return np.array([[1, 0, 0],
                     [0, np.cos(phi), -np.sin(phi)],
                     [0, np.sin(phi),  np.cos(phi)]])


def L_eb(phi, theta, psi):
    return [[np.cos(theta)*np.cos(psi), np.sin(phi)*np.sin(theta)*np.cos(psi) - np.cos(phi)*np.sin(psi),
         np.cos(phi)*np.sin(theta)*np.cos(psi) + np.sin(phi)*np.sin(psi)],
            [np.cos(theta)*np.sin(psi), np.sin(phi)*np.sin(theta)*np.sin(psi) + np.cos(phi)*np.cos(psi),
             np.cos(phi)*np.sin(theta)*np.sin(psi) - np.sin(phi)*np.cos(psi)],
            [-np.sin(theta), np.sin(phi)*np.cos(theta), np.cos(phi)*np.cos(theta)]]
    # return np.matmul(np.matmul(Lz(-psi), Ly(-theta)), Lx(-phi))