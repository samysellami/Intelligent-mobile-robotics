
import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, sqrt
# from filterpy.stats import plot_covariance_ellipse
import filterpy

delta_t  = 0.1
b= 0.2
landmarks = np.array([[5, 10], [10, 5], [15, 15]])
cmds = [np.array([1.3, 1.3])] * 200
sigma_r=0.3
sigma_phi=0.1

def predict_x(u, x, b):
    sr = u[1] * delta_t
    sl = u[0] * delta_t
    x[0] = x[0] + 0.5*(sr + sl) * cos( x[2] + 0.5 * (sr-sl)/b)
    x[1] = x[1] + 0.5 * (sr + sl) * sin(x[2] + 0.5 * (sr - sl) / b)
    x[2] = normalize_angle(x[2] +  (sr - sl)  / b)
    return x

def normalize_angle(theta):
    theta = theta % (2 *np.pi)
    if theta > np.pi:
        theta -= 2*np.pi
    return theta

def residual_h(z, ze):
    y = z -ze
    for i in range (0, len(y), 2):
        y[i+1]= normalize_angle(y[i+1])
    return y

def h(x, landmarks):
    hx = []
    for landmark in landmarks:
        px, py = landmark
        dist = sqrt((px - x[0])**2 + (py - x[1])**2)
        angle = np.arctan2(py - x[1], px - x[0])
        hx.append(dist)
        hx.append( normalize_angle(angle - x[2]) )
    return np.array(hx)

def predict_P( P, Q, Phi):

    P = np.dot(Phi, np.dot( P, Phi.transpose()) ) + Q
    return P

def update_P(P, K, H):
    return P - np.dot(K, np.dot(H, P))

def calcul_K(P, H, R):
    cov_v = np.dot( H, np.dot(P, H.transpose()) ) + R
    return np.dot( np.dot(P, H.transpose()),  np.linalg.inv(cov_v) )



def EKF_robot(landmarks, b, sigma_r, sigma_phi, cmds, ellipse_step = 10):
    x = np.array([2, 6, .3])
    P = np.diag([.1, .1, .05])
    R = np.diag([sigma_r ** 2, sigma_phi ** 2] * len(landmarks))
    Q = np.eye(3) * 0.0001

    plt.figure()
    if len(landmarks) > 0:
        plt.scatter(landmarks[:, 0], landmarks[:, 1], marker='s', s=60)

    path =[]
    for i, u in enumerate(cmds):
        x = predict_x(u, x, b)

        path.append(np.array(x))
        sr = u[1] * delta_t
        sl = u[0] * delta_t

        Phi = np.array([[1, 0, - 0.5 * (sr + sl) * sin(x[2] + 0.5 * (sr - sl))],
                        [0, 1, 0.5 * (sr + sl) * cos(x[2] + 0.5 * (sr - sl))], [0, 0, 1]])
        j=0
        H = np.zeros( [2* landmarks.shape[0], 3] )
        for landmark in (landmarks):
            px, py = landmark
            H[j, :] = [(-px + x[0]) / (sqrt((px - x[0]) ** 2 + (py - x[1]) ** 2)),
                   (-py + x[1]) / (sqrt((px - x[0]) ** 2 + (py - x[1]) ** 2)), 0]

            H[j+1, :] = [(py - x[1]) / (sqrt((px - x[0]) ** 2 + (py - x[1]) ** 2)),
                   (-px + x[0]) / (sqrt((px - x[0]) ** 2 + (py - x[1]) ** 2)), -1]
            j +=2

        ###  generate data from sensor
        z = []
        for landmark in landmarks:
            dx, dy = landmark[0] - x[0], landmark[1] - x[1]
            d = sqrt(dx**2 + dy**2) + np.random.randn()*sigma_r
            bearing = np.arctan2(dy, dx)
            a = normalize_angle(bearing - x[2] + np.random.randn()*sigma_phi)
            z.append(d)
            z.append(a)

        z = np.array(z)
        P_ = predict_P( P, Q, Phi)
        ze  = h(x, landmarks)
        v= residual_h(z, ze)
        K = calcul_K(P, H, R)

        x = x + np.dot(K, v)
        P = update_P(P, K, H)

        if i % ellipse_step == 0:
            filterpy.stats.plot_covariance_ellipse(
                (x[0], x[1]), P[0:2, 0:2], std=6,
                facecolor='g', alpha=0.8)

    return np.array(path)

track = EKF_robot(landmarks, b, sigma_r, sigma_phi, cmds)
# print('track= ' +str(track))
plt.plot(track[:, 0], track[:,1], color='k', lw=2)
plt.axis('equal')
plt.title("UKF Robot localization")
plt.show()












