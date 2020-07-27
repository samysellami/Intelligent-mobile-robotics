import numpy as np
import pandas  as pd
import matplotlib.pyplot as plt
import cv2


def line_equation(p1, p2):
    slope= (p2[1] - p1[1])/(p2[0] - p1[0])
    intercept= p1[1] - slope* p1[0]
    return [slope, intercept]

def polarToCartesian(p):
    x = p[1] * np.cos(p[0])
    y = p[1] * np.sin(p[0])
    return np.array([x, y])

def cartesianToPolar(p):
    rho = np.sqrt(p[0]**2 + p[1]**2)
    theta = np.arctan2(p[1], p[0])
    return np.array([theta, rho])


def line_least_square(p):
    y= p[1, :]
    x= p[0, :]
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    slope = np.dot((x - mean_x),  (y - mean_y))/np.dot ((x - mean_x), (x - mean_x))
    intercept = mean_y - slope * mean_x
    return [slope , intercept]

def mahalanobis_dist (window1 , window2):
    y1 = window1[1, :] - np.mean(window1[1, :])
    x1 = window1[0, :] - np.mean(window1[0, :])
    y2 = window2[1, :] - np.mean(window2[1, :])
    x2 = window2[0, :] - np.mean(window2[0, :])

    u1 = np.hstack([x1, y1])
    u2 = np.hstack([x2, y2])
    cov =  np.outer(u1, np.transpose(u2))
    dist = np.sqrt( np.dot(np.transpose(u1) , np.dot( np.linalg.inv(cov), u2 ) ))
    return dist

def cartesianToPixel(p, range_x, range_y, x_neg, y_neg):
    return [ int( (p[0] - x_neg) * 500 / range_x), int((p[1] - y_neg) * 500 / range_y ) ]

############################################################### Getting data #############################################################################
threshold_split = 0.05
threshold_merge = 0.1
threshold_noise = 0.5

data= pd.read_csv('data.csv', sep=';')
rho = np.transpose(np.array(data.iloc[9: , 0].values))
increment = data.iloc[4, 0]
increments = np.zeros([rho.shape[0]])
increments[0]= data.iloc[2, 0]

for i in range(1, increments.shape[0]):
    increments[i]= increments[i-1] + increment
points = np.vstack([increments, rho])
points = points[:, ~np.isnan(points).any(axis=0)]

################################################################## Removing noise ###########################################################################
inds = []
i=0
while i < (points.shape[1] - 1):
    j = 1
    while (np.abs(points[1, j+i] - points[1, i+j-1]) < threshold_noise) and j+i < points.shape[1]-1:
        j += 1

    if j < 50:
        for k in range(j):
            inds.append(i+k)
    i = i + j
points = np.delete(points, (inds), axis=1)


################################################################## fit line in a window ####################################################################
points_cartesian = np.zeros(points.shape)
for i in range(points.shape[1]):
    points_cartesian[:, i] = polarToCartesian(points[:, i])

Nf= 7
window = points_cartesian[:, :Nf]
line = line_least_square(window)

# x_range = (window[0, 0], window[0, -1])
# plt.scatter(points_cartesian[0, :Nf], points_cartesian[1, :Nf], c = "g", linewidth = 0.05, label = 'point cloud')
# plt.plot([x_range[0], x_range[1]], [x_range[0]*line[0] + line[1], x_range[1]*line[0] + line[1]], c="r", linewidth=2, label='first line')
# plt.show()

################################################################## Building the line and the fidelity array #################################################
lines = [line]
fidelity_array = []
for i in range(1, points_cartesian.shape[1] - Nf-1):
    window1 = points_cartesian[:, i-1 : i + Nf]
    window2 = points_cartesian[:, i-1 + 1 : i + Nf + 1]

    lines.append(np.array(line_least_square(window1)))
    fidelity_array.append( mahalanobis_dist(window1, window2) )

################################################################## Clustering and merging lines #############################################################
i = 0
j = 1
plt.scatter(points_cartesian[0, :], points_cartesian[1, :], c = "g", linewidth = 0.05, label = 'point cloud')

while i + j < len(fidelity_array):
    j = 1
    if not (np.isnan(fidelity_array[i]) ):
        while not(np.isnan(fidelity_array[i + j]) ):
            if (i + j < len(fidelity_array) - 1):
                j += 1
            else:
                break

    window = points_cartesian[:, i: i + j * Nf]
    line = line_least_square(window)
    x_range = (window[0, 0], window[0, -1])
    plt.plot([x_range[0], x_range[1]], [x_range[0] * line[0] + line[1], x_range[1] * line[0] + line[1]], c="r",
                 linewidth=2, label='first line')
    i = i+ j

plt.show()









