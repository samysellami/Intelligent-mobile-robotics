import numpy as np
import pandas  as pd
import matplotlib.pyplot as plt

def line_equation(p1, p2):
    slope= (p2[1] - p1[1])/(p2[0] - p1[0])
    intercept= p1[1] - slope* p1[0]
    return [slope, intercept]

def calcul_dist(line, p):
    return np.abs( ( -line[0]* p[0] + p[1] -  line[1] ) / ( np.sqrt( line[0]**2 + 1 ) ) )

def polarToCartesian(p):
    x = p[1] * np.cos(p[0])
    y = p[1] * np.sin(p[0])
    return np.array([x, y])

def cartesianToPolar(p):
    rho = np.sqrt(p[0]**2 + p[1]**2)
    theta = np.arctan2(p[1], p[0])
    return np.array([theta, rho])

def most_distant (points, line):
    d = 0
    k=0
    for i in range(1, points.shape[1]):
        dist = calcul_dist(line, points[:, i])
        if dist > d:
            d = dist
            k = i
    return d, k

def split(points_cartesian, line, threshold_split, p1, p2):
    d, k = most_distant(points_cartesian, line)
    global p_mosts

    if d > threshold_split:
        p_most = points_cartesian[:, k]
        p_mosts = np.vstack([p_mosts, p_most])

        points_split_left = points_cartesian[:, :k]
        points_split_right = points_cartesian[:, k:]

        line_left = line_equation(p1, p_most)
        line_right = line_equation(p2, p_most)

        split(points_split_left, line_left, threshold_split, p1, p_most)
        split(points_split_right, line_right, threshold_split, p_most, p2)
    return 0

# data= pd.read_csv('point_cloud.csv', sep=';')
# rho = np.array(data.iloc[0, 12:].values)
# increment = data['field.angle_increment']
# increments = np.zeros([rho.shape[0]])
# increments[0]= data.iloc[0, 4]
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

############################################################## Removing noise ###########################################################################
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

############################################################ Plotting the first line #####################################################################
p1 = polarToCartesian(points[:, 0])
p2 = polarToCartesian(points[:, -1])
line = line_equation(p1, p2)
points_cartesian = np.zeros(points.shape)

for i in range(points.shape[1]):
    points_cartesian[:, i] = polarToCartesian(points[:, i])

x_range = (p1[0], p2[0])
plt.scatter(points_cartesian[0, :], points_cartesian[1, :], c = "g", linewidth = 0.05, label = 'point cloud')
plt.plot([x_range[0], x_range[1]], [x_range[0]*line[0] + line[1], x_range[1]*line[0] + line[1]], c="r", linewidth=2, label='first line')
# plt.show()

##################################################### Splitting and sorting the resulting points ##########################################################
p_mosts = np.zeros(2)
split(points_cartesian, line, threshold_split, p1, p2)
p_mosts = np.delete(p_mosts, (0), axis=0)
p_mosts = np.vstack([p_mosts, p2])
p_mosts = np.vstack([p1, p_mosts])

plt.scatter( p_mosts[:, 0] , p_mosts[:, 1] ,c = "r", linewidth=2, label='lines_edges')
p_mosts_polar = np.zeros(p_mosts.shape)
for i in range(p_mosts.shape[0]):
    p_mosts_polar[i, :] = cartesianToPolar(p_mosts[i, :])
p_mosts_polar = p_mosts_polar[np.argsort(p_mosts_polar[:, 0]), :]

p_mosts_cartesian = np.zeros(p_mosts.shape)
for i in range(p_mosts.shape[0]):
    p_mosts_cartesian[i, :] = polarToCartesian(p_mosts_polar[i, :])

############################################################# creating the lines and Merging #################################################################
inds = []
lines = []
for i in range(p_mosts_cartesian.shape[0]-1):
    line = line_equation(p_mosts_cartesian[i, :], p_mosts_cartesian[i + 1, :])

    if i > 1:
        if np.abs(np.abs(line[0]) - np.abs(lines[-1][0])) < threshold_merge :
            line = line_equation(p_mosts_cartesian[i-1, :], p_mosts_cartesian[i + 1, :])
            lines[-1] = line
            inds.append(i)
        else:
            lines.append(line)
    else:
        lines.append(line)
p_mosts_cartesian = np.delete(p_mosts_cartesian, (inds), axis=0)

############################################################## Plotting the final lines ######################################################################
for i in range(p_mosts_cartesian.shape[0] - 1):
    x_range = (p_mosts_cartesian[i, 0], p_mosts_cartesian[i+1, 0])
    plt.plot([x_range[0], x_range[1]], [x_range[0]*lines[i][0] + lines[i][1], x_range[1]*lines[i][0] + lines[i][1]], c = "b", linewidth=2, label='line' +str(i))
    plt.legend()
plt.show()










