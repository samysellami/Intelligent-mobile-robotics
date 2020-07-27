import time
import numpy as np
import pandas  as pd
import matplotlib.pyplot as plt
import cv2

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


#########################################################################################################################################################
##################################################################### split and merge ####################################################################
##########################################################################################################################################################

start = time.time()

p1 = polarToCartesian(points[:, 0])
p2 = polarToCartesian(points[:, -1])
line = line_equation(p1, p2)
points_cartesian = np.zeros(points.shape)

for i in range(points.shape[1]):
    points_cartesian[:, i] = polarToCartesian(points[:, i])
##################################################### Splitting and sorting the resulting points ##########################################################
p_mosts = np.zeros(2)
split(points_cartesian, line, threshold_split, p1, p2)
p_mosts = np.delete(p_mosts, (0), axis=0)
p_mosts = np.vstack([p_mosts, p2])
p_mosts = np.vstack([p1, p_mosts])

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
plt.scatter( p_mosts[:, 0] , p_mosts[:, 1] ,c = "r", linewidth=2, label='lines_edges')
plt.scatter(points_cartesian[0, :], points_cartesian[1, :], c = "g", linewidth = 0.5, label = 'point cloud')
for i in range(p_mosts_cartesian.shape[0] - 1):
    x_range = (p_mosts_cartesian[i, 0], p_mosts_cartesian[i+1, 0])
    plt.plot([x_range[0], x_range[1]], [x_range[0]*lines[i][0] + lines[i][1], x_range[1]*lines[i][0] + lines[i][1]], c = "r", linewidth=2)
    plt.title('Split and merge')
    plt.legend()
plt.show()

end = time.time()
time_split = end-start

##########################################################################################################################################################
##################################################################### line regression ####################################################################
##########################################################################################################################################################


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

start = time.time()
################################################################## fit line in a window ####################################################################
Nf= 7
window = points_cartesian[:, :Nf]
line = line_least_square(window)

########################################################### Building the line and the fidelity array ######################################################
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
    plt.plot([x_range[0], x_range[1]], [x_range[0] * line[0] + line[1], x_range[1] * line[0] + line[1]], c="b", linewidth=2)
    i = i+ j
plt.legend()
plt.title('Line regression')
plt.show()

end = time.time()
time_reg = end-start

##########################################################################################################################################################
##################################################################### Hough transform ####################################################################
##########################################################################################################################################################


def cartesianToPixel(p, range_x, range_y, x_neg, y_neg):
    return [ int( (p[0] - x_neg) * 500 / range_x), int((p[1] - y_neg) * 500 / range_y ) ]

start = time.time()

x_neg = np.min(points_cartesian[0, :])
y_neg = np.min(points_cartesian[1, :])
range_x = np.max(points_cartesian[0, :]) - np.min(points_cartesian[0, :])
range_y = np.max(points_cartesian[1, :]) - np.min(points_cartesian[1, :])


image = np.zeros([800, 800])
for i in range(points_cartesian.shape[1]):
    image[ 500 +100 - cartesianToPixel(points_cartesian[:, i], range_x, range_y, x_neg, y_neg)[1], 150 + cartesianToPixel(points_cartesian[:, i], range_x, range_y, x_neg, y_neg)[0] ] = 255

cv2.imwrite("tmp.png", image)
image = cv2.imread("tmp.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
kernel = np.ones((3, 3), np.uint8)
gray = cv2.dilate(gray, kernel, iterations=1)
thresold_theta_min = 3 * np.pi /180

lines = cv2.HoughLines(gray,0.5,np.pi/180, 30)
thetas = []

for line in lines:
    for rho,theta in line:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        if not (sum(abs (np.abs(thetas) - np.abs(theta)) < thresold_theta_min) > 0 ) and len(thetas)> 0 or len(thetas)==0:
            cv2.line(image,(x1,y1),(x2,y2),(0,0,255),2)
        thetas.append(theta)

# cv2.imshow("figure", image)
# cv2.waitKey()
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)
plt.legend()
plt.title('Hough transform')
plt.show()

end = time.time()
time_Hough = end-start

################################################################# Checking the time of executions ######################################################
fig, ax = plt.subplots()
ind = np.arange(1, 4)
s, h, r = plt.bar(ind, [time_split, time_Hough, time_reg])
s.set_facecolor('g')
h.set_facecolor('b')
r.set_facecolor('r')
ax.set_xticks(ind)
ax.set_xticklabels(['Split and Merge', 'Hough Transform', 'Line Regression'])
ax.set_ylabel('Seconds')
ax.set_title('Line Extraction Algorithm')
plt.show()















