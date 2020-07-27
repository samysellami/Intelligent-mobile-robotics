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

points_cartesian = np.zeros(points.shape)
for i in range(points.shape[1]):
    points_cartesian[:, i] = polarToCartesian(points[:, i])


################################################################### Hough transform ######################################################################"

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

cv2.imshow("figure", image)
cv2.waitKey()













