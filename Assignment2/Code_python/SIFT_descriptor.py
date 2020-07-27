import numpy as np
import pandas  as pd
import matplotlib.pyplot as plt
import cv2
plt.rcParams["figure.figsize"] = (12, 7) # (w, h)


# image = cv2.imread("0000000130.png")
# cv2.imshow("figure", image)
# cv2.waitKey()
images = ['0000000130.png' , '0000000131.png', '0000000132.png', '0000000133.png', '0000000134.png', '0000000135.png']
for img in images:
    image = cv2.imread(img, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    cv2.rectangle(image,(750,100),(1150,350),(0,255,0),3)
    plt.imshow(image)
    # plt.show()

image = images[0]
gray = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
# gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
mask = np.zeros(gray.shape, np.uint8)
mask[750:1150 , 100:400] = 255
orb = cv2.ORB_create()
kp = orb.detect(gray,mask)
image_keypoins =cv2.drawKeypoints(gray,kp,image,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

plt.imshow(img_keypoins)
plt.xticks([]),plt.yticks([])
plt.show()