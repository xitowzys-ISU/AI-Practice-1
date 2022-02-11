import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("./data/internal_external.png", cv2.IMREAD_GRAYSCALE)
out = np.zeros_like(image)

# [Next, Previous, First_child, Parent]
contours, hierarchy = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

for i, contour in enumerate(contours):
    if hierarchy[0][i][3] == 4:
        cv2.drawContours(out, contours, i, 255, -1)

# print(hierarchy[0])

plt.subplot(121)
plt.imshow(image)
plt.subplot(122)
plt.imshow(out)
plt.show()