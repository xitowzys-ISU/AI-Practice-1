import cv2
import numpy as np
import matplotlib.pyplot as plt

template = cv2.imread("./data/reeses_puffs.png", 0)
many = cv2.imread("./data/many_cereals.jpg", 0)

orb = cv2.ORB_create()

key_points1, descriptors1 = orb.detectAndCompute(template, None)
key_points2, descriptors2 = orb.detectAndCompute(many, None)

matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = matcher.match(descriptors1, descriptors2)

print(matches[0].distance)

matches = sorted(matches, key=lambda match: match.distance)
image = cv2.drawMatches(template, key_points1, many, key_points2, matches, None)

plt.imshow(image)
plt.show()
# print(key_points1)
# print(descriptors1)