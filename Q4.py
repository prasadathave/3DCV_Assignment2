from cv2 import cv2
path = './image_data/I1/STA_0031.JPG'

img = cv2.imread(path)

print(img[0][1])
print(img[1][0])

img[0][1] = img[1][0]
