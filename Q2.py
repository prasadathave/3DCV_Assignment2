from cv2 import cv2
import os
import numpy as np
from ransac import ransac
###########################################


path = './image_data'

d = os.listdir(path)

for i in d:
    images = os.listdir(path+'/'+i)

    img1 = cv2.imread(path+'/'+i+'/'+images[0])

    for k in range(1,len(images)):
        img2 = cv2.imread(path+'/'+i+'/'+images[k])
        
        gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        
        kp1,des1 = sift.detectAndCompute(gray1,None)
        kp2,des2 = sift.detectAndCompute(gray2,None)

        bf = cv2.BFMatcher(cv2.NORM_L1,crossCheck=True)
        match_list = bf.match(des1,des2)
        # print(match_list)
        match_list_indices = []
        for kk in match_list:
            # print(kp1[kk.queryIdx].pt)
            (x1,y1) = kp1[kk.queryIdx].pt
            (x2,y2) = kp2[kk.trainIdx].pt
            match_list_indices.append([x1,y1,x2,y2])

        matches = np.matrix(match_list_indices)

        Homography_matrix,inliers = ransac(matches,5)
        print(Homography_matrix)
        
        # print(Homography_matrix)  
          
    
    

    # cv2.imwrite(path+'/'+i+'/'+'matching.JPG',img3)
######################################


























# img = cv2.imread('test.JPG')

# gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# ### getting sift points ###
# sift = cv2.SIFT_create()
# kp = sift.detect(gray,None)

# img=cv2.drawKeypoints(gray,kp,img)
# cv2.imwrite('sift_keypoints.jpg',img)