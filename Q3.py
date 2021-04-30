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
        height,width,channels = img1.shape
        height1,width1,channels1 = img2.shape
        for ht in range(height):
            for wdt in range(width):
                new_cords = list((np.dot(Homography_matrix,np.matrix(np.array([ht,wdt,1])))).reshape(3,1))
                ht1,wt1 = int(new_cords[0]),int(new_cords[1])
                
                if(ht1>=height1):
                    ht1 = height1-1
                if(wt1>=width1):
                    wt1 = width1-1
                    
                try:
                    img1[ht][wdt] = img2[ht1][wt1]
                except Exception as e:
                    print(e)
                    continue

        print("one image iterated")
    cv2.imwrite(path+'/'+i+'/'+'final_panorama.JPG',img1)

        

            



    

    # cv2.imwrite(path+'/'+i+'/'+'matching.JPG',img3)
######################################

