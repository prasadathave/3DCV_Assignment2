from cv2 import cv2
import os

###########################################

########### detecting and plotting the sift features #################
path = './image_data'
a = os.listdir(path)
p1 = './Q1'
os.mkdir(p1)

for i in a:
    b = os.listdir(path+'/'+i)
    os.mkdir(p1+'/'+i)

    for k in b:
        path1 = path +'/'+i+'/'+k
        img = cv2.imread(path1)
        img1 = cv2.imread(path1)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        kp = sift.detect(gray,None)
        img1 = cv2.drawKeypoints(gray,kp,img1)
        cv2.imwrite(p1+'/'+i+'/'+k,img1)

################################################

############## matching the features in two images #############
d = os.listdir(path)

for i in d:
    images = os.listdir(path+'/'+i)

    img1 = cv2.imread(path+'/'+i+'/'+images[0])
    img2 = cv2.imread(path+'/'+i+'/'+images[1])
    
    gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    
    kp1,des1 = sift.detectAndCompute(gray1,None)
    kp2,des2 = sift.detectAndCompute(gray2,None)

    bf = cv2.BFMatcher(cv2.NORM_L1,crossCheck=True)
    match_list = bf.match(des1,des2)
    
    match_list = sorted(match_list,key = lambda x:x.distance)
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,match_list[:10],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite(path+'/'+i+'/'+'matching.JPG',img3)
######################################


























# img = cv2.imread('test.JPG')

# gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# ### getting sift points ###
# sift = cv2.SIFT_create()
# kp = sift.detect(gray,None)

# img=cv2.drawKeypoints(gray,kp,img)
# cv2.imwrite('sift_keypoints.jpg',img)