import random
import numpy as np
def find_homography_matrix(random_points):
    #looping through the four random points and finding the Assemble matrix
    matra = []
    for point in random_points:
        point1 = np.matrix([point.item(0), point.item(1), 1])
        point2 = np.matrix([point.item(2), point.item(3), 1])

        matra2 = [0, 0, 0, -point2.item(2) * point1.item(0), -point2.item(2) * point1.item(1), -point2.item(2) * point1.item(2),
              point2.item(1) * point1.item(0), point2.item(1) * point1.item(1), point2.item(1) * point1.item(2)]

        matra1 = [-point2.item(2) * point1.item(0), -point2.item(2) * point1.item(1), -point2.item(2) * point1.item(2), 0, 0, 0,
              point2.item(0) * point1.item(0), point2.item(0) * point1.item(1), point2.item(0) * point1.item(2)]
        
        matra.append(matra1)
        matra.append(matra2)

    Assemble_matrix = np.matrix(matra)

    #dividing assemble matri into the svd
    u, s, v = np.linalg.svd(Assemble_matrix)

    #reshape the min singular value into a 3 by 3 matrix
    h = np.reshape(v[8], (3, 3))
    
    #normalize and now we have h
    h = (1/h.item(8)) * h
    return h


#### error between estimated and real #####
def Distance_for_point(point, Homography_temp):

    point1 = np.transpose(np.matrix([point[0].item(0), point[0].item(1), 1]))
    estimate_point2 = np.dot(Homography_temp, point1)
    estimate_point2 = (1/estimate_point2.item(2))*estimate_point2

    point2 = np.transpose(np.matrix([point[0].item(2), point[0].item(3), 1]))
    error = point2 - estimate_point2
    return np.linalg.norm(error)


def ransac(matches,threshold):
    maximum_inliers = []
    Homography_matrix = []
    for i in range(1000):
        #taking 4 random points for calculating homography matrix
        a = matches[random.randrange(0, len(matches))]
        b = matches[random.randrange(0, len(matches))]
        
        c = matches[random.randrange(0, len(matches))]
        
        d = matches[random.randrange(0, len(matches))]
        
        four_points = np.vstack((a, b))
        four_points = np.vstack((four_points, c))
        four_points = np.vstack((four_points, d))

        #call the homography function on those points
        Homography_temp = find_homography_matrix(four_points)
        inliers = []

        for i in range(len(matches)):
            d = Distance_for_point(matches[i], Homography_temp)
            if d < threshold:
                inliers.append(matches[i])
        

        if len(inliers) > len(maximum_inliers):
            maximum_inliers = inliers
            Homography_matrix = Homography_temp
    

    return Homography_matrix, maximum_inliers
