import numpy as np
import cv2
import matplotlib.pylab as plt
import math
import glob

# checkerboard Dimensions
cbrow = 6
cbcol = 9

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((cbrow * cbcol, 3), np.float32)
objp[:, :2] = np.mgrid[0:cbcol, 0:cbrow].T.reshape(-1, 2)

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

cap = cv2.VideoCapture(0)

while True:
    ret, image = cap.read()
    imageScaled = cv2.resize(image, (720, 480))
    imageScaled = cv2.flip(imageScaled,1)
    img = imageScaled
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (cbcol,cbrow),None)

    # If found, add object points, image points (after refining them)
    if ret:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        # img = cv2.drawChessboardCorners(img, (cbrow,cbcol), corners2,ret)

        xvals = []
        angles = [0] #0 for center vertex, need a odd number for cbcol
        for i in range(cbcol):
            xvals.append(corners2[i][0][0])
        distance = 10 #in inches from camera to calibration board
        cubeSize = 0.72 #in inches
        for i in range(math.floor(cbcol/2)):
            angles.append(math.atan((i+1)*cubeSize/distance)*round(180/math.pi))
            angles.insert(0,(math.atan((i + 1) * cubeSize / distance)*-1)*round(180/math.pi))

        x = np.array(xvals)
        y = np.array(angles)

        m = (len(x) * np.sum(x * y) - np.sum(x) * np.sum(y)) / (len(x) * np.sum(x * x) - np.sum(x) * np.sum(x))
        b = (np.sum(y) - m * np.sum(x)) / len(x)
        print(m,b)

        plt.scatter(x, y)
        plt.xlabel("X value of center pixel")
        plt.ylabel("Assumed angle in deg")
        plt.show()

        cv2.imshow('img',img)
        cv2.waitKey(30)
    else:
        cv2.imshow('img',img)
        cv2.waitKey(30)

cv2.destroyAllWindows()