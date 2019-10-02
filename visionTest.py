# import the necessary packages
import numpy as np
import argparse
import cv2
import matplotlib.pylab as plt
import math
cap = cv2.VideoCapture(0)


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

def nothing(x):
    pass

def loadImg():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", help="cube.jpeg")
    args = vars(ap.parse_args())

    # load the image
    image = cv2.imread(args["image"])
    imageScaled = cv2.resize(image, (360, 480))

    return imageScaled


def loadImgFromCam():

    # load the image
    ret, image = cap.read()
    imageScaled = cv2.resize(image, (720, 480))
    imageScaled = cv2.flip(imageScaled,1)
    return imageScaled


def ImgToHSV(scaledImg):
    return cv2.cvtColor(cv2.cvtColor(scaledImg, cv2.COLOR_BGRA2BGR), cv2.COLOR_BGR2HSV)

def initWindows():
    # Create a window
    cv2.namedWindow('sliders')
    cv2.resizeWindow('sliders',300,400)

    # create trackbars for color change
    cv2.createTrackbar('R1','sliders',0,255,nothing)
    cv2.createTrackbar('G1','sliders',0,255,nothing)
    cv2.createTrackbar('B1','sliders',0,255,nothing)
    cv2.createTrackbar('R2','sliders',0,255,nothing)
    cv2.createTrackbar('G2','sliders',0,255,nothing)
    cv2.createTrackbar('B2','sliders',0,255,nothing)

    #cube color switches
    cv2.createTrackbar('Purple','sliders',0,1,nothing)
    cv2.createTrackbar('Green', 'sliders', 0, 1, nothing)
    cv2.createTrackbar('Orange', 'sliders', 0, 1, nothing)
    cv2.createTrackbar('Calibration', 'sliders', 0, 1, nothing)

def makeBoundaryArray(orange1HSV = [],orange2HSV = [],green1HSV = [],green2HSV =[], purple1HSV = [],purple2HSV = []):
    # define the list of boundaries HSV
    boundaries = [
        (orange1HSV,orange2HSV ),
        (green1HSV,green2HSV ),
        (purple1HSV,purple2HSV)
    ]

    boundariesArray = np.array(boundaries, dtype="uint8")
    return boundariesArray

def manualSliders(hsvImage):
    # get current positions of four trackbars
    r1 = cv2.getTrackbarPos('R1', 'sliders')
    g1 = cv2.getTrackbarPos('G1', 'sliders')
    b1 = cv2.getTrackbarPos('B1', 'sliders')
    r2 = cv2.getTrackbarPos('R2', 'sliders')
    g2 = cv2.getTrackbarPos('G2', 'sliders')
    b2 = cv2.getTrackbarPos('B2', 'sliders')

    lower = np.array([r1,g1,b1], dtype = "uint8")
    upper = np.array([r2,g2,b2], dtype = "uint8")

    return cv2.inRange(hsvImage, lower, upper)



def makeMask(hsvImage, lowBound = [], highBound = []):
    return cv2.inRange(hsvImage, lowBound, highBound)


def dilateErode(mask):
    kernel = np.ones((25, 25), np.uint8)
    kernel1 = np.ones((10, 10), np.uint8)
    dilateMask = cv2.dilate(mask, kernel1)
    erodedMask = cv2.erode(dilateMask, kernel)
    dilateMask2 = cv2.dilate(erodedMask, kernel1)
    return dilateMask2

def readSwtiches():
    OR = cv2.getTrackbarPos('Orange', 'sliders')
    GR = cv2.getTrackbarPos('Green', 'sliders')
    PU = cv2.getTrackbarPos('Purple', 'sliders')
    return np.array([OR,GR,PU], dtype="uint8")

def getBlobParam():
    params = cv2.SimpleBlobDetector_Params()
    params.filterByInertia = True
    params.minInertiaRatio = 0.1
    params.maxInertiaRatio = 1
    params.filterByConvexity = False
    params.minConvexity
    params.filterByColor = True
    params.blobColor = 255
    params.filterByArea = True
    params.minArea = 100
    params.maxArea = 100000
    params.filterByCircularity = True
    params.minCircularity = 0.5
    return params

def calibrate(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (cbcol, cbrow), None)

    # If found, add object points, image points (after refining them)
    if ret:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        # img = cv2.drawChessboardCorners(img, (cbrow,cbcol), corners2,ret)

        xvals = []
        angles = [0]  # 0 for center vertex, need a odd number for cbcol
        for i in range(cbcol):
            xvals.append(corners2[i][0][0])
        distance = 10  # in inches from camera to calibration board
        cubeSize = 0.72  # in inches
        for i in range(math.floor(cbcol / 2)):
            angles.append(math.atan((i + 1) * cubeSize / distance) * round(180 / math.pi))
            angles.insert(0, (math.atan((i + 1) * cubeSize / distance) * -1) * round(180 / math.pi))

        x = np.array(xvals)
        y = np.array(angles)

        m = (len(x) * np.sum(x * y) - np.sum(x) * np.sum(y)) / (len(x) * np.sum(x * x) - np.sum(x) * np.sum(x))
        b = (np.sum(y) - m * np.sum(x)) / len(x)
        print(m, b)

        plt.scatter(x, y)
        plt.xlabel("X value of center pixel")
        plt.ylabel("Assumed angle in deg")
        plt.show()

        cv2.imshow('img', img)
        cv2.waitKey(30)
    else:
        cv2.imshow('img', img)
        cv2.waitKey(30)

imageScaled = loadImgFromCam()
hsvImage = ImgToHSV(imageScaled)
initWindows()

orangeLow = [4, 139, 137]
orangeHigh = [62, 255, 255]
greenLow = [118, 107, 16]
greenHigh = [255, 255, 255]
purpleLow = [55, 107, 16]
purpleHigh = [97, 255, 255]

boundArray = makeBoundaryArray(orangeLow,orangeHigh,purpleLow,purpleHigh,greenLow,greenHigh)

while(1):
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

    imageScaled = loadImgFromCam()
    #imageScaled = loadImg()
    im_with_keypoints = imageScaled
    hsvImage = ImgToHSV(imageScaled)
    colorsToFind = readSwtiches()

    if(cv2.getTrackbarPos('Calibration', 'sliders') == 1):
        calibrate(imageScaled)

    # finalMask = manualSliders(hsvImage)

    detector = cv2.SimpleBlobDetector_create(getBlobParam())
    i = 0
    for color in colorsToFind:
        if color == 1:
            finalMask = dilateErode(makeMask(hsvImage, boundArray[i][0], boundArray[i][1]))
            keypoints = detector.detect(finalMask)
            largestBlob = 0
            for keypoint in keypoints:
                if largestBlob != 0:
                    if keypoint.size > largestBlob.size:
                        largestBlob = keypoint
                else:
                    largestBlob = keypoint
            if largestBlob != 0:
                im_with_keypoints = cv2.drawKeypoints(im_with_keypoints, [largestBlob], np.array([]), (0, 0, 255),
                                                  cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                print("X: " + str(largestBlob.pt[0]))
                print("Y: " + str(largestBlob.pt[1]))
                im_with_keypoints = cv2.circle(im_with_keypoints, (int(largestBlob.pt[0]), int(largestBlob.pt[1])), 5,
                                               (100, 100, 100), 6)
        i = i + 1

    cv2.imshow("images", im_with_keypoints)






    #Unsued Edge Detection
    # # noise removal
    # kernel = np.ones((3, 3), np.uint8)
    # sure_bg = cv2.dilate(finalMask, kernel, iterations=3)
    # # Finding sure foreground area
    # dist_transform = cv2.distanceTransform(finalMask, cv2.DIST_L2, 5)
    # ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    # # Finding unknown region
    # sure_fg = np.uint8(sure_fg)
    # unknown = cv2.subtract(sure_bg, sure_fg)
    # # Marker labelling
    # ret, markers = cv2.connectedComponents(sure_fg)
    # # Add one to all labels so that sure background is not 0, but 1
    # markers = markers + 1
    # # Now, mark the region of unknown with zero
    # markers[unknown == 255] = 0

    #output = cv2.bitwise_and(imageScaled, imageScaled, mask=finalMask)

    # markers = cv2.watershed(output, markers)
    # output[markers == -1] = [255, 0, 0]