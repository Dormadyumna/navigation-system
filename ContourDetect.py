import cv2
from cv2 import contourArea
import cv2.aruco as aruco
import numpy as np
import os

# return augDict
def loadImages (path):

    myList = os.listdir(path)
    print(myList)
    noOfMarkers = len(myList)
    print("Total number of markers detected:", noOfMarkers)
    augDict = {}

    for imgPath in myList:
        key = int(os.path.splitext (imgPath)[0])
        imgAug = cv2.imread(f'{path}\{imgPath}')
        augDict[key] = imgAug
        print(key)

    return augDict

# return [bboxs, ids]
def findArucoMarkers(img, markerSize = 6, totalMarkers = 250, draw = True):

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # key = getattr(aruco, 'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    arucoDict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    arucoParam = aruco.DetectorParameters_create()
    bboxs, ids, _ = aruco.detectMarkers(imgGray, arucoDict, parameters = arucoParam)

    # print(ids)
    if draw:
        aruco.drawDetectedMarkers(img, bboxs)
    
    return [bboxs, ids]

# cv2.drawContours(imgCopy, contour, -1, (0,255,0), 2)
def getContours(imgCanny, imgCopy):
    contours, _ = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    i=0
    for contour in contours:

        area = cv2.contourArea(contour)
        if area > 500:
            perimeter = cv2.arcLength(contour,True)
            epsilon = 0.1*perimeter
            approx = cv2.approxPolyDP(contour,epsilon,True)
            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(imgCopy,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(imgCopy, str(i), (x + (w//2),y + (h//2)), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 2, cv2.FILLED)
            i = i+1

#MAIN
augDict= loadImages('ArucoDB')
img = cv2.imread('Images\squareTiles.jpg')
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgCanny = cv2.Canny(imgGray,50,50)
imgCopy = img.copy()


arucoFound = findArucoMarkers(img)


if len(arucoFound[0])!= 0:
    for bbox, id in zip(arucoFound[0], arucoFound[1]):
        if int(id) in augDict.keys():
            print(id)

getContours(imgCanny, imgCopy)

cv2.imshow('Contour Detection', imgCopy)
cv2.waitKey(10000000)

