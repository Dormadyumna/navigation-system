import cv2
import cv2.aruco as aruco
import numpy as np
import os

def loadImages (path):

    myList = os.listdir(path)
    noOfMarkers = len(myList)
    print("Total number of markers detected:", noOfMarkers)
    augDict = {}

    for imgPath in myList:
        key = int(os.path.splitext (imgPath)[0])
        imgAug = cv2.imread('{path}/{imgPath}')
        augDict[key] = imgAug
        print(key)

    return augDict

def findArucoMarkers(img, markerSize = 6, totalMarkers = 250, draw = True):

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # key = getattr(aruco, 'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    arucoDict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    arucoParam = aruco.DetectorParameters_create()
    bboxs, ids, _ = aruco.detectMarkers(imgGray, arucoDict, parameters = arucoParam)

    # print(ids)
    if draw:
        aruco.drawDetectedMarkers(img, bboxs)
    
    print(bboxs)
    return [bboxs, ids]

#MAIN
augDict= loadImages("ArucoDB")

img = cv2.imread('singlemarkerssource.png')

arucoFound = findArucoMarkers(img)

if len(arucoFound[0])!= 0:
    for bbox, id in zip(arucoFound[0], arucoFound[1]):
        if int(id) in augDict.keys():
            print(id)

    cv2.imshow("Image", img)
    cv2.waitKey(100000)

