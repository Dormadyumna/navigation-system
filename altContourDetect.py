import cv2
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
    arucoDict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    arucoParam = aruco.DetectorParameters_create()
    bboxs, ids, _ = aruco.detectMarkers(imgGray, arucoDict, parameters = arucoParam)

    # print(ids)
    if draw:
        aruco.drawDetectedMarkers(img, bboxs)
    
    return [bboxs, ids]

# cv2.drawContours(imgCopy, contour, -1, (0,255,0), 2)

class Spot:
	def __init__(self, x, y, w, h, id):
		self.x = x
		self.y = y
		self.w = w
		self.h = h
		self.id = id
		self.neighbors = []
		self.centre = (x+w/2, y+h/2)

minContours = []

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
            minContours.append(approx)            
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

id = 0
spots = []
for cnt in minContours:
    x, y, w, h = cv2.boundingRect(cnt)
    spots.append(Spot(x,y,w,h,id))
    id = id + 1

for spot in spots:
    i = 0
    for cnt in minContours:
        if(cv2.pointPolygonTest(cnt, (spot.centre[0] + w, spot.centre[1]), False)==1):
            spot.neighbors.append(i)
        elif(cv2.pointPolygonTest(cnt, (spot.centre[0] - w, spot.centre[1]), False)==1):
            spot.neighbors.append(i)
        elif(cv2.pointPolygonTest(cnt, (spot.centre[0] , spot.centre[1] + h), False)==1):
            spot.neighbors.append(i)
        elif(cv2.pointPolygonTest(cnt, (spot.centre[0] , spot.centre[1] - h), False)==1):
            spot.neighbors.append(i)
        i = i + 1

for spot in spots:
    print(spot.id, spot.neighbors)

start = spots[0]
end = spots[63]

# hue = []
# for spot in spots:
#     hue.append( abs(spot.centre[0]-end.centre[0]) + abs(spot.centre[1]-end.centre[1]) )

# print(hue)

# g = []

cv2.imshow('Contour Detection', imgCopy)
cv2.waitKey(10000000)
