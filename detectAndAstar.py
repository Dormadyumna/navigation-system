import cv2
import cv2.aruco as aruco
import numpy as np
import os
from queue import PriorityQueue

aruco_begin = 23
aruco_end = 98

begin = 0
last = 0

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



minContours = []
total = 0

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

    print("THE TOTAL NO. OF NODES ARE")
    print(i)
    global total
    total = i

class Spot:
    def __init__(self, x, y, w, h, id, left = -1, up = -1, right = -1, down = -1):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.left = -1
        self.up = -1
        self.right = -1
        self.down = -1
        self.id = id    
        self.neighbors = []
        self.centre = (x+w/2, y+h/2)

#MAIN
augDict= loadImages('ArucoDB')
img = cv2.imread('Images\pathAruco.png')
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgCanny = cv2.Canny(imgGray,50,50)
imgCopy = img.copy()

# getContours(imgCanny, imgCopy)

# cv2.imshow('Contour Detection', imgCopy)
# cv2.waitKey(10000000)

arucoFound = findArucoMarkers(img)
print(arucoFound)

arucoDetected = []

if len(arucoFound[0])!= 0:
    for bbox, id in zip(arucoFound[0], arucoFound[1]):
        if int(id) in augDict.keys():
            arucoDetected.append([id, (bbox[0][0][0]+bbox[0][2][0])/2, (bbox[0][0][1]+bbox[0][2][1])/2])
            print( (bbox[0][0][0]+bbox[0][2][0])/2, (bbox[0][0][1]+bbox[0][2][1])/2 )

begin_point = (0,0)
begin_orient = 0
last_point = (0,0)
last_orient = 0

print(arucoDetected)

if len(arucoFound[0])!= 0:
    for bbox, id in zip(arucoFound[0], arucoFound[1]):
        if int(id) == aruco_begin:
            begin_point = ((bbox[0][0][0]+bbox[0][2][0])/2, (bbox[0][0][1]+bbox[0][2][1])/2)
            begin_orient = bbox
        elif int(id) == aruco_end:
            last_point = ((bbox[0][0][0]+bbox[0][2][0])/2, (bbox[0][0][1]+bbox[0][2][1])/2)
            last_orient = bbox

print(begin_orient, last_orient)
print(begin_point, last_point)

# i = 0
# for cnt in minContours:
#     if(cv2.pointPolygonTest(cnt, begin_point, False)==1):
#         begin = i
#         print("begin", i)
#     elif(cv2.pointPolygonTest(cnt, last_point, False)==1):
#         last = i 
#         print("last", i)

getContours(imgCanny, imgCopy)

w_all = []
h_all = []

id = 0
spots = []

for cnt in minContours:
    x, y, w, h = cv2.boundingRect(cnt)
    w_all.append(w)
    h_all.append(h)
    spots.append(Spot(x,y,w,h,id))
    
    print(id)
    print( (x + w/2, y + h/2) )

    id = id + 1

for spot in spots:
    if(begin_point[0]>=spot.x and begin_point[0]<=(spot.x + spot.w) and begin_point[1]>=spot.y and begin_point[1]<=(spot.y+spot.h) ):
        begin = spot.id
    elif(last_point[0]>=spot.x and last_point[0]<=(spot.x + spot.w) and last_point[1]>=spot.y and last_point[1]<=(spot.y+spot.h) ):
        last = spot.id

w_sum = 0
h_sum = 0

for w in w_all:
    w_sum = w_sum + w

for h in h_all:
    h_sum = h_sum + h

w_av = w_sum/total
h_av = h_sum/total

for spot in spots:
    i = 0
    for cnt in minContours:
        # print(spot)
        if(cv2.pointPolygonTest(cnt, (spot.centre[0] + w_av, spot.centre[1]), False)==1):
            spot.right = i
            spot.neighbors.append(i)
        elif(cv2.pointPolygonTest(cnt, (spot.centre[0] - w_av, spot.centre[1]), False)==1):
            spot.neighbors.append(i)
            spot.left = i
        elif(cv2.pointPolygonTest(cnt, (spot.centre[0] , spot.centre[1] + h_av), False)==1):
            spot.neighbors.append(i)
            spot.down = i
        elif(cv2.pointPolygonTest(cnt, (spot.centre[0] , spot.centre[1] - h_av), False)==1):
            spot.up = i
            spot.neighbors.append(i)
        i = i + 1

for spot in spots:
    print(spot.id, spot.neighbors)


start = spots[begin]
end = spots[last]

hue = []
for spot in spots:
    hue.append( abs(spot.centre[0]-end.centre[0]) + abs(spot.centre[1]-end.centre[1]) )
    
count = 0
open_set = PriorityQueue()
open_set.put( (0, count, begin) ) #f_score, count(for tie breaking if 2 have same f_score), spot_id) 
came_from = {}
g_score = {spot: float("inf") for spot in range(0,i+1)}
g_score[begin] = 0
f_score = {spot: float("inf") for spot in range(0,i+1)}
f_score[begin] = abs(spots[0].centre[0]-spots[last].centre[0]) + abs(spots[0].centre[1]-spots[last].centre[1])

# print(f_score[0])

open_set_hash = {begin}

path = []

while not open_set.empty():
    current = open_set.get()[2]
    print(open_set_hash)
    print(current)
    path.append(current)
    
    open_set_hash.remove(current)

    if current == last:
        break
    
    for neighbor in spots[current].neighbors:
        temp_g_score = g_score[current] + 45

        if temp_g_score < g_score[neighbor]:
            came_from[neighbor] = current
            g_score[neighbor] = temp_g_score
            f_score[neighbor] = temp_g_score + hue[neighbor]

            if neighbor not in open_set_hash:
                count = count + 1
                open_set.put( (f_score[neighbor], count, neighbor) )
                open_set_hash.add(neighbor)

direction = [0]

def spotDirection(curr, prev):
    if(curr == spots[prev].left):
        print(curr, prev)
        print("left")
        return "left"
    elif(curr == spots[prev].up):
        print(curr, prev)
        print("up")
        return "up"
    elif(curr == spots[prev].right):
        print(curr, prev)
        print("right")
        return "right"    
    elif(curr == spots[prev].down):
        print(curr, prev)
        print("down")
        return "down"

pathDirection = []
for i in range(1, len(path)):
    pathDirection.append(spotDirection(path[i],path[i-1]))

print(pathDirection)

nonOrientedCommands = []
initial_orient = 0

print("begin_orient[0][0][0]", begin_orient[0][0][0])
print("begin_orient[0][0][1]", begin_orient[0][0][1])
print("begin_orient[0][2][0]", begin_orient[0][2][0])
print("begin_orient[0][2][1]", begin_orient[0][2][1])
print("begin_orient[0][0][0]", begin_orient[0][0][0])
print("begin_orient[0][0][1]", begin_orient[0][0][1])

if(begin_orient[0][0][0] <= begin_orient[0][1][0] and begin_orient[0][0][1] <= begin_orient[0][3][1]):
    initial_orient = "up"
elif(begin_orient[0][0][0] <= begin_orient[0][3][0] and begin_orient[0][0][1] >= begin_orient[0][1][1]):
    initial_orient = "right"
elif(begin_orient[0][0][0] >= begin_orient[0][1][0] and begin_orient[0][0][1] <= begin_orient[0][3][1]):
    initial_orient = "down"
elif(begin_orient[0][0][0] >= begin_orient[0][3][0] and begin_orient[0][0][1] <= begin_orient[0][1][1]):
    initial_orient = "left"

# print(initial_orient)


nonOrientedCommands.append("forward")
for i in range(0, len(pathDirection)-1):
    if(pathDirection[i+1] == pathDirection[i]):
        nonOrientedCommands.append("forward")
    elif( (pathDirection[i+1]=="up" and pathDirection[i]=="left") or (pathDirection[i+1]=="right" and pathDirection[i]=="up") or (pathDirection[i+1]=="down" and pathDirection[i]=="right") or (pathDirection[i+1]=="left" and pathDirection[i]=="down") ):
        nonOrientedCommands.append("right")
    elif( (pathDirection[i+1]=="down" and pathDirection[i]=="left") or (pathDirection[i+1]=="right" and pathDirection[i]=="down") or (pathDirection[i+1]=="up" and pathDirection[i]=="right") or (pathDirection[i+1]=="left" and pathDirection[i]=="up") ):
        nonOrientedCommands.append("left")
   
print(nonOrientedCommands)
commandSequence = nonOrientedCommands.copy()

if(pathDirection[0] == initial_orient):
    commandSequence[0] = "forward"
elif( (pathDirection[0]=="up" and initial_orient =="left") or (pathDirection[0]=="right" and initial_orient=="up") or (pathDirection[0]=="down" and initial_orient=="right") or (pathDirection[0]=="left" and initial_orient=="down") ):
    commandSequence[0] = "right"
elif( (pathDirection[0]=="down" and initial_orient=="left") or (pathDirection[0]=="right" and initial_orient=="down") or (pathDirection[0]=="up" and initial_orient=="right") or (pathDirection[0]=="left" and initial_orient=="up") ):
    commandSequence[0] = "left"
elif( (pathDirection[0]=="down" and initial_orient=="up") or (pathDirection[0]=="right" and initial_orient=="left") or (pathDirection[0]=="up" and initial_orient=="down") or (pathDirection[0]=="left" and initial_orient=="right") ):
    commandSequence[0]= "reverse"

print(commandSequence)

# for i in range(len(path)):
#     if 

# print(hue)
# print("\n\n")
# print(g_score)
# print("\n\n")
# print(f_score)

print(path)
# imgF = cv2.resize(img, (1600,800))
# cv2.imshow('ArucoDetection', imgF)
# cv2.waitKey(10000000)

# print(begin_orient[0][0][0])

imgFinal = cv2.resize(imgCopy, (1600,800))
findArucoMarkers(imgFinal)
cv2.imshow('Contour Detection', imgFinal)
cv2.waitKey(10000000)