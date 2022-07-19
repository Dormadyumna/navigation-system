import cv2
from cv2 import contourArea

def getContours(imgCanny, imgCopy):
    contours, _ = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for contour in contours:

        area = cv2.contourArea(contour)
        
        if area > 400:
            cv2.drawContours(imgCopy, contour, -1, (0,255,0), 2) 



img = cv2.imread("squareTiles.jpg")
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgCanny = cv2.Canny(imgGray,50,50)
imgCopy = img.copy()

getContours(imgCanny, imgCopy)

cv2.imshow('Contour Detection', imgCanny)
cv2.waitKey(10000000)

