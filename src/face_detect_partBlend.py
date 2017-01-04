import cv2
import sys
import numpy as np
import smile_detect as sd
import imutils


# Get user supplied values
cascPath = sys.argv[1]
lightPath = sys.argv[2]
# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

lightImg = cv2.imread(lightPath)
video_capture = cv2.VideoCapture(0)
video_capture.set(3, 480)
video_capture.set(4, 320)
##deal with light Image
resizeLightImg = cv2.resize(lightImg, (480, 320))
while True:
    useLightImg = resizeLightImg.copy()
    cv2.imshow("light2",useLightImg)
    ret, image = video_capture.read()
# Read the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50),
        flags = cv2.cv.CV_HAAR_SCALE_IMAGE
    )
# Draw a rectangle around the faces

    
    if sd.isSmile(image):
        # cv2.imshow("cut_face", cut_face)
        last_x = 0
        last_y = 0
        last_w = 320
        last_h = 480
        tmpFianlImage = image.copy()
        tmpFianlImage[0:320, 0:480] = [255, 255, 255]
        finalImage[0:320, 0:480] = [0, 0, 0]
        cv2.imshow('tmpfinal', tmpFianlImage)
        for i in xrange(81):
            tmpImage = image.copy()
            x = 2*i
            y = 3*i
            w = 320 - 4*i
            h = 480 - 6*i
            tmpImage = cv2.addWeighted(tmpImage, 0.7, useLightImg, 0.8 - (i+1)*0.01, 0)
            cv2.imshow("tmp", tmpImage)
            tmpFianlImage[last_x:last_x+2, last_y:last_y+last_h] = tmpImage[last_x:last_x+2, last_y:last_y+last_h].copy()
            tmpFianlImage[last_x+last_w-2:last_x+last_w, last_y:last_y+last_h] = tmpImage[last_x+last_w-2:last_x+last_w, last_y:last_y+last_h].copy()
            tmpFianlImage[last_x:last_x+last_w, last_y:last_y+3] = tmpImage[last_x:last_x+last_w, last_y:last_y+3].copy()
            tmpFianlImage[last_x:last_x+last_w, last_y+last_h-3:last_y+last_h] = tmpImage[last_x:last_x+last_w, last_y+last_h-3:last_y+last_h].copy()
            cv2.imshow('tmp2', tmpFianlImage)
            last_x = x
            last_y = y
            last_w = w
            last_h = h
        finalImage = tmpFianlImage
            
    else:
        finalImage = image
    cv2.imshow("test2", finalImage)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#cv2.waitKey(0)
'''cv2.imshow("Faces found", image)
if cv2.waitKey(1) & 0xFF == ord('q'):
	break'''
video_capture.release()
cv2.destroyAllWindows()