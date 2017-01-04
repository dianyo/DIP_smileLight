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
smileDegree = 0
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
    	smileDegree += 3
    	if smileDegree > 15:
    		smileDegree = 15
        # cv2.imshow("cut_face", cut_face)

            
    else:
    	smileDegree -= 1
    	if smileDegree < 0:
    		smileDegree = 0
    finalImage = cv2.addWeighted(image, 0.7, useLightImg, 0.05 * smileDegree, 0)
    cv2.imshow("test2", finalImage)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#cv2.waitKey(0)
'''cv2.imshow("Faces found", image)
if cv2.waitKey(1) & 0xFF == ord('q'):
	break'''
video_capture.release()
cv2.destroyAllWindows()