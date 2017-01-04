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
blurParam = 10
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
        XY = []
        cut_faces = []
        for (x, y, w, h) in faces:
            cut_face = image[y:y+h, x:x+w].copy()
            image[y:y+h, x:x+w] = [0,0,0]
            useLightImg[y:y+h, x:x+w] = [0, 0, 0]
            cv2.imshow("light", useLightImg)
            cut_faces.append(cut_face)
            XY.append((x,y,w,h))
        # cv2.imshow("cut_face", cut_face)
        finalImage = cv2.addWeighted(image, 0.7, useLightImg, 0.3, 0)

        for i in range(len(XY)):
            tmp_face = cut_faces[i]
    	   # cv2.imshow("test3", tmp_face)
            (x, y, w, h) = XY[i]
            finalImage[y:y+h, x:x+w] = tmp_face*0.7
            finalImage[y:y+h, x-blurParam:x+blurParam] = cv2.GaussianBlur(finalImage[y:y+h, x-blurParam:x+blurParam],(5,5),0)
            finalImage[y:y+h, x+w-blurParam:x+w+blurParam] = cv2.GaussianBlur(finalImage[y:y+h, x+w-blurParam:x+w+blurParam],(5,5),0)
            finalImage[y-blurParam:y+blurParam, x:x+w] = cv2.GaussianBlur(finalImage[y-blurParam:y+blurParam, x:x+w],(5,5),0)
            finalImage[y+h-blurParam:y+h+blurParam, x:x+w] = cv2.GaussianBlur(finalImage[y+h-blurParam:y+h+blurParam, x:x+w],(5,5),0)

            
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