import cv
import cv2
import numpy as np

#input: image in numpy array
#output: numpy array of mouth

def show(image):
	cv2.imshow('Image', np.asmatrix(image))
	cv2.waitKey(0)
	cv2.destroyAllWindows()


def crop(image, x, y, w, h):
	return image[y : y + h, x : x + w]

def detect_mouth(image):

	facemodel = cv.Load('face_model')
	mouthmodel = cv.Load('mouth_model')
	storage = cv.CreateMemStorage()
	faces = cv.HaarDetectObjects(image, facemodel, storage)

	bigFace = 0
	bigFaceSize = 0
	for face in faces:
		if(face[0][2] * face[0][3] > bigFaceSize):
			bigFace = face
			bigFaceSize = face[0][2] * face[0][3]
	
	if(bigFace == 0): #NO FACE DETECTED
		return 0
	
	mouths = cv.HaarDetectObjects(image, mouthmodel, storage)
	bigMouth = 0
	bigMouthSize = 0
	for mouth in mouths:
		if(mouth[0][2] * mouth[0][3] > bigMouthSize and
		   mouth[0][1] + mouth[0][3] < bigFace[0][1] + bigFace[0][3] and
		   mouth[0][1] > bigFace[0][1] + bigFace[0][3] / 2):
			bigMouth = mouth
			bigMouthSize = mouth[0][2] * mouth[0][3]

	if(bigMouth == 0):
		return 0

	return crop(image, bigMouth[0][0], bigMouth[0][1], bigMouth[0][2], bigMouth[0][3])








	


