import numpy as np
import logistic
import csv
import cv
import cv2
from PIL import Image

WIDTH, HEIGHT = 28, 10
dim = WIDTH * HEIGHT

def vectorize(filename):
	size = WIDTH, HEIGHT
	im = Image.open(filename)
	im = im.resize(size, Image.ANTIALIAS).convert('L')
	return np.array(im).reshape(1, size[0] * size[1])

smilefiles = []
with open('smiles.csv', 'rb') as csvfile:
	for rec in csv.reader(csvfile, delimiter='	'):
		smilefiles += rec

neutralfiles = []
with open('neutral.csv', 'rb') as csvfile:
	for rec in csv.reader(csvfile, delimiter='	'):
        	neutralfiles += rec

phi = np.zeros((len(smilefiles) + len(neutralfiles), dim))
labels = []

PATH = "../data/smile/"
for idx, filename in enumerate(smilefiles):
	phi[idx] = vectorize(PATH + filename)
	labels.append(1)

PATH = "../data/neutral/"
offset = idx + 1
for idx, filename in enumerate(neutralfiles):
	phi[idx + offset] = vectorize(PATH + filename)
	labels.append(0)

lr = logistic.Logistic(dim)
lr.train(phi, labels)
np.save("smile_model", lr.weights)



