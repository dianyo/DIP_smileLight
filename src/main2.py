import mouth_detector as md
import numpy as np
import cv
import cv2
from PIL import Image

WIDTH, HEIGHT = 28, 10
dim = WIDTH * HEIGHT


img = cv2.imread('testn.jpg', 0)
mouth = md.detect_mouth(cv.fromarray(img))



          	







