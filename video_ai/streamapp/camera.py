import tensorflow
from imutils.video import VideoStream
import imutils
import cv2,os,urllib.request
import numpy as np
from django.conf import settings
from model.srgan import generator
from model.wdsr import wdsr_b
from model import resolve_single

# load our serialized face detector model from disk
# SRGAN
srgan_model = generator()
srgan_model.load_weights('weights/srgan/gan_generator.h5')

#WDSR
wdsr_model = wdsr_b(scale=4, num_res_blocks=32)
wdsr_model.load_weights('weights/wdsr/wdsr-b-32-x4.h5')


class VideoCamera(object):
	def __init__(self):
		self.video = cv2.VideoCapture(0)

	def __del__(self):
		self.video.release()

	def get_frame(self):
		success, image = self.video.read()
		# We are using Motion JPEG, but OpenCV defaults to capture raw images,
		# so we must encode it into JPEG in order to correctly display the
		# video stream
		
		ret, jpeg = cv2.imencode('.jpg', image)
		return jpeg.tobytes()
