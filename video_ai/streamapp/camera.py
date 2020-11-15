import tensorflow as tf
from imutils.video import VideoStream
import imutils
import cv2,os,urllib.request
import numpy as np
from django.conf import settings
from model.srgan import generator
from model.wdsr import wdsr_b
from model import resolve_single

# SRGAN
srgan_model = generator()
srgan_model.load_weights('weights/srgan/gan_generator.h5')

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


class VideoCamera(object):
	def __init__(self):
		self.video = cv2.VideoCapture(0)
		self.image = self.get_frame()

	def __del__(self):
		self.video.release()

	def get_frame(self):
		success, image = self.video.read() # Bool, numpy array

		# We are using Motion JPEG, but OpenCV defaults to capture raw images,
		# so we must encode it into JPEG in order to correctly display the
		# video stream
		ret, jpeg = cv2.imencode('.jpg', image)
		return jpeg.tobytes()



class CompressedCamera(object):
	def __init__(self, scale=30):
		self.video = cv2.VideoCapture(0)
		self.image = self.get_frame()
		self.scale = scale

	def __del__(self):
		self.video.release()

	def get_frame(self):
		success, image = self.video.read() # Bool, numpy array
		width = int(image.shape[1] * (self.scale/100))
		height = int(image.shape[0] * (self.scale/100))
		dim = (width, height)
		crop = cv2.resize(frame, dim, interpolation = cv2.INTER_CUBIC)
		ret, jpeg = cv2.imencode('.jpg', crop)
		return jpeg.tobytes()



