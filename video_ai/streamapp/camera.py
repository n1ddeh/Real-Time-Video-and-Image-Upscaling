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
	def __init__(self, scale=20):
		self.video = cv2.VideoCapture(0)
		self.success, self.frame = self.video.read() # Bool, numpy array
		
		self.scale = scale
		self.compressed_image = self.compress_frame() # numpy array

		self.upscaled_frame = self.upscale_frame() # numpy array

	def __del__(self):
		self.video.release()

	def get_image(self):
		# We are using Motion JPEG, but OpenCV defaults to capture raw images,
		# so we must encode it into JPEG in order to correctly display the
		# video stream
		ret, jpeg = cv2.imencode('.jpg', self.frame)
		return jpeg.tobytes()

	def compress_frame(self):
		width = int(self.frame.shape[1] * (self.scale/100))
		height = int(self.frame.shape[0] * (self.scale/100))
		dim = (width, height)
		compression = cv2.resize(self.frame, dim, interpolation = cv2.INTER_CUBIC)
		return compression

	def upscale_frame(self):
		upscale = resolve_single(srgan_model, self.compressed_image).numpy()
		ret, jpeg = cv2.imencode('.jpg', upscale)
		return jpeg
		

	def get_scale(self):
		return self.scale

	def get_compressed_image(self):
		ret, jpeg = cv2.imencode('.jpg', self.compressed_image)
		return jpeg

	def get_upscaled_image(self):
		ret, jpeg = cv2.imencode('.jpg', self.upscale)
		return jpeg