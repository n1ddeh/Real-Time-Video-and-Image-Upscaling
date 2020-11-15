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
	def __init__(self, scale=20):
		self.video = cv2.VideoCapture(0)
		self.image = self.get_frame()
		success, self.frame = self.video.read()
		self.scale = scale
		self.compressed_image = self.compress_frame()
		self.upscaled_frame = self.upscale_frame()

	def __del__(self):
		self.video.release()

	def get_frame(self):
		success, self.image = self.video.read() 
		# We are using Motion JPEG, but OpenCV defaults to capture raw images,
		# so we must encode it into JPEG in order to correctly display the
		# video stream
		ret, jpeg = cv2.imencode('.jpg', self.image)
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
		

	def get_image(self):
		return self.image

	def get_scale(self):
		return self.scale

	def get_compressed_image(self):
		return self.compressed_image

	def get_upscaled_image(self):
		return self.upscaled_frame