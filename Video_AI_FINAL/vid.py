from model.srgan import generator
from model import resolve_single
import utils
import cv2
import numpy as np
import tensorflow as tf
import sys

LR_VID_LOCATION = 'vid/lr/1.mp4'
HR_VID_LOCATION = 'vid/hr/1.mp4'

def main():
	try:
		LR_VID_LOCATION = sys.argv[1]
	except AssertionError as e:
		# Low resolution video path must be provided in first argument.
		print(e)
	try:
		HR_VID_LOCATION = sys.argv[2]
	except AssertionError as e:
		# High resolution video path must be be provided in second argument
		print(e)

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

	# SRGAN
	srgan_model = generator()
	srgan_model.load_weights('weights/srgan/gan_generator.h5')

	cap = cv2.VideoCapture(LR_VID_LOCATION)

	success, frame = cap.read()
	if success == True:
		upscale = resolve_single(srgan_model, frame)
		width = int(upscale.shape[1])
		height = int(upscale.shape[0])
		dim=(width,height)
		fourcc = cv2.VideoWriter_fourcc(*'M','J','P','G')
		fps = cap.get(cv2.CAP_PROP_FPS)
		out = cv2.VideoWriter(HR_VID_LOCATION, fourcc, fps, dim)

		while success == True:
			upscale = resolve_single(srgan_model, frame).numpy()
			out.write(upscale)

	cap.release()
	out.release()



if __name__ == "__main__":
	main()