from model.srgan import generator
from model.wdsr import wdsr_b
from model import resolve_single
import utils
import cv2
import numpy as np
import tensorflow as tf
import sys



def main():
	scale = 30
	if sys.argv[1]:
		scale = int(sys.argv[1])

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

	cap = cv2.VideoCapture(0)
	ret, frame = cap.read()

	while(True):
		# Get dimentions of cropped image.
		width = int(frame.shape[1] * (scale/100))
		height = int(frame.shape[0] * (scale/100))
		dim = (width, height)

		# Capture frame-by-frame
		ret, frame = cap.read()

		# Crop image by scale
		crop = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

		# Upscale image
		upscale = resolve_single(srgan_model, crop).numpy()
		
		# Resize crop and upscaled image to be the same as input image.
		width = int(frame.shape[1])
		height = int(frame.shape[0])
		dim = (width, height)
		crop = cv2.resize(crop, dim, interpolation = cv2.INTER_AREA)
		upscale = cv2.resize(upscale, dim, interpolation = cv2.INTER_AREA)
		
		# Display the Resulting Frames
		cv2.imshow('input', frame)
		cv2.imshow('cropped', crop)
		cv2.imshow('srgan', upscale)
		
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	# When everything done, release the capture
	cap.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
	main()