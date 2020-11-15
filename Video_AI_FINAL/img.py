from model.srgan import generator
from model import resolve_single
import utils
import cv2
import numpy as np
import tensorflow as tf
import sys

LR_IMG_LOCATION = 'img/lr/1.png'
HR_IMG_LOCATION = 'img/hr/1.png'

def main():
	if sys.argv[1]:
		LR_IMG_LOCATION = sys.argv[1]

	if sys.argv[2]:
		HR_IMG_LOCATION = sys.argv[2]

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

	low_res = cv2.imread(LR_IMG_LOCATION)
	high_res = resolve_single(srgan_model, low_res).numpy()

	cv2.imwrite(HR_IMG_LOCATION, high_res, [cv2.IMWRITE_PNG_COMPRESSION, 9])



if __name__ == "__main__":
	main()
