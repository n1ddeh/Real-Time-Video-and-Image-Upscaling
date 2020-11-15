import cv2
import numpy as np
from model.srgan import generator
from model import resolve_single
import utils
import os.path

print("\n------------------------\nInput video file name")

video_name = input()

video = 'Videos/'+video_name

if os.path.exists(video) == False:
    print("Video {} does not exist".format(video_name))
    exit(1)

print("\n------------------------\n")
model = generator()
model.load_weights('weights/srgan/gan_generator.h5')

#lr = utils.load_image('demo/0805x4-crop.png')
#sr = resolve_single(model, lr)
#utils.plot_sample(lr, sr)
cap = cv2.VideoCapture(video)
ret, frame = cap.read()


scale = 30
width = int(frame.shape[1] * (scale/100))
height = int(frame.shape[0] * (scale/100))
dim = (width, height)

crop = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
upscale = resolve_single(model, crop).numpy()

print("Upscale Width: {}".format(upscale.shape[1]))
print("Upscale Height: {}".format(upscale.shape[0]))

crop_out = cv2.VideoWriter('Outputs/crop.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, dim)
upscale_out = cv2.VideoWriter('Outputs/upscale.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (upscale.shape[1],upscale.shape[0]))
i = 1
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
        crop = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
        upscale = resolve_single(model, crop).numpy()

        crop_out.write(crop)
        upscale_out.write(upscale)

    else:
        break
    
print("Total Writes: {}".format(i))
print("\n---------------------\nDONE EDITING VIDEOS\n---------------------\n")

# When everything done, release the capture
cap.release()

crop_out.release()
upscale_out.release()

cv2.destroyAllWindows()
