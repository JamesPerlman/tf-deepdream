import os
from dream_funcs import layer, render_deepdream, T
from PIL import Image, ImageChops
import numpy as np

out_dir = "images/test/"

files = os.listdir(out_dir)
num_files = len(files)

layer = 'mixed5b_1x1'
n_iter = 128
step = 1.5
octaves = 8
octave_scale = 1.2
channel = 314


sample_input = "images/test/18.jpg"
# 139 flowers


#img_noise = np.random.uniform(size=(1024, 1024,3)) + 128.0
#im = Image.fromarray(np.uint8(img_noise))
#im.save('images/noise.jpg')
def dream(img_in, img_out, channel):
    render_deepdream(T(layer)[:,:,:,channel], img_in, img_out, n_iter, step, octaves, octave_scale)


dream(sample_input, out_dir + str(num_files) + ".jpg", channel)