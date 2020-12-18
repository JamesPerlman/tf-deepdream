import os
from dream_funcs import layer, render_deepdream, T
from PIL import Image, ImageChops
import numpy as np

base_dir = "E:/2020/Instagram/35. Pizza/dream/"

in_dir = base_dir + "in/"
out_dir = base_dir + "out/"

files = os.listdir(in_dir)
num_files = len(files)

layer = 'mixed5b_1x1_pre_relu'
nMax = 384
n_iter = 200
step = 1.5
octaves = 8
octave_scale = 1.5


sample_input = "images/noise.jpg"

#img_noise = np.random.uniform(size=(1024, 1024,3)) + 128.0
#im = Image.fromarray(np.uint8(img_noise))
#im.save('images/noise.jpg')
tensor = T(layer)

def dream(img_in, img_out, channel):
    render_deepdream(tensor[:,:,:,channel], img_in, img_out, n_iter, step, octaves, octave_scale)

for i in range(0, nMax):
    filename = files[i]
    sample_dir = "samples/" + layer
    if not os.path.exists(sample_dir):
        os.mkdir(sample_dir)
    
    old_img_name = sample_dir + "/" + str(i) + ".jpg"
    img_name = sample_dir + "/" + layer + " chan=" + str(i) + " iter=" + str(n_iter) + " oct=" + str(octaves) + " scale=" + str(octave_scale) + ".jpg"
    if os.path.exists(old_img_name):
        os.rename(old_img_name, img_name)

    if not os.path.exists(img_name):
        dream(sample_input, img_name, i)

