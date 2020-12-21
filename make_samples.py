import os
from dream_funcs import DeepDream
import numpy as np

# layer = 'mixed5b_1x1_pre_relu' unfinished
layer = 'mixed5b_1x1_pre_relu'
nMax = 128
n_iter = 200
step = 1.5
octaves = 8
octave_scale = 1.5

sample_input = "images/noise.jpg"

for i in range(0, nMax):
    sample_dir = "samples/" + layer
    if not os.path.exists(sample_dir):
        os.mkdir(sample_dir)
    
    old_img_name = sample_dir + "/" + str(i) + ".jpg"
    img_name = sample_dir + "/" + layer + " chan=" + str(i) + " iter=" + str(n_iter) + " oct=" + str(octaves) + " scale=" + str(octave_scale) + ".jpg"
    if os.path.exists(old_img_name):
        os.rename(old_img_name, img_name)

    if not os.path.exists(img_name):
        dd = DeepDream(layer, (lambda T: T[:,:,:,i]))
        dd.render(sample_input, img_name, n_iter, step, octaves)
        dd.cleanup()
