import os
import csv
from dream_funcs import DeepDream
from PIL import Image, ImageChops
import numpy as np
from image_funcs import blended

# deepdream setup
blend_frames = True
alpha_step = 0.5

layer = 'mixed4d_3x3'
channel = 1
n_iter = 40
step = 1.5
octaves = 8
octave_scale = 1.5

dd = DeepDream(layer, (lambda T: T[:,:,:,channel]))

# file data
base_dir = "E:/2020/Instagram/21. Parkour/source/dream/"
in_dir = base_dir + "in/"
out_dir = base_dir + "layer2/"

if not os.path.exists(out_dir):
    os.mkdir(out_dir)

out_dir = "{} ch={:d} n={:d}  st={:.2f} oc={:d} sc={:.2f} bl={:.2f}".format(out_dir + layer, channel, n_iter, step, octaves, octave_scale, alpha_step) + "/"

if not os.path.exists(out_dir):
    os.mkdir(out_dir)

files = os.listdir(in_dir)
num_files = len(files)

motionData = list(csv.DictReader(open(base_dir + "layer2.csv")))

for i in range(0, num_files):
    filename = files[i]
    img_out = out_dir + filename

    position = motionData[i]["Position"]
    
    sx, sy, sz = (100,100,100)
    px, py, pz = map(float, position.split(","))

    if os.path.exists(img_out):
        print("Output file already exists. Skipping: " + filename)
        continue

    # if i > 0, use blended image (created from previous iteration) as input
    if i > 0 and blend_frames:
        img_in = out_dir + "blended_" + filename
        if not os.path.exists(img_in):
            img_cur = in_dir + files[i]
            dream_last = out_dir + files[i - 1]
            blended_img = blended(img_cur, dream_last, alpha_step, scale=(sx, sy), translate=(px, py))
            blended_img.save(img_in)
    else:
        img_in = in_dir + filename

    print("Dreaming: {:d} of {:d} ({:.1f}%): ".format(i, num_files, 100 * float(i) / float(num_files)) + filename)
    
    dd.render(img_in, img_out, n_iter, step, octaves, octave_scale)

    # remove blended file
    if i > 0 and blend_frames:
        os.remove(img_in)

    # if i < len(files) then we blend the dreamed output with the raw input of the next frame
    if i + 1 < num_files and blend_frames:
        nextfile = files[i + 1]
        img_next = in_dir + nextfile
        out_path = out_dir + "blended_" + nextfile
        blended_img = blended(img_next, img_out, alpha_step, scale=(sx, sy), translate=(px, py))
        blended_img.save(out_path)
    
dd.cleanup()