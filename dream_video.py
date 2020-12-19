import os
import csv
from dream_funcs import DeepDream
from PIL import Image, ImageChops
import numpy as np

# deepdream setup
blend_frames = True
alpha_step = 0.5

layer = 'mixed4d_5x5'
channel = 43
n_iter = 50
step = 1.5
octaves = 8
octave_scale = 1.5

dd = DeepDream(layer, (lambda T: T[:,:,:,channel]))

# file data
base_dir = "E:/2020/Instagram/21. Parkour/source/dream/"
in_dir = base_dir + "in/"
out_dir = "{} n_iter={:d} step={:.2f} oct={:d} scale={:.2f} blend={:.2f}".format(base_dir + layer, n_iter, step, octaves, octave_scale, alpha_step) + "/"

if not os.path.exists(out_dir):
    os.mkdir(out_dir)

files = os.listdir(in_dir)
num_files = len(files)

motionData = list(csv.DictReader(open(base_dir + "motion.csv")))


def blend(img_in, img_out, img_blend, alpha, scale, translate):
    image1 = Image.open(img_in).convert("RGBA")
    image2 = Image.open(img_out).convert("RGBA")
    image3 = Image.open(img_out).convert("RGBA")

    w, h = image1.size
    
    # alpha-blend the images
    sx, sy = (1, 1) #np.divide(scale, 100)
    tx, ty = map(int, translate)

    image2 = image2.resize((int(w * sx), int(h * sy)))
    image3.paste(image2, (-tx - int(w * (1 - 1 / sx) / 2), -ty - int(h * (1 - 1 / sy) / 2)))
    image2 = image3

    alphaBlended = ImageChops.blend(image1, image2, alpha=alpha)
    rgb_img = alphaBlended.convert('RGB')
    rgb_img.save(img_blend)

for i in range(0, num_files):
    filename = files[i]
    img_out = out_dir + filename

    scale = motionData[i]["Scale"]
    position = motionData[i]["Position"]
    
    sx, sy, sz = map(float, scale.split(","))
    px, py, pz = map(float, position.split(","))

    if os.path.exists(img_out):
        print("Output file already exists. Skipping.\n")
        continue

    # if i > 0, use blended image (created from previous iteration) as input
    if i > 0 and blend_frames:
        img_in = out_dir + "blended_" + filename
        if not os.path.exists(img_in):
            img_cur = in_dir + files[i]
            dream_last = out_dir + files[i - 1]
            blend(img_cur, dream_last, img_in, alpha_step, scale=(sx, sy), translate=(px, py))
    else:
        img_in = in_dir + filename

    print("dreaming" + img_in + " to " + img_out)
    
    dd.render(img_in, img_out, n_iter, step, octaves, octave_scale)

    # remove blended file
    if i > 0 and blend_frames:
        os.remove(img_in)

    # if i < len(files) then we blend the dreamed output with the raw input of the next frame
    if i + 1 < num_files and blend_frames:
        nextfile = files[i + 1]
        img_next = in_dir + nextfile
        img_blend = out_dir + "blended_" + nextfile
        blend(img_next, img_out, img_blend, alpha_step, scale=(sx, sy), translate=(px, py))
    
dd.cleanup()