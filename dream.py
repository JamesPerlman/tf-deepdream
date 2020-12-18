import os
from dream_funcs import layer, render_deepdream, T
from PIL import Image, ImageChops

base_dir = "E:/2020/Misc/elena-dream/dream/"

in_dir = base_dir + "in/"
out_dir = base_dir + "out/"

files = os.listdir(in_dir)
num_files = len(files)

layer = 'mixed4d_3x3_bottleneck_pre_relu'

blend_frames = True
scale = 1.0
alpha_step = 0.4

channel = 139

def dream(img_in, img_out, n_iter):
    render_deepdream(T(layer)[:,:,:,channel], img_in, img_out, n_iter, 1.5, 10, 1.5)

def blend(img_in, img_out, img_blend, alpha, i, n):
    image1 = Image.open(img_in).convert("RGBA")
    image2 = Image.open(img_out).convert("RGBA")
    image3 = Image.open(img_out).convert("RGBA")

    w, h = image1.size

    # alpha-blend the images
    
    image2 = image2.resize((int(w * scale), int(h * scale)))
    image3.paste(image2, (-int(w * (1 - 1/scale) / 2), -int(h * (1 - 1/scale) / 2)))
    image2 = image3

    alphaBlended = ImageChops.blend(image1, image2, alpha=alpha)
    rgb_img = alphaBlended.convert('RGB')
    rgb_img.save(img_blend)


for i in range(0, num_files):
    filename = files[i]
    print("Processing file ", i, " of ", num_files, "\n");

    n_iter = min(100, abs(2 * (i - 250)))
    if i < 100:
        n_iter = i
    

    img_out = out_dir + filename

    if os.path.exists(img_out):
        print("Output file already exists. Skipping.\n")
        continue

    
    # if i > 0, use blended image (created from previous iteration) as input
    if i > 0 and blend_frames:
        img_in = out_dir + "blended_" + filename
        if not os.path.exists(img_in):
            img_cur = in_dir + files[i]
            dream_last = out_dir + files[i - 1]
            blend(img_cur, dream_last, img_in, alpha_step, i, num_files)
    else:
        img_in = in_dir + filename

    print("dreaming" + img_in + " to " + img_out)
    

    dream(img_in, img_out, n_iter)
    
    # remove blended file
    if i > 0 and blend_frames:
        os.remove(img_in)

    # if i < len(files) then we blend the dreamed output with the raw input of the next frame
    if i + 1 < num_files and blend_frames:
        nextfile = files[i + 1]
        img_next = in_dir + nextfile
        img_blend = out_dir + "blended_" + nextfile
        blend(img_next, img_out, img_blend, alpha_step, i, num_files)

