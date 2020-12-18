from __future__ import print_function
import os
from PIL import Image, ImageChops
import subprocess
from io import BytesIO
import numpy as np
from functools import partial

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
out_dir = "images/zoom3/"
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

num_frames = 900

layer = 'mixed4d_3x3_bottleneck_pre_relu'

blend_frames = True
scale = 1.015
alpha_step = 1.0

layer = 'mixed5b_1x1'
n_iter = 1
step = 1.5
octaves = 8
octave_scale = 1.5
channel = 16


model_fn = 'models/tensorflow_inception_graph.pb'

# creating TensorFlow session and loading the model
graph = tf.Graph()
sess = tf.InteractiveSession(graph=graph)

with tf.gfile.FastGFile(model_fn, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
t_input = tf.placeholder(np.float32, name='input') # define the input tensor
imagenet_mean = 117.0
t_preprocessed = tf.expand_dims(t_input-imagenet_mean, 0)
tf.import_graph_def(graph_def, {'input':t_preprocessed})

layers = [op.name for op in graph.get_operations() if op.type=='Conv2D' and 'import/' in op.name]
feature_nums = [int(graph.get_tensor_by_name(name+':0').get_shape()[-1]) for name in layers]

def save_image(a, name):
    a = np.uint8(np.clip(a, 0, 1)*255)
    img = Image.fromarray(a)
    rgb_img = img.convert('RGB')
    rgb_img.save(name)
    
def savearray(a, path):
    a = np.uint8(np.clip(a, 0, 1)*255)
    with open(path, 'w') as outfile:
        Image.fromarray(a).save(outfile, fmt)
    
def visstd(a, s=0.1):
    '''Normalize the image range for visualization'''
    return (a-a.mean())/max(a.std(), 1e-4)*s + 0.5

def T(layer):
    '''Helper for getting layer output tensor'''
    return graph.get_tensor_by_name("import/%s:0"%layer)

def tffunc(*argtypes):
    '''Helper that transforms TF-graph generating function into a regular one.
    See "resize" function below.
    '''
    placeholders = list(map(tf.placeholder, argtypes))
    def wrap(f):
        out = f(*placeholders)
        def wrapper(*args, **kw):
            return out.eval(dict(zip(placeholders, args)), session=kw.get('session'))
        return wrapper
    return wrap

# Helper function that uses TF to resize an image
def resize(img, size):
    img = tf.expand_dims(img, 0)
    return tf.image.resize_bilinear(img, size)[0,:,:,:]
resize = tffunc(np.float32, np.int32)(resize)
  
# render_lapnorm(T(layer)[:,:,:,channel])
def calc_grad_tiled(img, t_grad, tile_size, i):
    '''Compute the value of tensor t_grad over the image in a tiled way.
    Random shifts are applied to the image to blur tile boundaries over 
    multiple iterations.'''
    sz = tile_size
    h, w = img.shape[:2]
    #sx, sy = np.random.randint(sz, size=2)
    sx = i % tile_size
    sy = sx

    print("sx=" + str(sx) + ", sy=" + str(sy))
    img_shift = np.roll(np.roll(img, sx, 1), sy, 0)
    grad = np.zeros_like(img)
    for y in range(0, max(h-sz//2, sz),sz):
        for x in range(0, max(w-sz//2, sz),sz):
            sub = img_shift[y:y+sz,x:x+sz]
            g = sess.run(t_grad, {t_input:sub})
            grad[y:y+sz,x:x+sz] = g
    return np.roll(np.roll(grad, -sx, 1), -sy, 0)

t_obj = T(layer)[:,:,:,channel]
t_score = tf.reduce_mean(t_obj) # defining the optimization objective
t_grad = tf.gradients(t_score, t_input)[0] # behold the power of automatic differentiation!

def render_deepdream(img_in, img_out, iter_n=1, step=1.5, octave_n=1, octave_scale=1.5):

    # split the image into a number of octaves
    img = Image.open(img_in)
    img = np.float32(img)
    octaves = []
    for i in range(octave_n-1):
        hw = img.shape[:2]
        lo = resize(img, np.int32(np.float32(hw)/octave_scale))
        hi = img-resize(lo, hw)
        img = lo
        octaves.append(hi)
    
    # generate details octave by octave
    for octave in range(0, octave_n):
        if octave>0:
            hi = octaves[-octave]
            img = resize(img, hi.shape[:2])+hi
        for i in range(iter_n):
            g = calc_grad_tiled(img, t_grad, 512, octave * iter_n + i)
            img += g*(step / (np.abs(g).mean()+1e-7))
    save_image(img/255.0, img_out)


#img_noise = np.random.uniform(size=(1920, 1080,3)) + 128.0
#im = Image.fromarray(np.uint8(img_noise))
#im.save('images/noise.jpg')

def dream(img_in, img_out, n_iter):
    render_deepdream(img_in, img_out, n_iter, step, octaves, octave_scale)

def blend(img_in, img_out, img_blend, alpha):
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


for i in range(1, num_frames):
    filename = str(i) + ".jpg"
    print("Processing file ", i, " of ", num_frames, "\n");

    img_out = out_dir + filename

    if os.path.exists(img_out):
        print("Output file already exists. Skipping.\n")
        continue
    
    # if i > 0, use blended image (created from previous iteration) as input
    if i > 1 and blend_frames:
        img_in = out_dir + "blended_" + str(i - 1) + ".jpg"
        if not os.path.exists(img_in):
            prev2_img = out_dir + str(i - 2) + ".jpg"
            prev_img = out_dir + str(i - 1) + ".jpg"
            blend(prev2_img, prev_img, img_in, alpha_step)
    else:
        img_in = out_dir + str(i - 1) + ".jpg"

    print("dreaming" + img_in + " to " + img_out)
    
    dream(img_in, img_out, n_iter)
    
    # remove blended file
    if i > 1 and blend_frames:
        os.remove(img_in)

    # if i < len(files) then we blend the dreamed output with the raw input of the next frame
    if i + 1 < num_frames and blend_frames:
        prev_img = out_dir + str(i - 1) + ".jpg"
        img_blend = out_dir + "blended_" + str(i) + ".jpg"
        blend(prev_img, img_out, img_blend, alpha_step)

