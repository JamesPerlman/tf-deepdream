from __future__ import print_function
import os
import subprocess
from io import BytesIO
import numpy as np
import cupy as cp
from functools import partial
import PIL.Image

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def load_graph_def():
    model_fn = 'models/tensorflow_inception_graph.pb'

    with tf.gfile.FastGFile(model_fn, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

        return graph_def


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

def resize(img, size):
    img = tf.expand_dims(img, 0)
    return tf.image.resize_bilinear(img, size)[0,:,:,:]

graph_def = load_graph_def()

class DeepDream:
    # creating TensorFlow session and loading the model
    def __init__(self, tensor_name, get_t_obj):
        with tf.Graph().as_default() as graph:
            sess = tf.InteractiveSession(graph=graph)

            # define the input tensor
            t_input = tf.placeholder(cp.float32, name='input')
            imagenet_mean = 117.0
            t_preprocessed = tf.expand_dims(t_input - imagenet_mean, 0)
            tf.import_graph_def(graph_def, { 'input' : t_preprocessed })

            tensor = graph.get_tensor_by_name("import/%s:0"%tensor_name)
            t_obj = get_t_obj(tensor)
            t_score = tf.reduce_mean(t_obj) # defining the optimization objective

            self.t_grad = tf.gradients(t_score, t_input)[0] # behold the power of automatic differentiation!
            self.resize = tffunc(np.float32, np.int32)(resize)
            self.t_input = t_input
            self.session = sess
    
    def cleanup(self):
        self.session.close()
        tf.reset_default_graph()

    # deepdream funcs
    def save_image(self, a, name):
        a = cp.uint8(cp.clip(a, 0, 1) * 255)
        img = PIL.Image.fromarray(cp.asnumpy(a))
        rgb_img = img.convert('RGB')
        rgb_img.save(name)

    def calc_grad_tiled(self, img, tile_size, i):
        '''Compute the value of tensor t_grad over the image in a tiled way.
        Random shifts are applied to the image to blur tile boundaries over 
        multiple iterations.'''
        sz = tile_size
        h, w = img.shape[:2]
        #sx, sy = np.random.randint(sz, size=2)
        sx = i % tile_size
        sy = sx

        img_shift = cp.roll(cp.roll(img, sx, 1), sy, 0)
        grad = cp.zeros_like(img)

        sess = tf.get_default_session()
        
        for y in range(0, max(h - sz//2, sz), sz):
            for x in range(0, max(w - sz//2, sz), sz):
                sub = img_shift[y: y + sz, x: x + sz]
                g = sess.run(self.t_grad, { self.t_input: cp.asnumpy(sub) })
                grad[y:y+sz,x:x+sz] = cp.asarray(g)
        return cp.roll(cp.roll(grad, -sx, 1), -sy, 0)

    def render(self, img_in, img_out, iter_n=1, step=1.5, octave_n=1, octave_scale=1.5):
        # split the image into a number of octaves
        img = PIL.Image.open(img_in)
        img = cp.asarray(np.float32(img))
        octaves = []
        for i in range(octave_n - 1):
            (*hw, c) = img.shape
            lo = cp.resize(img, (*cp.int32(cp.float32(hw) / octave_scale), c))
            hi = img - cp.resize(lo, (*hw, c))
            img = lo
            octaves.append(hi)
        
        # generate details octave by octave
        for octave in range(0, octave_n):
            if octave>0:
                hi = octaves[-octave]
                img = cp.resize(img, hi.shape) + hi
            for i in range(iter_n):
                g = self.calc_grad_tiled(img, 512, octave * iter_n + i)
                img += g * (step / (cp.abs(g).mean() + 1e-7))
        self.save_image(img / 255.0, img_out)
