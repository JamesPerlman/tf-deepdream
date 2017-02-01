from __future__ import print_function
import os
import subprocess
from io import BytesIO
import numpy as np
from functools import partial
import PIL.Image
from IPython.display import clear_output, Image, display, HTML

import tensorflow as tf

img_noise = np.random.uniform(size=(224,224,3)) + 100.0

def load_dependencies():
    if not os.path.exists('models/tensorflow_inception_graph.pb'):
        subprocess.call(['wget', '-O', 'models/inception5h.zip',
                         'https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip'])
        subprocess.call(['unzip', 'inception5h.zip'], cwd='models')


def strip_consts(graph_def, max_const_size=32):
    """Strip large constant values from graph_def."""
    strip_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add()
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = bytes("<stripped %d bytes>" % size, 'utf-8')
    return strip_def



def rename_nodes(graph_def, rename_func):
    res_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = res_def.node.add()
        n.MergeFrom(n0)
        n.name = rename_func(n.name)
        for i, s in enumerate(n.input):
            n.input[i] = rename_func(s) if s[0] != '^' else '^' + rename_func(s[1:])
    return res_def


def showarray(a, fmt='jpeg'):
    a = np.uint8(np.clip(a, 0, 1) * 255)
    f = BytesIO()
    PIL.Image.fromarray(a).save(f, fmt)
    display(Image(data=f.getvalue()))


def savearray(a, path, fmt='jpeg'):
    a = np.uint8(np.clip(a, 0, 1) * 255)
    with open(path, 'w') as outfile:
        PIL.Image.fromarray(a).save(outfile, fmt)


def visstd(a, s=0.1):
    '''Normalize the image range for visualization'''
    return (a - a.mean()) / max(a.std(), 1e-4) * s + 0.5


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


def _resize(img, size):
    img = tf.expand_dims(img, 0)
    return tf.image.resize_bilinear(img, size)[0, :, :, :]

def show_pic():
    img0 = PIL.Image.open('images/pilatus800.jpg')
    img0 = np.float32(img0)
    showarray(img0 / 255.0)

class Deepdream:
    def __init__(self, model_fn='models/tensorflow_inception_graph.pb'):
        # creating TensorFlow session and loading the model
        self.graph = tf.Graph()
        self.sess = tf.InteractiveSession(graph=self.graph)
        with tf.gfile.FastGFile(model_fn, 'rb') as f:
            self.graph_def = tf.GraphDef()
            self.graph_def.ParseFromString(f.read())
        self.t_input = tf.placeholder(np.float32, name='input')  # define the input tensor
        imagenet_mean = 117.0
        t_preprocessed = tf.expand_dims(self.t_input - imagenet_mean, 0)
        tf.import_graph_def(self.graph_def, {'input': t_preprocessed})

        self.k = np.float32([1, 4, 6, 4, 1])
        self.k = np.outer(self.k, self.k)
        self.k5x5 = self.k[:, :, None, None] / self.k.sum() * np.eye(3, dtype=np.float32)
        self.img0 = PIL.Image.open('images/pilatus800.jpg')
        self.img0 = np.float32(self.img0)

        self.resize = tffunc(np.float32, np.int32)(_resize)

    def show_lap_graph(self):
        lap_graph = tf.Graph()
        with lap_graph.as_default():
            lap_in = tf.placeholder(np.float32, name='lap_in')
            lap_out = self.lap_normalize(lap_in)
        self.show_graph(lap_graph)

    def calc_grad_tiled(self, img, t_grad, tile_size=512):
        '''Compute the value of tensor t_grad over the image in a tiled way.
        Random shifts are applied to the image to blur tile boundaries over
        multiple iterations.'''
        sz = tile_size
        h, w = img.shape[:2]
        sx, sy = np.random.randint(sz, size=2)
        img_shift = np.roll(np.roll(img, sx, 1), sy, 0)
        grad = np.zeros_like(img)
        for y in range(0, max(h - sz // 2, sz), sz):
            for x in range(0, max(w - sz // 2, sz), sz):
                sub = img_shift[y:y + sz, x:x + sz]
                g = self.sess.run(t_grad, {self.t_input: sub})
                grad[y:y + sz, x:x + sz] = g
        return np.roll(np.roll(grad, -sx, 1), -sy, 0)

    def render_naive(self, t_obj, img0=img_noise, iter_n=20, step=1.0):
        t_score = tf.reduce_mean(t_obj)  # defining the optimization objective
        t_grad = tf.gradients(t_score, self.t_input)[0]  # behold the power of automatic differentiation!

        img = img0.copy()
        for i in range(iter_n):
            g, score = self.sess.run([t_grad, t_score], {self.t_input: img})
            # normalizing the gradient, so the same step size should work
            g /= g.std() + 1e-8  # for different layers and networks
            img += g * step
            print(score, end=' ')
        clear_output()
        showarray(visstd(img))

    def T(self, layer):
        '''Helper for getting layer output tensor'''
        return self.graph.get_tensor_by_name("import/%s:0" % layer)

    def show_graph(self, graph_def, max_const_size=32):
        """Visualize TensorFlow graph."""
        if hasattr(self.graph_def, 'as_graph_def'):
            self.graph_def = self.graph_def.as_graph_def()
        strip_def = strip_consts(self.graph_def, max_const_size=max_const_size)
        code = """
            <script>
              function load() {{
                document.getElementById("{id}").pbtxt = {data};
              }}
            </script>
            <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
            <div style="height:600px">
              <tf-graph-basic id="{id}"></tf-graph-basic>
            </div>
        """.format(data=repr(str(strip_def)), id='graph' + str(np.random.rand()))

        iframe = """
            <iframe seamless style="width:1024px;height:768px;border:0" srcdoc="{}"></iframe>
        """.format(code.replace('"', '&quot;'))
        display(HTML(iframe))

    def render_multiscale(self, t_obj, img0=img_noise, iter_n=10, step=1.0, octave_n=3, octave_scale=1.4):
        t_score = tf.reduce_mean(t_obj)  # defining the optimization objective
        t_grad = tf.gradients(t_score, self.t_input)[0]  # behold the power of automatic differentiation!

        img = img0.copy()
        for octave in range(octave_n):
            if octave > 0:
                hw = np.float32(img.shape[:2]) * octave_scale
                img = self.resize(img, np.int32(hw))
            for i in range(iter_n):
                g = self.calc_grad_tiled(img, t_grad)
                # normalizing the gradient, so the same step size should work
                g /= g.std() + 1e-8  # for different layers and networks
                img += g * step
                print('.', end=' ')
            clear_output()
            showarray(visstd(img))

    def print_stats(self):
        layers = [op.name for op in self.graph.get_operations() if op.type == 'Conv2D' and 'import/' in op.name]
        feature_nums = [int(self.graph.get_tensor_by_name(name + ':0').get_shape()[-1]) for name in layers]

        print('Number of layers', len(layers))
        print('Total number of feature channels:', sum(feature_nums))

    def lap_split(self,img):
        '''Split the image into lo and hi frequency components'''
        with tf.name_scope('split'):
            lo = tf.nn.conv2d(img, self.k5x5, [1, 2, 2, 1], 'SAME')
            lo2 = tf.nn.conv2d_transpose(lo, self.k5x5 * 4, tf.shape(img), [1, 2, 2, 1])
            hi = img - lo2
        return lo, hi

    def lap_split_n(self,img, n):
        '''Build Laplacian pyramid with n splits'''
        levels = []
        for i in range(n):
            img, hi = self.lap_split(img)
            levels.append(hi)
        levels.append(img)
        return levels[::-1]

    def lap_merge(self,levels):
        '''Merge Laplacian pyramid'''
        img = levels[0]
        for hi in levels[1:]:
            with tf.name_scope('merge'):
                img = tf.nn.conv2d_transpose(img, self.k5x5 * 4, tf.shape(hi), [1, 2, 2, 1]) + hi
        return img

    def normalize_std(self,img, eps=1e-10):
        '''Normalize image by making its standard deviation = 1.0'''
        with tf.name_scope('normalize'):
            std = tf.sqrt(tf.reduce_mean(tf.square(img)))
            return img / tf.maximum(std, eps)

    def lap_normalize(self,img, scale_n=4):
        '''Perform the Laplacian pyramid normalization.'''
        img = tf.expand_dims(img, 0)
        tlevels = self.lap_split_n(img, scale_n)
        tlevels = list(map(self.normalize_std, tlevels))
        out = self.lap_merge(tlevels)
        return out[0, :, :, :]

    def get_lapnorm(self, t_obj, img0=img_noise, visfunc=visstd,
                    iter_n=10, step=1.0, octave_n=3, octave_scale=1.4, lap_n=4):
        t_score = tf.reduce_mean(t_obj)  # defining the optimization objective
        t_grad = tf.gradients(t_score, self.t_input)[0]  # behold the power of automatic differentiation!
        # build the laplacian normalization graph
        lap_norm_func = tffunc(np.float32)(partial(self.lap_normalize, scale_n=lap_n))

        img = img0.copy()
        for octave in range(octave_n):
            if octave > 0:
                hw = np.float32(img.shape[:2]) * octave_scale
                img = self.resize(img, np.int32(hw))
            for i in range(iter_n):
                g = self.calc_grad_tiled(img, t_grad)
                g = lap_norm_func(g)
                img += g * step
                print('.', end=' ')
            clear_output()
            showarray(visfunc(img))

        return visfunc(img)

    def render_lapnorm(self, *args, **kwargs):
        self.get_lapnorm(*args, **kwargs)

    def render_deepdream(self, t_obj, img0=img_noise,
                         iter_n=10, step=1.5, octave_n=4, octave_scale=1.4):
        t_score = tf.reduce_mean(t_obj)  # defining the optimization objective
        t_grad = tf.gradients(t_score, self.t_input)[0]  # behold the power of automatic differentiation!

        # split the image into a number of octaves
        img = img0
        octaves = []
        for i in range(octave_n - 1):
            hw = img.shape[:2]
            lo = self.resize(img, np.int32(np.float32(hw) / octave_scale))
            hi = img - self.resize(lo, hw)
            img = lo
            octaves.append(hi)

        # generate details octave by octave
        for octave in range(octave_n):
            if octave > 0:
                hi = octaves[-octave]
                img = self.resize(img, hi.shape[:2]) + hi
            for i in range(iter_n):
                g = self.calc_grad_tiled(img, t_grad)
                img += g * (step / (np.abs(g).mean() + 1e-7))
                print('.', end=' ')
            clear_output()
            showarray(img / 255.0)


# def main():
#     load_dependencies()
#     net = Deepdream()
#     T = net.T
#
#     layer = 'mixed4d_3x3_bottleneck_pre_relu'
#     channel = 139
#
#     net.render_naive(T(layer)[:, :, :, channel])
#     net.render_multiscale(T(layer)[:, :, :, channel])
#
#
# if __name__== '__main__':
#     main()

