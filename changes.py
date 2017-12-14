from __future__ import print_function

import time
from PIL import Image
import numpy as np

from keras import backend
from keras.models import Model
from keras.applications.vgg16 import VGG16

from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave


height=1024
width=1024

img_path="C:/Users/NISARG/Pictures/nis.jpg"
content_img=Image.open(img_path)
content_img=content_img.resize((height,width))
#content_img.show()

style_image_path = 'C:/Users/NISARG/Desktop/WinPython/WinPython-64bit-3.5.3.1Qt5/How-to-Generate-Art-Demo-master/images/styles/starry_night.jpg'
style_image = Image.open(style_image_path)
style_image = style_image.resize((height, width))
#style_image.show()

new_image=np.asarray(content_img,dtype="float32")
new_image=np.expand_dims(new_image,axis=0)
print(new_image.shape)

new_style=np.asarray(style_image,dtype="float32")
new_style=np.expand_dims(new_style,axis=0)
print(new_style.shape)

new_image[:, :, :, 0] -= 103.939
new_image[:, :, :, 1] -= 116.779
new_image[:, :, :, 2] -= 123.68
new_image =new_image[:, :, :, ::-1]

new_style[:, :, :, 0] -= 103.939
new_style[:, :, :, 1] -= 116.779
new_style[:, :, :, 2] -= 123.68
new_style = new_style[:, :, :, ::-1]


content=backend.variable(new_image)
style=backend.variable(new_style)
combination_image = backend.placeholder((1, height, width, 3))

input_tensor = backend.concatenate([content,
                                    style,
                                    combination_image], axis=0)

model = VGG16(input_tensor=input_tensor, weights='imagenet',
              include_top=False)

layers = dict([(layer.name, layer.output) for layer in model.layers])
print(layers)

content_weight = 0.04
style_weight = 7.0
total_variation_weight = 2.0
loss = backend.variable(0.)

def content_loss(content, combination):
    return backend.sum(backend.square(combination - content))

layer_features = layers['block2_conv2']
content_image_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]

loss += content_weight * content_loss(content_image_features,
                                      combination_features)

def gram_matrix(x):
    features = backend.batch_flatten(backend.permute_dimensions(x, (2, 0, 1)))
    gram = backend.dot(features, backend.transpose(features))
    return gram

def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = height * width
    return backend.sum(backend.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

feature_layers = ['block1_conv2', 'block2_conv2',
                  'block3_conv3', 'block4_conv3',
                  'block5_conv3']

for layer_name in feature_layers:
    layer_features = layers[layer_name]
    style_features = layer_features[1, :, :, :]
    combination_features = layer_features[2, :, :, :]
    sl = style_loss(style_features, combination_features)
    loss += (style_weight / len(feature_layers)) * sl
    
def total_variation_loss(x):
    a = backend.square(x[:, :height-1, :width-1, :] - x[:, 1:, :width-1, :])
    b = backend.square(x[:, :height-1, :width-1, :] - x[:, :height-1, 1:, :])
    return backend.sum(backend.pow(a + b, 1.25))

grads = backend.gradients(loss, combination_image)

loss += total_variation_weight * total_variation_loss(combination_image) 

outputs = [loss]
outputs += grads
f_outputs = backend.function([combination_image], outputs)

def eval_loss_and_grads(x):
    x = x.reshape((1, height, width, 3))
    outs = f_outputs([x])
    loss_value = outs[0]
    grad_values = outs[1].flatten().astype('float64')
    return loss_value, grad_values

class Evaluator(object):

    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

evaluator = Evaluator()

x = np.random.uniform(0, 255, (1, height, width, 3)) - 128.

iterations = 10

for i in range(iterations):
    print('Start of iteration', i)
    start_time = time.time()
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                     fprime=evaluator.grads, maxfun=20)
    print('Current loss value:', min_val)
    end_time = time.time()
    print('Iteration %d completed in %ds' % (i, end_time - start_time))
    
x = x.reshape((height, width, 3))
x = x[:, :, ::-1]
x[:, :, 0] += 103.939
x[:, :, 1] += 116.779
x[:, :, 2] += 123.68
x = np.clip(x, 0, 255).astype('uint8')

Image.fromarray(x)   
