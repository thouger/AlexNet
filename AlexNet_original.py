# -*- coding: utf-8 -*-
# @Time    : 2018/11/20 18:31
# @Author  : thouger
# @Email   : 1030490158@qq.com
# @File    : AlexNet_original.py
# @Software: PyCharm

import tensorflow as tf
import numpy as np
from scipy.misc import imread
from caffe_classes import class_names

# todo 留下疑问1：图像有四层，为什么只取了前面三层
# todo 留下疑问2：为什么图片的第一层和第三层要互换
# todo 留下疑问3：为什么不符合尺寸的图像就不能用这个卷积神经网络，为什么输入的图像维度都是要固定的

weights = {
    'wc1': (11, 11, 1, 96),
    'wc2': (5, 5, 96, 256),
    'wc3': (3, 3, 256, 384),
    'wc4': (3, 3, 192, 384),
    'wc5': (3, 3, 192, 256),
    'wc6': (256 * 6 * 6, 4096),
    'wc7': (4096, 4096),
    'wc8': (4096, 10)
}
biases = {
    'bc1': 96,
    'bc2': 256,
    'bc3': 384,
    'bc4': 384,
    'bc5': 256,
    'bc6': 4096,
    'bc7': 4096,
    'bc8': 10
}

# AlexNet网络成功之处：
# 1.使用ReLU作为CNN的激活函数，比Sigmoid效果好
# 2.使用DroPout避免过拟合
# 3.使用最大池化，并且池的步长比池化核尺寸要小，最大池化相对比平均池化，避免了模糊性，步长小于池核，使得输出有重叠和覆盖
# 4.使用LRN层，使神经元响应中，较大的值相对较大，同时抑制其他较小的值，增强泛化能力
x = tf.placeholder(tf.float32, (None, 227, 227, 3))
net_data = np.load(open("../input/bvlc_alexnet.npy", "rb"), encoding="latin1").item()

im1 = (imread("laska.png")[:, :, :3]).astype(np.float32)
im1 = im1 - np.mean(im1)
im1[:, :, 0], im1[:, :, 2] = im1[:, :, 2], im1[:, :, 0]

im2 = (imread("poodle.png")[:, :, :3]).astype(np.float32)
im2[:, :, 0], im2[:, :, 2] = im2[:, :, 2], im2[:, :, 0]


def conv(input, kernel, biases, strides_weight, strides_height, group=1):
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, strides_weight, strides_height, 1], padding="SAME")

    if group == 1:
        conv = convolve(input, kernel)
    else:
        input_groups = tf.split(input, group, 3)
        kernel_groups = tf.split(kernel, group, 3)
        output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
        conv = tf.concat(output_groups, 3)
    return tf.reshape(tf.nn.bias_add(conv, biases), [-1] + conv.get_shape().as_list()[1:])


def max_pool(x):
    return tf.nn.max_pool(x, [1, 3, 3, 1], [1, 2, 2, 1], 'VALID')


def alex_net(x):
    conv1_w = tf.Variable(net_data['conv1'][0])
    conv1_b = tf.Variable(net_data['conv1'][1])
    conv1 = conv(x, conv1_w, conv1_b, 4, 4, 1)
    conv1 = tf.nn.relu(conv1)
    lrn1 = tf.nn.local_response_normalization(conv1, 2, 1, 2e-05, 0.75)
    pool1 = max_pool(lrn1)

    conv2_w = tf.Variable(net_data['conv2'][0])
    conv2_b = tf.Variable(net_data['conv2'][1])
    conv2 = conv(pool1, conv2_w, conv2_b, 1, 1, 2)
    conv2 = tf.nn.relu(conv2)
    lrn2 = tf.nn.local_response_normalization(conv2, 2, 1, 2e-05, 0.75)
    pool2 = max_pool(lrn2)

    conv3_w = tf.Variable(net_data['conv3'][0])
    conv3_b = tf.Variable(net_data['conv3'][1])
    conv3 = conv(pool2, conv3_w, conv3_b, 1, 1, 1)
    conv3 = tf.nn.relu(conv3)

    conv4_w = tf.Variable(net_data['conv4'][0])
    conv4_b = tf.Variable(net_data['conv4'][1])
    conv4 = conv(conv3, conv4_w, conv4_b, 1, 1, 2)
    conv4 = tf.nn.relu(conv4)

    conv5_w = tf.Variable(net_data['conv5'][0])
    conv5_b = tf.Variable(net_data['conv5'][1])
    conv5 = conv(conv4, conv5_w, conv5_b, 1, 1, 2)
    conv5 = tf.nn.relu(conv5)
    pool5 = max_pool(conv5)

    conv6_w = tf.Variable(net_data['fc6'][0])
    conv6_b = tf.Variable(net_data['fc6'][1])
    fc6 = tf.nn.relu_layer(tf.reshape(pool5, [-1, int(np.prod(pool5.get_shape()[1:]))]), conv6_w, conv6_b)

    conv7_w = tf.Variable(net_data['fc7'][0])
    conv7_b = tf.Variable(net_data['fc7'][1])
    fc7 = tf.nn.relu_layer(fc6, conv7_w, conv7_b)

    conv8_w = tf.Variable(net_data['fc8'][0])
    conv8_b = tf.Variable(net_data['fc8'][1])
    fc8 = tf.nn.xw_plus_b(fc7, conv8_w, conv8_b)
    prob = tf.nn.softmax(fc8)
    return prob


prob = alex_net(x)
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    output = sess.run(prob, feed_dict={x: [im1, im2]})

    for input_im_ind in range(output.shape[0]):
        inds = np.argsort(output)[input_im_ind, :]
        print("Image", input_im_ind)
        for i in range(5):
            result = inds[-1 - i]
            print(class_names[result], output[input_im_ind, result])
