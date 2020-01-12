# -*- coding: utf-8 -*-
import vgg16
import tensorflow as tf
import utils
import cv2
import numpy as np
from scipy.linalg import norm
import os
import time

images = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3])
vgg = vgg16.Vgg16()
vgg.build(images)


def get_feats_dirs(imagefold):
    vgg16_feats = np.zeros((27, 4096)) #这可能是生成一个矩阵，27是我这个文件夹里图片的各数，4096是特征码的长度
    for root, dirs, files in os.walk(imagefold):

        with tf.Session() as sess:
            start00 = time.clock()
            for i in range(len(files)):
                imagePath = root + '\\' + files[i] # 拼凑图片路径
                start1 = time.clock()
                print("开始对 %d 生成特征: %s\n" % (i, imagePath))
                img_list = utils.load_image(imagePath) # 加载图片
                start2 = time.clock()
                print("载入图片花费:%ds" % (time.clock() - start1))
                batch = img_list.reshape((1, 224, 224, 3)) # 可以叫做对图片格式化吧，做成vgg16的标准格式
                start4 = time.clock()
                feature = sess.run(vgg.fc6, feed_dict={images: batch})  # 喂入图片，提取fc6层的特征
                start3 = time.clock()
                feature = np.reshape(feature, [4096])
                feature /= norm(feature)  # 特征归一化
                feature = np.array(feature)
                vgg16_feats[i, :] = feature  # 每张图片的特征向量为1行
                dict1.update({i: imagePath})
                print("完成生成特征，花费:%ds\n" % (time.clock() - start1))
    print("完成所有，总花费:%ds/%d\n" % (time.clock() - start00, len(dict1)))

    vgg16_feats = np.save(r'd:\\config\\feats', vgg16_feats) # 保存所有特征
    return vgg16_feats


if __name__ == '__main__':
    get_feats_dirs("D:\\pic")