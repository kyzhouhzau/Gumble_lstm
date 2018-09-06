#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:zhoukaiyin
"""
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
import numpy as np
def sample_gumble(shape,eps=1e-20):
    U = random_ops.random_uniform(shape)
    part = math_ops.log(U+eps)-math_ops.log(1+eps-U)
    return part

def gumble_softmax_sample(logits,temperature):
    y = logits+sample_gumble(logits.shape)
    # gmble_y = gen_nn_ops.softmax(y/temperature)
    gmble_y = math_ops.sigmoid(y/temperature)
    return gmble_y

def Gumble(x,name=None,temperature=0.2):
    with ops.name_scope(name, "Gumble", [x]):
        x = ops.convert_to_tensor(x, name="x")
        return gumble_softmax_sample(x,temperature)


def test():
    x = np.random.normal(size = [200,10]).astype(np.float32)
    w = tf.get_variable("zhou",[10,1000],initializer=tf.random_uniform_initializer())
    #logits = tf.nn.sigmoid(tf.matmul(x,w))
    logits = Gumble(tf.matmul(x,w))
    # loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y)
    # train_step = tf.train.AdamOptimizer(0.01).minimize(loss)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    # sess.run(train_step)
    y = sess.run(logits)
    result = []
    for i in y:
        for n in i:
            result.append(n)
    count_0=0
    count_1 = 0
    count_2 = 0
    for m in result:
        if m>=0 and m<=0.1:
            count_0+=1
        elif m>0.9 and m<=1:
            count_1+=1
        elif m>0.1 and m<0.9:
            count_2 += 1

    print("0-0.1之间的值有：",count_0/len(result))
    print("0.9-1之间的值有",count_1/len(result))
    print("0.1-0.9之间的值有",count_2/len(result))





if __name__=="__main__":
    test()
