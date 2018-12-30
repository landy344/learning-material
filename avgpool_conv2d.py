import numpy as np
import tensorflow as tf

def build_net():
    x = np.ones((1,4,4,1), dtype=np.float32)
    x = np.pad(x, ((0,0),(1,1),(1,1),(0,0)), 'constant')
    x.tofile('/home/sheldon/work/code/test/x.txt', sep='\n', format='%f')

    y = tf.nn.avg_pool(value=x, ksize=[1,3,3,1], strides=[1,1,1,1], padding='VALID')

    with tf.Session() as sess:
        #init_op = tf.global_variables_initializer()
        sess.run(y)
        print(y)

if __name__ == "__main__":
    build_net()
    print("build net end")