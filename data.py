from cProfile import label
import numpy as np
import imageio
import tensorflow as tf

image_path = "C:/Users/sun20/Documents/jupyter/My_UNET/image/"
label_path = "C:/Users/sun20/Documents/jupyter/My_UNET/label/"


def getData():
    imgs = []
    labels = []
    for i in range(0, 30):
        img = imageio.imread(image_path+str(i)+".png")
        label = imageio.imread(label_path+str(i)+".png")
        #img = np.resize(img, (256, 256))
        #label = np.resize(label, (256, 256))
        img = np.expand_dims(img, -1)
        label = np.expand_dims(label, -1)
        imgs.append(img)
        labels.append(label)
    imgs = np.array(imgs)
    labels = np.array(labels)
    imgs = tf.image.resize(imgs, (256, 256))
    labels = tf.image.resize(labels, (256, 256))
    return imgs, labels

if __name__ == '__main__':
    imgs, labels = getData()
