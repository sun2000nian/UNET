import tensorflow as tf
from model import unet
from data import getData
import numpy as np
import imageio


def lr_callback(epoch, lr):
    if epoch == 2:
        return lr*0.1


imgs, labels = getData()
model = unet(input_size=(256, 256, 1))

model.fit(
    x=imgs,
    y=labels,
    epochs=10,
    verbose=1)
