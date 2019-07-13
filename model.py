import os
import random
import numpy as np
import matplotlib.pyplot as plt
from keras import optimizers
from keras.backend import concatenate
from keras import applications
from keras.layers import Input, BatchNormalization, Lambda, Add, merge
from keras import initializers, optimizers
from keras.models import Model, Sequential
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.core import Reshape, Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D, MaxPooling2D, ZeroPadding2D, Cropping2D
from keras.models import load_model
from keras.layers.merge import concatenate, add
from keras.optimizers import adam
import scipy
import imageio

os.chdir('working/directory/')

def read_bin(name, count, a):
    arr = np.fromfile(name, dtype=np.uint8)
    arr = np.reshape(arr, (count, a, a, 3))
    return arr

train_x = read_bin('train_x', 1000, 512)
train_y = read_bin('train_y', 1000, 512)
test_x = read_bin('test_x', 20, 512)
test_y = read_bin('test_y', 20, 512)

print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)

def normalize(data):
    data = (data.astype(np.float32)-127.5)/127.5
    return data

train_x = normalize(train_x)
train_y = normalize(train_y)
test_x = normalize(test_x)
test_y = normalize(test_y)

def Res_block():
    _input = Input(shape=(None, None, 64))

    conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(_input)
    conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv)

    out = add(inputs=[_input, conv])
    out = Activation('relu')(out)

    model = Model(inputs=_input, outputs=out)

    return model

def build_model():
    _input = Input(shape=(None, None, 3), name='input')

    Feature = Conv2D(filters=64, kernel_size=(9, 9), strides=(1, 1), padding='same', activation='relu')(_input)
    Feature_out = Res_block()(Feature)

    Reslayer1 = Res_block()(Feature_out)

    Reslayer2 = Res_block()(Reslayer1)

    # ***************//
    Multi_scale1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), 
                          padding='same', activation='relu')(Reslayer2)

    Multi_scale2a = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1),
                           padding='same', activation='relu')(Multi_scale1)

    Multi_scale2b = Conv2D(filters=64, kernel_size=(1, 3), strides=(1, 1),
                           padding='same', activation='relu')(Multi_scale1)
    Multi_scale2b = Conv2D(filters=64, kernel_size=(3, 1), strides=(1, 1),
                           padding='same', activation='relu')(Multi_scale2b)

    Multi_scale2c = Conv2D(filters=64, kernel_size=(1, 5), strides=(1, 1),
                           padding='same', activation='relu')(Multi_scale1)
    Multi_scale2c = Conv2D(filters=64, kernel_size=(5, 1), strides=(1, 1),
                           padding='same', activation='relu')(Multi_scale2c)

    Multi_scale2d = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1),
                           padding='same', activation='relu')(Multi_scale1)
    Multi_scale2d = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1),
                           padding='same', activation='relu')(Multi_scale2d)

    Multi_scale2 = concatenate(inputs=[Multi_scale2a, Multi_scale2b, Multi_scale2c, Multi_scale2d])

    out = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='tanh')(Multi_scale2)
    model = Model(input=_input, output=out)

    return model

model = build_model()

model.compile(optimizer=optimizers.Adam(lr=0.0003), loss='logcosh', metrics=['accuracy'])
history = model.fit(train_x, train_y, validation_split=0.2, batch_size=3, nb_epoch=50)
model.save("model.h5")

scores = model.evaluate(test_x, test_y)
print('%s: %.2f%%'% (model.metrics_names[1], scores[1]*100))

def denormalize(data):
    data = (data.astype(np.float32) * 127.5) + 127.5
    return data

i = np.random.randint(0, len(train_x))
print(i)
temp = []
temp.append(train_x[i])
img = np.array(temp)
prediction = model.predict(img)
prediction = np.squeeze(prediction, axis=0)
f, a = plt.subplots(1, 3)
a[0].set_title('Original')
a[0].imshow(train_y[i])
a[1].set_title('Input')
a[1].imshow(train_x[i])
a[2].set_title('Predicted')
a[2].imshow(prediction)
plt.show()

imageio.imwrite(str(i)+'predicted.png', prediction)



