import os
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from PIL import Image
import skimage.io as io
import scipy

def normalize(data):
    data = (data.astype(np.float32)-127.5)/127.5
    return data

# If size of the output image has to be increased then use this function
def resize_(img, factor, name, format_):    
    # width, height = img.size
    # img = img.resize((int(width/factor), int(height/factor)), Image.BICUBIC)    
    # img.save(name+'_bicubic'+format_)
    # img = Image.open(name+'_bicubic'+format_)
    width, height = img.size
    img = img.resize((width*factor, height*factor), Image.BICUBIC)
    img.save(name+'_bicubic'+format_)
    return img

def prediction():
    import scipy
    temp = []
    name = '1'
    format_ = '.png'
    factor = 4
    img = Image.open(name+format_)
    width, height = img.size
    # If size of the output image has to be of same isze as input image then comment the resize_() function
    # img = resize_(img, factor, name, format_)
    print(img.size)
    img_a = np.array(img)
    temp.append(img_a)
    img_g = np.array(temp)
    img_g = normalize(img_g)
    model1 = load_model('model.h5')
    prediction1 = model1.predict(img_g)
    prediction1 = np.squeeze(prediction1, axis=0)
    prediction1 = prediction1 * 127.5 + 127.5
    f, a = plt.subplots(1, 2)
    a[0].imshow(img)
    a[1].imshow(prediction1.astype(np.uint8))
    scipy.misc.imsave(name+'predicted'+format_, prediction1)
    plt.show()

prediction()


