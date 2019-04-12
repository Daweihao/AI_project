from keras.utils import np_utils
from keras import optimizers
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
import numpy as np
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
import matplotlib.pyplot as plt
import os
from keras_vggface.vggface import VGGFace
from keras.engine import Model
import keras.metrics as metrics

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def acc_top3(y_true, y_pred):
    return metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)

img_list = []
y_list = []
ref = {}
i = 0
for file in os.listdir("Cropped/"):
    yi = int(file.split('_')[0]) - 1
    img = preprocess_image("Cropped/"+ file)[0,:]
    img_list.append(img)
    ref[i] = file.split('_')[0]
    y_list.append(yi)
    i+=1
X = np.array(img_list)
y = np.array(y_list)

num_classes = 190
hidden_dim = 1024

vgg_model = VGGFace(include_top=False, input_shape=(224, 224, 3))
for layer in vgg_model.layers:
    layer.trainable = False
last_layer = vgg_model.get_layer('pool5').output
x = Flatten(name='flatten')(last_layer)
x = Dense(hidden_dim, activation='relu', name='fc6')(x)
x = Dense(hidden_dim, activation='relu', name='fc7')(x)
out = Dense(num_classes, activation='softmax', name='fc8')(x)
custom_vgg_model = Model(vgg_model.input, out)

# adam = optimizers.Adam(lr=1e-3 )
sgd = optimizers.SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
custom_vgg_model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy',acc_top3])

print(custom_vgg_model.summary())
y = np_utils.to_categorical(y, num_classes)
history = custom_vgg_model.fit(X,y,shuffle=True,epochs=10)
custom_vgg_model.save_weights('ear_vgg16.h5')

plt.switch_backend('agg')
plt.plot(history.history['acc'])
plt.plot(history.history['acc_top3'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['acc', 'acc_top3'])
plt.savefig("acc_vgg.png")

plt.switch_backend('agg')
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss'])
plt.savefig("loss_vgg.png")
