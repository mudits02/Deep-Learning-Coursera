import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, Dense, BatchNormalization, MaxPool2D, GlobalAveragePooling2D


model = tf.keras.Sequential(
    [
        Input(shape = (28,28,1)),## Conv2D reuqires a 4 dimensional tensor as input
        Conv2D(32 , (3,3) , activation = 'relu'),
        Conv2D(64 , (3,3) , activation = 'relu'),
        MaxPool2D(),
        BatchNormalization(),

        Conv2D(128 , (3 ,3) , activation = 'relu'),
        MaxPool2D(),
        BatchNormalization(),

        Dense(64 , activation = 'relu'),
        Dense(10 , activation = 'Softmax')
    ]
)


def image_display(examples , labels):

    plt.figure(figsize = (10 , 10))

    for i in range(25):

        index = np.random.randint(0 , examples.shape[0] - 1)
        label = labels[index]
        img = examples[index]
        plt.subplot(5, 5, i + 1)
        plt.title(str(label))
        plt.tight_layout()
        plt.imshow(img , cmap = 'gray')


    plt.show()

    




if __name__ == '__main__':
    (x_train , y_train) , (x_test , y_test) = tf.keras.datasets.mnist.load_data()

    ##print("Size of training x = " , x_train.shape)
    #print("Size of training y = " , y_train.shape)
    #print("Size of test x = " , x_test.shape)
    #print("Size of test y = " , y_test.shape)


    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    x_train = np.expand_dims(x_train , axis = -1)
    x_test = np.expand_dims(x_test , axis = -1)

    model.compile(optimizer = 'adam' , loss = 'categorial_crossentropy' , metrics = 'accuracy')
    model.fit(x_train , y_train , batch_size = 64 , epochs = 10 , validation_split = 0.2)

    model.evaluate(x_test , y_test , batch_size = 64)
    ##image_display(x_train , y_train)


    
