from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

def get_CNN_model(train,test,num_classes,img_size):
    # Model / data parameters
    input_shape = (img_size,img_size,1) # model requires an image shape = 28*28
    
    # Scale images to [0,1] and reshape asrequired by input shape
    x_train = (train[0].astype('float32')/255).reshape(-1,img_size,img_size)
    x_test = (test[0].astype('float32')/255).reshape(-1,img_size,img_size)
    
    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
        
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(train[1], num_classes)
    y_test = keras.utils.to_categorical(test[1], num_classes)
    
    # define a CNN model
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(16, kernel_size=(3, 3), activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.5),
            layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.5),
            layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(num_classes,activation='softmax'),
        ]
    )
    return model,x_train,y_train,x_test,y_test
