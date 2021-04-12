# %%
import os
import shutil


import matplotlib.pyplot as plt
import numpy as np
import PIL
from PIL import Image
import os, sys
from scipy.io import loadmat

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.constraints import maxnorm
from keras.utils import np_utils

# %%

def load_data(data_path, classes, dataset='train', image_size=64):

    num_images = 0
    for i in range(len(classes)):
        dirs = sorted(os.listdir(data_path + dataset + '/' + classes[i]))
        num_images += len(dirs)
                                
    x = np.zeros((num_images, image_size, image_size, 3))
    y = np.zeros((num_images, 1))
    
    current_index = 0
    
    # Parcours des différents répertoires pour collecter les images
    for idx_class in range(len(classes)):
        dirs = sorted(os.listdir(data_path + dataset + '/' + classes[idx_class]))
        num_images += len(dirs)
    
        # Chargement des images, 
        for idx_img in range(len(dirs)):
            item = dirs[idx_img]
            if os.path.isfile(data_path + dataset + '/' + classes[idx_class] + '/' + item):
                # Ouverture de l'image
                img = Image.open(data_path + dataset + '/' + classes[idx_class] + '/' + item)
                # Redimensionnement de l'image et écriture dans la variable de retour x 
                img = img.resize((image_size,image_size))
                x[current_index] = np.asarray(img)
                # Écriture du label associé dans la variable de retour y
                y[current_index] = idx_class
                current_index += 1
                
    return x, y

    


# %%

path = "../data/"
labels = ["Chogath", "Ezreal", "Lucian", "Malzahar", "Morgana", "Poppy", "Reksai", "Senna", "Syndra", "Teemo"]

#x_train, y_train = load_data(path, labels, dataset='train', image_size=600)
#print(x_train.shape, y_train.shape)

#x_val, y_val = load_data(path, labels, dataset='validation', image_size=600)
#print(x_val.shape, y_val.shape)

#x_test, y_test = load_data(path, labels, dataset='test', image_size=600)
#print(x_test.shape, y_test.shape)

#class_num = y_test.shape[1]

#plt.figure(figsize=(12, 12))
#shuffle_indices = np.random.permutation(9)
#for i in range(0, 9):
#    plt.subplot(3, 3, i+1)
#    image = x_train[shuffle_indices[i]]
#    plt.title(labels[int(y_train[shuffle_indices[i]])])
#    plt.imshow(image/255)

#plt.tight_layout()
#plt.show()

# %% 
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator()

# load and iterate training dataset
train_it = datagen.flow_from_directory('../data/train/', class_mode='sparse', batch_size=64)
# load and iterate validation dataset
val_it = datagen.flow_from_directory('../data/validation/', class_mode='sparse', batch_size=64)
# load and iterate test dataset
test_it = datagen.flow_from_directory('../data/test/', class_mode='sparse', batch_size=64)



# %%
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(3, 600, 600), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dropout(0.2))

model.add(Dense(256, kernel_constraint=maxnorm(3)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
    
model.add(Dense(128, kernel_constraint=maxnorm(3)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(len(labels)))
model.add(Activation('softmax'))


optimizer = 'adam'

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

print(model.summary())

# %%
step_size_train = train_it.n//train_it.batch_size
step_size_valid = val_it.n//val_it.batch_size

model.fit_generator(generator=train_it,
                   steps_per_epoch = step_size_train,
                   epochs = 10,
                   verbose = 1,
                   validation_data = val_it,
                   validation_steps = step_size_valid)

# %%
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))