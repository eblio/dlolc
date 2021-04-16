# %%


# %%



# %%



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

model.add(Conv2D(32, (3, 3), input_shape=(600,600,3), padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
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
model.add(Dense(num_classes))
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