#import tensorflow.compat.v1 as tf
import tensorflow as tf 
from tensorflow.keras.preprocessing.image import *
import numpy as np
import pickle
tf.compat.v1.disable_eager_execution()

classifier = tf.keras.Sequential()
# 32 means 32 kernals to extract
classifier.add(tf.keras.layers.Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
classifier.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
classifier.add(tf.keras.layers.Conv2D(32, (3, 3), activation= "relu"))
classifier.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
classifier.add(tf.keras.layers.Flatten())

classifier.add(tf.keras.layers.Dense(units = 128, activation = 'relu'))
classifier.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./225)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 160,
                         epochs = 1,
                         validation_data = test_set,
                         validation_steps = 4)

# pickle.dump(classifier, open('neural.sav', 'w'))
# classifier2 = pickle.load(open('neural.sav','rb'))
test_image = load_img('dataset/single_prediction/moto_img.png', target_size = (64, 64))
test_image = img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices

# first [0] is parent directory and second [0] is first folder inside the parent directory
if result[0][0] == 1:
    prediction = 'moto'
else:
    prediction = 'apple'

print(prediction)