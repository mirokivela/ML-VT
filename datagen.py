# Miro Kivela
# 19.3.2020
import cv2
import numpy as np
from random import shuffle
from math import ceil
from keras.preprocessing.image import ImageDataGenerator
import copy

# Data generators
# Heavily inspired by: https://www.dbersan.com/blog/neural-boilerplate/

def hdf5Batcher(frames, labels, image_datagen, batch_size, seed):
    sample_count  = frames.shape[0]
    count         = 0

    while True:
        batch_index = 0
        batches_list = list(range(int(ceil(float(sample_count) / batch_size))))
        shuffle(batches_list)

        while batch_index < len(batches_list):
            batch_number = batches_list[batch_index]
            start        = batch_number * batch_size
            end          = min(start + batch_size, sample_count)

            # Load data from disk and resize
            x = frames[start: end]
            y = labels[start: end]
            # Augment batch
            generator = image_datagen.flow(x,y,batch_size=batch_size, seed = seed + count)
            augmented_x, augmented_y = next(generator)

            batch_index += 1
            count       += 1

            yield augmented_x, augmented_y

# Augmentor for training dataset
train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.8,1.2],
        fill_mode='nearest')

# Augmentor for validation dataset 
val_datagen = ImageDataGenerator(rescale=1./255)

#  An augmentor for visual examination of the augmentations
visual_datagen = copy.deepcopy(train_datagen)
visual_datagen.rescale = None