# Miro Kivela
# 19.3.2020

import h5py
from random import randint
import matplotlib.pyplot as plt
import numpy as np
from numpy import expand_dims
from keras.preprocessing.image import ImageDataGenerator
import datagen

# Setting the font to Times New Roman, so that it matched my thesis
plt.rcParams['font.family'] = 'Times New Roman'

# Dataset filename 
DATA_FILE = 'cracks400x400.h5'

# Open dataset, find images and labels
dataset = h5py.File(DATA_FILE, 'r')
x_train = dataset['train/frames']
y_train = dataset['train/labels']
data_count = y_train.shape[0]

# Stats from the dataset
crack_count = np.count_nonzero(y_train)
empty_count = data_count - crack_count
print(f'This data set has {crack_count} cracks and {empty_count} empty frames')

# Select random image
index = randint(0, data_count)
image = x_train[index]
label = y_train[index]

# Plot reference image
grid = plt.GridSpec(3,4, wspace = 0.1, hspace=0.05)
sub = plt.subplot(grid[1,0])
sub.axis('off')
sub.title.set_text('Alkuperäinen')
sub.text(0.5, -0.1, f'Label: {label}', size=12, ha="center", va="top", 
         transform=sub.transAxes)
plt.imshow(image)

# Plot example augmentations
augment = datagen.visual_datagen
image = expand_dims(image, 0)
it = augment.flow(image, batch_size=1)

for i in range(3):
    for j in range(1,3+1):
        sub = plt.subplot(grid[i,j])
        sub.axis('off')
        batch = it.next()
        image = batch[0].astype('uint8')
        plt.imshow(image)

# Show the figure
plt.show()

# Visualize bactch generation
batch_size = 16
seed = 0

batch_generator = datagen.hdf5Batcher(x_train, y_train, augment, batch_size, seed)
x, y = next(batch_generator)

for i in range(16):
    sub = plt.subplot(4, 4, 1+i)
    sub.axis('off')
    sub.title.set_text(f'{y[i]}')
    image = x[i].astype('uint8')
    plt.imshow(image)

# Show the figure
plt.tight_layout()
plt.show()

dataset.close()
