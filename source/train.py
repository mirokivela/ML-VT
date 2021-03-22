# Miro Kivela
# 19.3.2020

from datetime import datetime
import h5py
# Keras imports
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from keras.applications.vgg16 import VGG16
# Datagenerator imports
from datagen import train_datagen, val_datagen, hdf5Batcher

DATA_FILE = 'testingSetS000.h5'
IMG_SIZE = 400
IMG_CHANNELS = 3
batch_size = 30
seed = 0
EPOCHS = 3

# Open file
hdf5_file = h5py.File(DATA_FILE, 'r')

# Load images
train_images = hdf5_file['train/frames']
val_images = hdf5_file['val/frames']

# Load labels
train_labels = hdf5_file['train/labels']
val_labels  = hdf5_file['val/labels']

train_count = train_images.shape[0]
#test_count = hdf5_file['test/frames'].shape[0]

# Data generators
train_generator = hdf5Batcher(train_images, train_labels, train_datagen, batch_size, seed)
val_generator  = hdf5Batcher(val_images, val_labels, val_datagen, 1, seed)

# Take Vgg16 but leave the top off
vgg16 = VGG16(weights = None, input_shape=(400,400,3), include_top = False)

# Build the rest
model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(14, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile model and print summary
model.compile(
    loss='binary_crossentropy',
    optimizer=keras.optimizers.Adam(),
    metrics=['accuracy']
)
model.summary()
input("Continue with input:")

logDir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
saveDir = "models/" + datetime.now().strftime("%m%d-%H%M")
callbacks = [  keras.callbacks.TensorBoard(log_dir=logDir, histogram_freq=1)
            , keras.callbacks.ModelCheckpoint( saveDir+'model.{epoch:02d}-{val_loss:.2f}.h5'
            , monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only = True) ]

model.fit_generator(train_generator, validation_data = val_generator, epochs=EPOCHS
            , steps_per_epoch= train_count//batch_size, callbacks = callbacks, validation_steps = 20)
print("Finished training")

finalSaveDir = "models/finalModel" + datetime.now().strftime("%m%d-%H%M" + ".h5")
model.save(finalSaveDir)
hdf5_file.close()
