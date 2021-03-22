# Miro Kivela
# 19.3.2020
# Inspiration from this guide https://www.dbersan.com/blog/neural-boilerplate/, shows basic steps for handlifn hdf5 data

# Basically takes in the smaller hdf5 files and combines them into one big file, for supposed ease of use
# Resizes images to IMAGE_HEIGHT and IMAGE_WIDTH
# Optionally restrict the dataset to have 1:1 or 2:1 data, of each class
# Splits the data to train, val, and test in a 70:20:10 split

# The created dataset has the following structure
# HDF5-file/
# ├── train/
# │   ├── frames
# │   └── labels
# ├── test/
# │   ├── frames
# │   └── labels
# └── val/
#     ├── frames
#     └── labels

import cv2
import os
import os.path
import h5py
import h5py_cache
import numpy as np
from random import shuffle


IMAGE_HEIGHT = 400
IMAGE_WIDTH = 400

def main():
    print("This script will create or edit an .h5 dataset")
    cmd = input("Create Database file (C) or Add frames to database (A): ")
    cmd = cmd.strip().lower()
    if(cmd == 'a'):
        editDatabase()
    elif( cmd == 'c'):
        createDataset()
    else:
        print("Error: Unknown command, use A or C")

def editDatabase():
    database = input("Enter the target database filename: ")
    if os.path.isfile(database):
        # Ran into performance issues, due to lack of cache memory (among others) 
        # Found this solution, which uses another package, h5py_cache
        # https://stackoverflow.com/questions/39087689/writing-data-to-h5py-on-ssd-disk-appears-slow-what-can-i-do-to-speed-it-up/42966070#42966070
        # Might be able to use this for all the h5py tasks, but I'll leave as is for now
        targetDB = h5py_cache.File(database, mode='r+', chunk_cache_mem_size=500*1024**2)
        IMAGE_HEIGHT = targetDB['train/frames'].shape[1]
        IMAGE_WIDTH = targetDB['train/frames'].shape[2]
        while(True):
            origin = input("Enter the existing origin .h5 file or blank to exit: ")
            if origin.strip() == "":
                break
            if os.path.isfile(origin):
                addFrames(targetDB, origin)
            else:
                print("Error: Database couldn't be found")
        targetDB.close()
    else:
        print("Error: Database couldn't be found")
   

def resizeImage(image):
    img = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
    return img


def addFrames(targetDB, originDatabaseName):
    originDB = h5py.File(originDatabaseName, 'r')
    X = originDB['labels'].shape[0]
    Y = originDB['labels'].shape[1]
    cracks = []
    empty = []
    for x in range(X):
        for y in range(Y):
            if(originDB['labels'][x][y] == 1):
                cracks.append((x, y))
            else:
                empty.append((x, y))

    shuffle(cracks)
    shuffle(empty)

    # Making the data even
    # if(len(cracks) < len(empty)):
    #     empty = empty[:len(cracks)]
    #     print("Empty was limited to cracks count")

    # Making the data 2:1
    # if(2 * len(cracks) < len(empty)):
    #     empty = empty[:2 * len(cracks)]
    #     print("Empty was limited to 2 times the crack count")


    #The notation here is pretty awful, but essentially we're doing an 10:20:70 split
    crack10 = int(len(cracks) * 0.1)
    crack30 = int(len(cracks) * 0.3)
    empty10 = int(len(empty) * 0.1)
    empty30 = int(len(empty) * 0.3)

    trainSet = cracks[crack30:] + empty[empty30:]
    valSet = cracks[crack10:crack30] + empty[empty10:empty30]
    testSet = cracks[:crack10] + empty[:empty10]

    print(f"Found {len(cracks)} crack and {len(empty)} empty frames")
    print(f"Dividing into {len(trainSet)}:{len(valSet)}:{len(testSet)}")
     
    #Now, opening the directories within the targetDB
    trainFramesDB = targetDB['train/frames']
    trainLabelsDB = targetDB['train/labels']

    valFramesDB = targetDB['val/frames']
    valLabelsDB = targetDB['val/labels']

    testFramesDB = targetDB['test/frames']
    testLabelsDB = targetDB['test/labels']
    
    #Then saving the data
    print("Saving training set")
    saveData(trainFramesDB, trainLabelsDB, originDB, trainSet)
    print("Saving validation set")
    saveData(valFramesDB, valLabelsDB, originDB, valSet)
    print("Saving testing set")
    saveData(testFramesDB, testLabelsDB, originDB, testSet)

    originDB.close()

    print("Data written successfully")


def saveData(frameDB, labelsDB, originDB, dataset):
    oldSizeFrames = frameDB.shape[0]
    oldSizeLabels = labelsDB.shape[0]
    print(f"Old size: {oldSizeFrames} frames and {oldSizeLabels} labels")

    frameDB.resize(oldSizeFrames+len(dataset), axis=0)
    labelsDB.resize(oldSizeLabels+len(dataset), axis=0)
    
    #Saving the values
    for i in range(len(dataset)):
        if i%50 == 0 and i > 1 :
            print(f"Saved {i} of {len(dataset)}")
        frameDB[oldSizeFrames+i] = resizeImage(originDB['frames'][dataset[i][0]][dataset[i][1]])
        #frameDB[oldSizeFrames+i] = originDB['frames'][dataset[i][0]][dataset[i][1]]
        labelsDB[oldSizeLabels+i] = originDB['labels'][dataset[i][0]][dataset[i][1]]


def createDataset():
    f_name = input("Enter a filename for the new dataset: ")

    if(not(str(f_name).lower().endswith(".h5"))):
            print("Error: Wrong file extension, use '.h5'")
            return
    for File in os.listdir("."):
            if(File == f_name):
                    print("Error: File already exists")
                    return
                    
    new_file = h5py.File(f_name, mode='w')
    train = new_file.create_group("train")
    val = new_file.create_group("val")
    test = new_file.create_group("test")
    
    groups = [train, val, test]
    for group in groups:
            group.create_dataset('frames', (0, IMAGE_HEIGHT, IMAGE_WIDTH, 3), np.uint8,  
                    compression="gzip", maxshape=(None, IMAGE_HEIGHT, IMAGE_WIDTH, 3))
            group.create_dataset('labels', (0, 1), np.uint8, maxshape=(None, 1))
    new_file.close()
    print("Database created successfully")

main()
