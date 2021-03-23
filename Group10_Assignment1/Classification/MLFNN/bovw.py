

import numpy as np
import matplotlib.pyplot as plt
import math
import cv2 as cv
import os
from sklearn.cluster import KMeans
import pickle


def load_images_from_folder(folder):
    images = {}
    for filename in os.listdir(folder):
        category = []
        path = folder + "/" + filename
        for image in os.listdir(path):
            img = cv.imread(path + "/" + image)
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if img is not None:
                category.append(img)
        images[filename] = category
    return images


train_images = load_images_from_folder(r'C:\Users\Shikhar Kushwah\Desktop\projects-and-learnings\CS671-Deep-Learning-and-its-Applications\Assignment1\Group10\Classification\Image_Group10\train')  # take all images category by category 
test_images = load_images_from_folder(r'C:\Users\Shikhar Kushwah\Desktop\projects-and-learnings\CS671-Deep-Learning-and-its-Applications\Assignment1\Group10\Classification\Image_Group10\test') # take test images
val_images = load_images_from_folder(r'C:\Users\Shikhar Kushwah\Desktop\projects-and-learnings\CS671-Deep-Learning-and-its-Applications\Assignment1\Group10\Classification\Image_Group10\validation')

train_images["movie_theater_indoor"] = np.array(train_images["movie_theater_indoor"])
train_images["rock_arch"] = np.array(train_images["rock_arch"])
train_images["valley"] = np.array(train_images["valley"])
#print(train_images["movie_theater_indoor"].shape)
test_images["movie_theater_indoor"] = np.array(test_images["movie_theater_indoor"])
test_images["rock_arch"] = np.array(test_images["rock_arch"])
test_images["valley"] = np.array(test_images["valley"])
#print(test_images["valley"][5].shape)
#print(val_images.keys())
val_images["movie_theater_indoor"] = np.array(val_images["movie_theater_indoor"])
val_images["rock_arch"] = np.array(val_images["rock_arch"])
val_images["valley"] = np.array(val_images["valley"])       

def generate_feature_vector(img):
    #print("Image Size is : ",img.shape[1],"X",img.shape[0])
    #print(img[0][0])
    width = img.shape[1]
    height = img.shape[0]
    if(width%32 != 0):
        width = 32*((width//32) + 1)
    if(height%32 != 0):
        height = 32*((height//32) + 1)

    final_image = cv.resize(img,(width,height))
    #print(img.shape , final_image.shape)
    i = 0 
    feature_vector = []
    while(i < height):
        j = 0
        while(j < width):
            ki = i
            kj = j
            patch = []
            while(ki<(i+32)):
                kj = j
                while(kj<(j+32)):
                    patch.append(final_image[ki][kj])
                    kj = kj + 1
                ki = ki + 1
            #cnt = cnt + 1
            #print(cnt)
            #patches.append(patch)
            b = []
            g = []
            r = []
            for p in patch:
                b.append(p[0])
                g.append(p[1])
                r.append(p[2])
            feature_vector1 , x1 = np.histogram(b,bins=8)
            feature_vector2 , x2 = np.histogram(g,bins=8)
            feature_vector3 , x3 = np.histogram(r,bins=8)
            feature_vector.append(np.concatenate((feature_vector1,feature_vector2,feature_vector3)))
            j = kj
        i = ki 
    feature_vector = np.array(feature_vector)
    return feature_vector

#a = generate_feature_vector(train_images["valley"][0])
#print(a.shape)

#K-Means Step
#Making dataset for K-Means
#Feature vectors of all the training images , together will form the data for K-Means

#---------------------------------------------------------------
#dataset_structure train->val->test

dataset = []
cnt = 0
for image in train_images["movie_theater_indoor"]:
    cnt = cnt+1
    print("Task completed for image : ",cnt)
    f_vector = generate_feature_vector(image)
    for t in f_vector:
        dataset.append(t)
cnt =0
for image in train_images["rock_arch"]:
    cnt = cnt+1
    print("Task completed for image : ",cnt)
    f_vector = generate_feature_vector(image)
    for t in f_vector:
        dataset.append(t)
cnt =0
for image in train_images["valley"]:
    cnt = cnt+1
    print("Task completed for image : ",cnt)
    f_vector = generate_feature_vector(image)
    for t in f_vector:
        dataset.append(t)
print("Work Completed for train images")
cnt = 0
for image in val_images["movie_theater_indoor"]:
    cnt = cnt+1
    print("Task completed for image : ",cnt)
    f_vector = generate_feature_vector(image)
    for t in f_vector:
        dataset.append(t)
cnt =0
for image in val_images["rock_arch"]:
    cnt = cnt+1
    print("Task completed for image : ",cnt)
    f_vector = generate_feature_vector(image)
    for t in f_vector:
        dataset.append(t)
cnt =0
for image in val_images["valley"]:
    cnt = cnt+1
    print("Task completed for image : ",cnt)
    f_vector = generate_feature_vector(image)
    for t in f_vector:
        dataset.append(t)
print("work completed for validation images")
cnt = 0
for image in test_images["movie_theater_indoor"]:
    cnt = cnt+1
    print("Task completed for image : ",cnt)
    f_vector = generate_feature_vector(image)
    for t in f_vector:
        dataset.append(t)
cnt =0
for image in test_images["rock_arch"]:
    cnt = cnt+1
    print("Task completed for image : ",cnt)
    f_vector = generate_feature_vector(image)
    for t in f_vector:
        dataset.append(t)
cnt =0
for image in test_images["valley"]:
    cnt = cnt+1
    print("Task completed for image : ",cnt)
    f_vector = generate_feature_vector(image)
    for t in f_vector:
        dataset.append(t)
print("All tasks finished!!")
dataset = np.array(dataset)
print(dataset.shape)
np.savetxt("data.txt",dataset)
#--------------------------------------------------------------------------------------------------------
#print(dataset[0],dataset[1])

f = np.loadtxt('data.txt')
#print(f)
f = f.astype(int)
#print(f)
train_images_feature_count = 0
temp = ["movie_theater_indoor","rock_arch","valley"]
for i in temp:
    for img in train_images[i]:
        width = img.shape[1]
        height = img.shape[0]
        if(width%32 != 0):
            width = 32*((width//32) + 1)
        if(height%32 != 0):
            height = 32*((height//32) + 1)
        train_images_feature_count = train_images_feature_count + int((width*height)/(32*32))
#print(train_images_feature_count)
train_images_feature_vector = f[0:train_images_feature_count]

#Uncomment this part of code to train the model
#but before that, delete saved_model.sav file

kmeans = KMeans(init="random",n_clusters=32,n_init=10,max_iter=300,random_state=10)
kmeans.fit(train_images_feature_vector)
pickle.dump(kmeans,open('saved_model.sav','wb'))

#print(kmeans.cluster_centers_)
loaded_model = pickle.load(open('saved_model.sav','rb'))

#print(loaded_model.predict(f[0:100000]))

x_train = []
x_val = []
x_test = []

i = 0
k = ["movie_theater_indoor","rock_arch","valley"]
for i in k:
    print('initiating task for ',i)
    for image in train_images[i]:
        temp = generate_feature_vector(image)
        labels = loaded_model.predict(temp)
        width = image.shape[1]
        height = image.shape[0]
        if(width%32 != 0):
            width = 32*((width//32) + 1)
        if(height%32 != 0):
            height = 32*((height//32) + 1)
        patches_count = int((height*width)/(32*32))
        bovw = np.zeros(32)
        for j in range(len(labels)):
            bovw[labels[j]] += 1
        bovw = bovw/patches_count
        x_train.append(bovw)
    print('task completed for train images ',i)
    for image in val_images[i]:
        temp = generate_feature_vector(image)
        labels = loaded_model.predict(temp)
        width = image.shape[1]
        height = image.shape[0]
        if(width%32 != 0):
            width = 32*((width//32) + 1)
        if(height%32 != 0):
            height = 32*((height//32) + 1)
        patches_count = int((height*width)/(32*32))
        bovw = np.zeros(32)
        for j in range(len(labels)):
            bovw[labels[j]] += 1
        bovw = bovw/patches_count
        x_val.append(bovw)
    print('task completed for val images ',i)
    for image in test_images[i]:
        temp = generate_feature_vector(image)
        labels = loaded_model.predict(temp)
        width = image.shape[1]
        height = image.shape[0]
        if(width%32 != 0):
            width = 32*((width//32) + 1)
        if(height%32 != 0):
            height = 32*((height//32) + 1)
        patches_count = int((height*width)/(32*32))
        bovw = np.zeros(32)
        for j in range(len(labels)):
            bovw[labels[j]] += 1
        bovw = bovw/patches_count
        x_test.append(bovw)
    print('task completed for test images ',i)
x_train = np.array(x_train)
x_val = np.array(x_val)
x_test = np.array(x_test)
np.savetxt('x_train.txt',x_train)
np.savetxt('x_val.txt',x_val)
np.savetxt('x_test.txt',x_test)

