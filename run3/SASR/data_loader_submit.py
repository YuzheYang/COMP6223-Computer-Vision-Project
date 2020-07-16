import os
import numpy as np
import cv2

def training_data_loader():
    root_path_trainig = './training/'
    class_name = os.listdir('training')
    x_train = []
    x_test = []
    y_train = []
    y_test = []

    for idx, folder_name in enumerate(class_name):
        one_hot_label = np.zeros(len(class_name))
        image_path = root_path_trainig + folder_name
        image_name = os.listdir(image_path)
        for img_name in image_name[0:90]:
            img = cv2.imread(image_path+'/'+img_name)
            if img is None:
                print(image_path + '/' + img_name)
            img = np.float32(img)
            # img /= 255.
            img = (img -127.5)/127.5
            img = cv2.resize(img, (224, 224))
            x_train.append(img)
            one_hot_label[idx]=1
            y_train.append(one_hot_label)

        for img_name in image_name[90:100]:
            img = cv2.imread(image_path + '/' + img_name)
            if img is None:
                print(image_path + '/' + img_name)
            img = np.float32(img)
            img = (img -127.5)/127.5
            img = cv2.resize(img, (224, 224))
            x_test.append(img)
            one_hot_label[idx] = 1
            y_test.append(one_hot_label)
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    print('Training set Dataloader Shape Check...')
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    return  x_train, y_train, x_test, y_test

def saliency_train_data_loader():
    root_path_trainig = './SaliencyMap/training/'
    class_name = os.listdir('./SaliencyMap/training/')
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for idx, folder_name in enumerate(class_name):
        one_hot_label = np.zeros(len(class_name))
        image_path = root_path_trainig + folder_name
        image_name = os.listdir(image_path)
        for img_name in image_name[0:90]:
            img = cv2.imread(image_path+'/'+img_name)
            # if img is None:
            #     print(image_path + '/' + img_name)
            img = np.float32(img)
            img /= 255.
            img = cv2.resize(img, (224, 224))
            x_train.append(img)
            one_hot_label[idx]=1
            y_train.append(one_hot_label)
        for img_name in image_name[90:100]:
            img = cv2.imread(image_path+'/'+img_name)
            if img is None:
                print(image_path + '/' + img_name)
            img = np.float32(img)
            img /= 255.
            img = cv2.resize(img, (224, 224))
            x_test.append(img)
            one_hot_label[idx]=1
            y_test.append(one_hot_label)

    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    print('Training set SaliencyMap Dataloader Shape Check...')
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    return x_train, y_train, x_test, y_test