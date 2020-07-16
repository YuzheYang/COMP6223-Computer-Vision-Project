from __future__ import division
from keras.layers import Input
from keras.models import Model
import os
import sys
import numpy as np
from config import *
from utilities import postprocess_predictions
from update_models import sam_vgg
from imageio import imread, imsave
import cv2

import random


def get_test(data):
    Xims_224 = np.zeros((1, 224, 224, 3))
    img = imread(data['image'])
    img_name = os.path.basename(data['image'])
    gaussian = np.zeros((1, 14, 14, nb_gaussian))
    if img.ndim == 2:
        copy = np.zeros((img.shape[0], img.shape[1], 3))
        copy[:, :, 0] = img
        copy[:, :, 1] = img
        copy[:, :, 2] = img
        img = copy

    r_img = cv2.resize(img, (224, 224))
    r_img = r_img.astype(np.float32)
    r_img[:, :, 0] -= img_channel_mean[0]
    r_img[:, :, 1] -= img_channel_mean[1]
    r_img[:, :, 2] -= img_channel_mean[2]
    r_img = r_img[:, :, ::-1]
    Xims_224[0, :] = np.copy(r_img)
    return [Xims_224, gaussian], img, img_name

if __name__ == '__main__':
    if len(sys.argv) != 1:
        raise NotImplementedError
    else:
        seed = 7
        random.seed(seed)
        test_data = []

        testing_images = [datasest_path + f for f in os.listdir(datasest_path) if
                           f.endswith(('.jpg', '.jpeg', '.png'))]
        testing_images.sort()

        for image in testing_images:
            annotation_data = {'image': image}
            test_data.append(annotation_data)

        phase = 'test'
        if phase == "test":
            x = Input(batch_shape=(1, 224, 224, 3))
            x_maps = Input(batch_shape=(1, 14, 14, nb_gaussian))
            m = Model(inputs=[x, x_maps], outputs=sam_vgg([x, x_maps]))

            print("Loading weights")
            m.load_weights('ASNet.h5')
            print("Making prediction")


            # if not os.path.exists(saliency_output):
            #     os.makedirs(saliency_output)

            for data in test_data:
                Ximg, original_image, img_name = get_test(data)
                predictions = m.predict(Ximg, batch_size=1)
                res_saliency = postprocess_predictions(predictions[6][0, :, :, 0], original_image.shape[1],
                                                       original_image.shape[0])
                imsave(saliency_output + '%s.png' % img_name[0:-4], res_saliency.astype(int))
                m.reset_states()
        else:
            raise NotImplementedError