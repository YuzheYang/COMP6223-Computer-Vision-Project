import cv2
import numpy as np
import os
from keras.models import load_model

root_path = './testing/'
root_path_SAL = 'SaliencyMap/testing/'
img_name = os.listdir(root_path)



f = open('run3.txt','w+')
for idx, name in enumerate(img_name):
    img_name[idx] = name.replace('.jpg','')
files = sorted(img_name, key=int)
print(len(files))
model = load_model(r'C:\Users\ian_c\Desktop\ECS\COMP6223CV\CourseWork3\SSRSubmitModel.h5')

for idx in files:
    img = cv2.imread(root_path+idx+'.jpg')
    if img is None:
         continue
    img = np.float32(img)
    # img /= 255.
    img = (img-127.5)/127.5
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)

    sal = cv2.imread(root_path_SAL + idx + '.png')
    if sal is None:
        continue
    sal = np.float32(sal)
    sal /= 255.

    sal = cv2.resize(sal, (224, 224))
    sal = np.expand_dims(sal, axis=0)

    pred = model.predict([img, sal])[0]
    max_index = np.argmax(pred)
    class_name = os.listdir('training')
    pred_class = class_name[int(max_index)]
    f.writelines(str(idx)+'.jpg' + ' '+ pred_class)
    f.writelines('\n')

print('Done!')