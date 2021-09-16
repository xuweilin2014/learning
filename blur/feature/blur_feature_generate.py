import numpy as np
import os
import cv2
from blur_kernel.blur import generate_blur_kernel
from blur_feature import *
import random

def generate_blur_feature(image, ):
    pass

def synthesize_dataset(path, patch_size=8):
    counter = 0

    blur_dir = './blur_dataset/'
    sharp_dir = './sharp_dataset/'

    for file_path, dir_names, file_names in os.walk(path):
        for dir_name in dir_names:
            dir_path = os.path.join(file_path, dir_name)
            for img_name in os.listdir(dir_path):
                image = cv2.imread(os.path.join(dir_path, img_name))

                if len(image.shape) == 2:
                    image = np.reshape(image, image.shape + (1,))

                hs, ws = image.shape[0] // patch_size, image.shape[1] // patch_size
                for i in range(hs):
                    for j in range(ws):
                        img = image[i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size, :]
                        blur_length = random.randint(20, 50)
                        blur_angle = random.randint(0, 180)

                        kernel, anchor = generate_blur_kernel(blur_length, blur_angle)
                        img_blur = cv2.filter2D(img, -1, kernel, anchor=anchor)

                        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        img_gray = (img_gray - np.min(img_gray)) / (np.max(img_gray) - np.min(img_gray))
                        img_blur_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
                        img_blur_gray = (img_blur_gray - np.min(img_blur_gray)) / (np.max(img_blur_gray) - np.min(img_blur_gray))

                        if np.mean(local_auto_correlation(img_gray)) <= 0.02 or np.mean(local_auto_correlation(img_blur_gray)) <= 0.02:
                            continue

                        if not os.path.exists(blur_dir):
                            os.mkdir(blur_dir)
                        if not os.path.exists(sharp_dir):
                            os.mkdir(sharp_dir)

                        cv2.imwrite(blur_dir + str(counter) + '_blur.jpg', img_blur)
                        cv2.imwrite(sharp_dir + str(counter) + '.jpg', img)

                        counter += 1


if __name__ == '__main__':
    dir_path = 'C:\\Users\\Lenovo\\Desktop\\corel dataset'
    synthesize_dataset(dir_path, 64)


