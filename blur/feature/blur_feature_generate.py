import os
import random
from skimage.feature import hog
from blur_kernel.blur import generate_blur_kernel
from gradient_magnitude.gradient_magnitude import *
from local_kurtosis.local_kurtosis import *
import shutil
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

dir_path = 'C:\\Users\\Lenovo\\Desktop\\corel images'

def generate_blur_feature(path):
    counter = 0
    dst_dir = 'C:\\Users\\Lenovo\\Desktop\\corel images\\'
    for file_path, dir_names, file_names in os.walk(path):
        for dir_name in dir_names:
            dir_path = os.path.join(file_path, dir_name)
            for img_name in os.listdir(dir_path):
                src_path = os.path.join(dir_path, img_name)
                dst_path = os.path.join(dst_dir, str(counter) + '.jpg')
                shutil.move(src_path, dst_path)
                counter += 1

def synthesize_dataset(img_id, dictionary, lock, patch_size=64):
    image = cv2.imread(os.path.join(dir_path, str(img_id) + '.jpg'))

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

            I3 = local_auto_correlation(img_gray)
            B3 = local_auto_correlation(img_blur_gray)

            if np.mean(I3) <= 0.02 or np.mean(B3) <= 0.02:
                continue

            I1 = local_kurtosis(img_gray, 11, blur=False, display=False)
            I2 = gradient_histogram_span(img_gray, 11, blur=False, display=False)
            I = np.concatenate((I1, I2, I3, [0]), axis=0)

            B1 = local_kurtosis(img_blur_gray, 11, blur=True, display=False)
            B2 = gradient_histogram_span(img_blur_gray, 11, blur=True, display=False)
            B = np.concatenate((B1, B2, B3, [1]), axis=0)

            # Ihog = hog(img_gray, orientations=12, block_norm='L1', pixels_per_cell=[8, 8], cells_per_block=[4, 4], visualize=False, transform_sqrt=True)
            # Bhog = hog(img_blur_gray, orientations=12, block_norm='L1', pixels_per_cell=[8, 8], cells_per_block=[4, 4], visualize=False, transform_sqrt=True)
            #
            # Ihog = np.concatenate((Ihog, [0]), axis=0)
            # Bhog = np.concatenate((Bhog, [1]), axis=0)

            with lock:
                # hog_feature = dictionary['hog']
                blur_feature = dictionary['blur']

                if len(blur_feature) == 0:
                    blur_feature = np.concatenate((I.reshape((1, len(I))), B.reshape((1, len(B)))), axis=0)
                else:
                    blur_feature = np.concatenate((blur_feature, I.reshape((1, len(I))), B.reshape((1, len(B)))), axis=0)

                # if len(hog_feature) == 0:
                #     hog_feature = np.concatenate((Ihog.reshape((1, len(Ihog))), Bhog.reshape((1, len(Bhog)))), axis=0)
                # else:
                #     hog_feature = np.concatenate((hog_feature, Ihog.reshape((1, len(Ihog))), Bhog.reshape((1, len(Bhog)))), axis=0)

                # dictionary['hog'] = hog_feature
                dictionary['blur'] = blur_feature

    print(os.path.join(dir_path, str(img_id) + '.jpg') + ' done')


def main():
    limit = 3000
    works = 20

    d = mp.Manager().dict()
    lock = mp.Manager().Lock()

    for id_list in np.array_split(np.arange(0, limit), works):
        d['hog'] = np.array([])
        d['blur'] = np.array([])
        with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
            for i in id_list:
                executor.submit(synthesize_dataset, i, d, lock)
        executor.shutdown()
        path = './data/blur_feature' + str(id_list[0]) + '.npy'
        if not os.path.exists(path):
            open(path, 'w+').close()
        np.save('./data/blur_feature' + str(id_list[0]) + '.npy', d['blur'])
        print('saving ' + path)

    # np.save('blur_feature1.npy', d['blur'])
    # np.save('hog_feature1.npy', d['hog'])
    print('done')

if __name__ == '__main__':
    main()
    # f = open('test.npy', 'w+')
    # f.close()
    # blur = np.load('blur_feature1.npy')
    # print(blur)
    # hog = np.load('hog_feature1.npy')
    # print(hog)


