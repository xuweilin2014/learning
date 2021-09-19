import os
import random
from blur_kernel.blur import generate_blur_kernel
from gradient_magnitude.gradient_magnitude import *
from local_kurtosis.local_kurtosis import *
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from svm import blur_fit

dir_path = 'C:\\Users\\Lenovo\\Desktop\\corel images'
blur_dir = '../blur_dataset\\'
sharp_dir = '../sharp_dataset\\'

def synthesize_dataset(patch_size=64):
    counter = 0
    src_dir = 'C:\\Users\\Lenovo\\Desktop\\corel images\\'

    if not os.path.exists(blur_dir):
        os.mkdir(blur_dir)
    if not os.path.exists(sharp_dir):
        os.mkdir(sharp_dir)

    for img_name in os.listdir(src_dir):
        src_path = os.path.join(src_dir, img_name)
        image = cv2.imread(src_path)

        if len(image.shape) == 2:
            image = np.reshape(image, image.shape + (1,))

        hs, ws = image.shape[0] // patch_size, image.shape[1] // patch_size

        for i in range(hs):
            for j in range(ws):
                img = image[i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size, :]
                blur_length = random.randint(5, 32)
                blur_angle = random.randint(0, 180)

                kernel, anchor = generate_blur_kernel(blur_length, blur_angle)
                img_blur = cv2.filter2D(img, -1, kernel, anchor=anchor)

                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img_gray = (img_gray - np.min(img_gray)) / (np.max(img_gray) - np.min(img_gray))
                img_blur_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
                img_blur_gray = (img_blur_gray - np.min(img_blur_gray)) / (np.max(img_blur_gray) - np.min(img_blur_gray))

                cv2.imwrite(blur_dir + str(counter) + '_blur.jpg', img_blur)
                cv2.imwrite(sharp_dir + str(counter) + '.jpg', img)

                counter += 1

def generate_feature(img_id, dictionary, lock, patch_size=64):
    img = cv2.imread(os.path.join(sharp_dir, str(img_id) + '.jpg'))
    img_blur = cv2.imread(os.path.join(blur_dir, str(img_id) + '_blur.jpg'))

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = (img_gray - np.min(img_gray)) / (np.max(img_gray) - np.min(img_gray))
    img_blur_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
    img_blur_gray = (img_blur_gray - np.min(img_blur_gray)) / (np.max(img_blur_gray) - np.min(img_blur_gray))

    I3 = local_auto_correlation(img_gray)
    B3 = local_auto_correlation(img_blur_gray)

    if np.mean(I3) <= 0.02 or np.mean(B3) <= 0.02:
        return None

    I1 = local_kurtosis(img_gray, 11, blur=False, display=False)
    I2 = gradient_histogram_span(img_gray, 11, blur=False, display=False)
    I = np.concatenate((I1, I2, I3, [0]), axis=0)

    B1 = local_kurtosis(img_blur_gray, 11, blur=True, display=False)
    B2 = gradient_histogram_span(img_blur_gray, 11, blur=True, display=False)
    B = np.concatenate((B1, B2, B3, [1]), axis=0)

    with lock:
        blur_feature = dictionary['blur']

        if len(blur_feature) == 0:
            blur_feature = np.concatenate((I.reshape((1, len(I))), B.reshape((1, len(B)))), axis=0)
        else:
            blur_feature = np.concatenate((blur_feature, I.reshape((1, len(I))), B.reshape((1, len(B)))), axis=0)

        dictionary['blur'] = blur_feature

    print(os.path.join(sharp_dir, str(img_id) + '.jpg') + ' done')


def main():
    limit = 20000
    works = 20

    d = mp.Manager().dict()
    lock = mp.Manager().Lock()

    for id_list in np.array_split(np.arange(0, limit), works):
        d['blur'] = np.array([])
        with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
            for i in id_list:
                executor.submit(generate_feature, i, d, lock)
        executor.shutdown()
        path = './blur/blur_feature' + str(id_list[0]) + '.npy'
        if not os.path.exists(path):
            open(path, 'w+').close()
        np.save(path, d['blur'])
        print('saving ' + path)

    print('done')

if __name__ == '__main__':
    main()
    blur_fit.hog_feature_fit()
    # synthesize_dataset(patch_size=64)

