import cv2
import numpy as np

'''   自适应中值滤波器的python实现   '''
def AdaptProcess(src, i, j, minSize, maxSize):

    filter_size = minSize

    kernelSize = filter_size // 2
    rio = src[i-kernelSize:i+kernelSize+1, j-kernelSize:j+kernelSize+1]
    minPix = np.min(rio)
    maxPix = np.max(rio)
    medPix = np.median(rio)
    zxy = src[i,j]

    if (medPix > minPix) and (medPix < maxPix):
        if (zxy > minPix) and (zxy < maxPix):
            return zxy
        else:
            return medPix
    else:
        filter_size = filter_size + 2
        if filter_size <= maxSize:
            return AdaptProcess(src, i, j, filter_size, maxSize)
        else:
            return medPix


def adapt_meadian_filter(img, minsize, maxsize):

    borderSize = maxsize // 2

    src = cv2.copyMakeBorder(img, borderSize, borderSize, borderSize, borderSize, cv2.BORDER_REFLECT)

    for m in range(borderSize, src.shape[0] - borderSize):
        for n in range(borderSize, src.shape[1] - borderSize):
            src[m,n] = AdaptProcess(src, m, n, minsize, maxsize)

    dst = src[borderSize:borderSize+img.shape[0], borderSize:borderSize+img.shape[1]]
    return dst

if __name__ == '__main__':
    img = cv2.imread('../blur/angle/imgs/cameramen_30_frequency.png', 0)
    cv2.imshow('result', adapt_meadian_filter(img, 3, 7))
    cv2.waitKey(0)

