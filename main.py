import sys

import cv2
import numpy as np

IN_IMG = 'Wind Waker GC.bmp'
THRESHOLD = 0.5
ALFA = 0.7
BETA = 0.3


def bright_pass(img, threshold):
    mask = np.zeros_like(img)
    for row in range(len(img)):
        for col in range(len(img[0])):
            sum = (img[row][col][0] +
                   img[row][col][1] +
                   img[row][col][2]) / 3

            if sum > (threshold):
                mask[row, col] = img[row, col]
            else:
                mask[row, col] = 0

    return mask

def bloom(img, mask):
    output = np.zeros_like(img)
    for row in range(len(img)):
        for col in range(len(img[0])):
            output[row][col] = (ALFA * img[row][col]) + (BETA * mask[row][col])

    return output


if __name__ == '__main__':
    img = cv2.imread(IN_IMG)
    img = img.astype(np.float32) / 255

    if img is None:
        print('Erro abrindo %s' % IN_IMG)
        sys.exit()
    else:
        # cv2.imwrite('original.jpeg', img * 255)
        print("Start")

    mask = bright_pass(img, THRESHOLD)
    blurred_mask1 = cv2.GaussianBlur(mask, (0, 0), 10)
    blurred_mask2 = cv2.GaussianBlur(mask, (0, 0), 15)
    blurred_mask3 = cv2.GaussianBlur(mask, (0, 0), 30)

    gaussian_mask = blurred_mask1 + blurred_mask2 + blurred_mask3
    gaussian_bloom = bloom(img, gaussian_mask)
    cv2.imwrite('gaussian_bloom.jpeg', gaussian_bloom * 255)
    cv2.imshow('gaussian_bloom', gaussian_bloom.astype(np.float32))

    box_mask_A1 = cv2.boxFilter(src=mask, ddepth=-1, ksize=(19, 19))
    box_mask_A2 = cv2.boxFilter(src=box_mask_A1, ddepth=-1, ksize=(19, 19))
    box_mask_A3 = cv2.boxFilter(src=box_mask_A2, ddepth=-1, ksize=(19, 19))

    box_mask_B1 = cv2.boxFilter(src=mask, ddepth=-1, ksize=(29, 29))
    box_mask_B2 = cv2.boxFilter(src=box_mask_B1, ddepth=-1, ksize=(29, 29))
    box_mask_B3 = cv2.boxFilter(src=box_mask_B2, ddepth=-1, ksize=(29, 29))

    box_mask_C1 = cv2.boxFilter(src=mask, ddepth=-1, ksize=(51, 51))
    box_mask_C2 = cv2.boxFilter(src=box_mask_C1, ddepth=-1, ksize=(51, 51))
    box_mask_C3 = cv2.boxFilter(src=box_mask_C2, ddepth=-1, ksize=(51, 51))

    box_mask = box_mask_A3 + box_mask_B3 + box_mask_C3
    box_bloom = bloom(img, box_mask)
    cv2.imwrite('box_bloom.jpeg', box_bloom * 255)
    cv2.imshow('box_bloom', box_bloom.astype(np.float32))

    # cv2.imshow('box_mask_A3', box_mask_A3.astype(np.float32))
    # cv2.imshow('box_mask_B3', box_mask_B3.astype(np.float32))
    # cv2.imshow('box_mask_C3', box_mask_C3.astype(np.float32))

    # cv2.imshow('blurred_mask1', blurred_mask1.astype(np.float32))
    # cv2.imshow('blurred_mask2', blurred_mask2.astype(np.float32))
    # cv2.imshow('blurred_mask3', blurred_mask3.astype(np.float32))
    # cv2.imshow('gaussian_mask', gaussian_mask.astype(np.float32))

    # cv2.imshow('original', img)
    # cv2.imshow('mascara', mask)

    cv2.waitKey()
    cv2.destroyAllWindows()

