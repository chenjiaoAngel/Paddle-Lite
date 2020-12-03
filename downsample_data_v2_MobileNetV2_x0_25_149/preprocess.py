import cv2
import numpy as np
def process_val_image(refimg_path):
    # params
    img = cv2.imread(refimg_path)
    img  = cv2.pyrDown(img)
    center = (150, 162)
    imgscope = 168
    inpsize = 112
    img_mean = np.array([127.5, 127.5, 127.5]).reshape((1, 1, 3))
    img_std = np.array([128.0, 128.0, 128.0]).reshape((1, 1, 3))
    # preprocess
    img = cv2.imread(refimg_path)
    img = cv2.pyrDown(img)
    pts1 = np.float32([[center[0] - imgscope / 2.0, center[1] - imgscope / 2.0],
                       [center[0] - imgscope / 2.0, center[1] + imgscope / 2.0],
                       [center[0] + imgscope / 2.0, center[1] - imgscope / 2.0]])
    pts2 = np.float32([[0, 0], [0, inpsize], [inpsize, 0]])
    mat = cv2.getAffineTransform(pts1, pts2)
    img = cv2.warpAffine(img, mat, (inpsize, inpsize))
    img = img.astype(np.float32)
    img -= img_mean
    img /= img_std
    img = img.transpose(2, 0, 1)  # [C,H,W]
    return img

if __name__ == '__main__':
    refimg_path='demo.jpg'
    img = process_val_image(refimg_path=refimg_path)
    print(img)
