import numpy as np
import cv2
from PIL import Image

class HistogramEqualization(object):
    def __call__(self, img):
        img = np.array(img)
        img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
        return Image.fromarray(img)
