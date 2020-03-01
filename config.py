import numpy as np
import matplotlib.pyplot as plt
import cv2
import logging
from scipy import signal
from easydict import EasyDict
from transform_methods import *

logging.basicConfig(format="[%(filename)s: %(funcName)s] %(message)s", level=logging.DEBUG)
logger = logging.getLogger(__name__)

SMALL_SCREEN = False
cfg = EasyDict()
# train sources =
cfg.data_path = "MNIST/train-images-idx3-ubyte"
cfg.labels_path = "MNIST/train-labels-idx1-ubyte"
cfg.nrof_classes = 10
cfg.dataset_type = ['train', 'valid', 'test'][0]
cfg.shuffle = True
cfg.batch_size = 8
cfg.transforms = [GreyToBRG(), Pad3D(56, 0, mode='symmetric'),  Scale(56, 2), GaussianNoise3D(0, 0.1), Salt(0.05), Pepper(0.05)]
cfg.sample_type = ['default', 'balanced', 'prob'][0]
cfg.epoch_size = None
cfg.probabilities = [.5, .3, .2, 0, 0, 0, 0, 0, 0, 0]


def show_image(img, name='image'):
    max, min = img.max(), img.min()
    img = ((img - min) / (max-min) * 255).astype(np.uint8)

    if SMALL_SCREEN:
        img = cv2.resize(img, (int(img.shape[1] * 0.5), int(img.shape[0] * 0.5)))
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()