import numpy as np
import matplotlib.pyplot as plt
import cv2
import logging
from scipy import signal
from easydict import EasyDict
from transform_methods import *

#np.random.seed(2)
logging.basicConfig(format="[%(filename)s: %(funcName)s] %(message)s", level=logging.DEBUG)
logger = logging.getLogger(__name__)
SMALL_SCREEN = False
cfg = EasyDict()

cfg.SMALL_SCREEN = False
cfg.DATASET = ['MNIST', 'CIFAR'][0]


if cfg.DATASET == 'MNIST':
    cfg.data_path = "MNIST/train-images-idx3-ubyte"
    cfg.labels_path = "MNIST/train-labels-idx1-ubyte"
    cfg.nrof_classes = 10
    cfg.dataset_type = ['train', 'valid', 'test'][0]
    cfg.shuffle = True
    cfg.batch_size = 600
    cfg.transforms = [Scale(28, 0.8), RandomRotateImage(-10, 10), Translate2D(3, 'up', roll=False),
                      Translate2D(3, 'down', roll=False), Translate2D(3, 'right', roll=False),
                      Translate2D(3, 'left', roll=False),
                      AdaptiveNormalize()]
    cfg.transform_probs = [0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 1]
    cfg.sample_type = ['default', 'balanced', 'prob'][2]
    cfg.epoch_size = 10
    cfg.probabilities = np.array([0.04, 0.04, 0.12, 0.1, 0.1, 0.2, 0.08, 0.08, 0.09, 0.15])

else:
    cfg.data_path = "CIFAR/train"
    cfg.labels_path = "CIFAR/trainLabels.csv"
    cfg.nrof_classes = 10
    cfg.dataset_type = ['train', 'valid', 'test'][0]
    cfg.shuffle = True
    cfg.batch_size = 10
    cfg.transforms = [Pad3D(56), Scale(56, 2), ChangeChannel(200, 2)]
    cfg.transform_probs = [1, 1, 0.5]
    cfg.sample_type = ['default', 'balanced', 'prob'][1]
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