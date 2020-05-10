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

cfg.test_data_path = "MNIST/t10k-images-idx3-ubyte"
cfg.test_label_path = "MNIST/t10k-labels-idx1-ubyte"


cfg.train_data_path = "MNIST/train-images-idx3-ubyte"
cfg.train_labels_path = "MNIST/train-labels-idx1-ubyte"
cfg.nrof_classes = 10
cfg.dataset_type = ['train', 'valid', 'test'][0]
cfg.shuffle = True
cfg.batch_size = 32
cfg.transforms = [AdaptiveNormalize()]
cfg.transform_probs = [1]
cfg.sample_type = ['default', 'balanced', 'prob'][1]
cfg.epoch_size = 64
cfg.probabilities = np.array([0.04, 0.04, 0.12, 0.1, 0.1, 0.2, 0.08, 0.08, 0.09, 0.15])


def show_image(img, name='image'):
    max, min = img.max(), img.min()
    img = ((img - min) / (max-min) * 255).astype(np.uint8)

    if SMALL_SCREEN:
        img = cv2.resize(img, (int(img.shape[1] * 0.5), int(img.shape[0] * 0.5)))
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()