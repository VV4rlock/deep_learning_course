from config import *


class Pad2D(object):
    def __init__(self, image_size, fill=0, mode='constant'):
        """
        :param image_size (int or tuple): размер итогового изображения. Если одно число, на выходе будет
        квадратное изображение. Если 2 числа - прямоугольное.
        :param fill (int or tuple): значение, которым будет заполнены поля. Если одно число, все каналы будут заполнены
        этим числом. Если 3 - соответственно по каналам.
        :param mode (string): тип заполнения:
        constant: все поля будут заполнены значение fill;
        edge: все поля будут заполнены пикселями на границе;
        reflect: отображение изображения по краям (прим. [1, 2, 3, 4] => [3, 2, 1, 2, 3, 4, 3, 2])
        symmetric: симметричное отображение изображения по краям (прим. [1, 2, 3, 4] => [2, 1, 1, 2, 3, 4, 4, 3])
        """
        mode_function = {'constant':self.constant_pad, 'edge': self.edge_pad,
                         'reflect': self.reflect_pad, 'symmetric': self.symmetric_pad}
        assert mode in ['constant', 'edge', 'reflect', 'symmetric']
        self.call = mode_function[mode]

        if isinstance(image_size, int):
            self.out_shape = (image_size, image_size)
        elif isinstance(image_size, tuple):
            self.out_shape = image_size
        else:
            raise Exception("Unknown image_size type")

        self.fill = fill

    def is_train_only(self):
        return False

    def is_mandatory(self):
        return True

    def constant_pad(self, image):
        h, w = (self.out_shape[0] - image.shape[0]) // 2, (self.out_shape[1] - image.shape[1]) // 2
        out = np.zeros(self.out_shape, dtype=np.uint8) + self.fill
        H, W = out.shape
        out[h:H - h, w:W - w] = image

        return out

    def edge_pad(self, image):
        h, w = (self.out_shape[0] - image.shape[0]) // 2, (self.out_shape[1] - image.shape[1]) // 2
        out = np.zeros(self.out_shape, dtype=np.uint8)
        H, W = out.shape
        out[h:H - h, w:W - w] = image

        out[0:h, 0:w] = out[h, w]
        out[0:h, W - w:W] = out[h, W - w]
        out[H - h:H, W - w:W] = out[H - h, W - w]
        out[H - h:H, 0:w] = out[H - h, w]

        out[0:h, w:W - w] = out[h, w:W - w]
        out[H - h:H, w:W - w] = out[H - h, w:W - w]
        out[h:H - h, 0:w] = out[h:H - h, w].reshape((H - 2 * h, 1))
        out[h:H - h, W - w:W] = out[h:H - h, W - w].reshape((H - 2 * h, 1))
        return out

    def reflect_pad(self, image):
        h, w = (self.out_shape[0] - image.shape[0]) // 2, (self.out_shape[1] - image.shape[1]) // 2
        out = np.zeros(self.out_shape, dtype=np.uint8)
        H, W = out.shape
        out[h:H - h, w:W - w] = image

        out[0:h, 0:w] = np.flip(np.flip(out[h + 1:2 * h + 1, w + 1: 2 * w + 1], 0), 1)
        out[0:h, W - w:W] = np.flip(np.flip(out[h + 1:2 * h + 1, W - 2 * w - 1: W - w - 1], 0), 1)
        out[H - h:H, W - w:W] = np.flip(np.flip(out[H - 2 * h - 1:H - h - 1, W - 2 * w - 1: W - w - 1], 0), 1)
        out[H - h:H, 0:w] = np.flip(np.flip(out[H - 2 * h - 1:H - h - 1, w + 1: 2 * w + 1], 0), 1)

        out[0:h, w:W - w] = np.flip(out[h + 1:2 * h + 1, w:W - w], 0)
        out[H - h:H, w:W - w] = np.flip(out[H - 2 * h - 1:H - h - 1, w:W - w], 0)
        out[h:H - h, 0:w] = np.flip(out[h:H - h, w + 1: 2 * w + 1], 1)
        out[h:H - h, W - w:W] = np.flip(out[h:H - h, W - 2 * w - 1: W - w - 1], 1)
        return out

    def symmetric_pad(self, image):
        h, w = (self.out_shape[0] - image.shape[0]) // 2, (self.out_shape[1] - image.shape[1]) // 2
        out = np.zeros(self.out_shape, dtype=np.uint8)
        H, W = out.shape
        out[h:H - h, w:W - w] = image

        out[0:h, 0:w] = np.flip(np.flip(out[h:2 * h, w: 2 * w], 0), 1)
        out[0:h, W - w:W] = np.flip(np.flip(out[h:2 * h, W - 2 * w: W - w], 0), 1)
        out[H - h:H, W - w:W] = np.flip(np.flip(out[H - 2 * h:H - h, W - 2 * w: W - w], 0), 1)
        out[H - h:H, 0:w] = np.flip(np.flip(out[H - 2 * h:H - h, w: 2 * w], 0), 1)

        out[0:h, w:W - w] = np.flip(out[h:2 * h, w:W - w], 0)
        out[H - h:H, w:W - w] = np.flip(out[H - 2 * h:H - h, w:W - w], 0)
        out[h:H - h, 0:w] = np.flip(out[h:H - h, w: 2 * w], 1)
        out[h:H - h, W - w:W] = np.flip(out[h:H - h, W - 2 * w: W - w], 1)
        return out


class Pad3D(object):
    def __init__(self, image_size, fill=0, mode='constant'):
        """
        :param image_size (int or tuple): размер итогового изображения. Если одно число, на выходе будет
        квадратное изображение. Если 2 числа - прямоугольное.
        :param fill (int or tuple): значение, которым будет заполнены поля. Если одно число, все каналы будут заполнены
        этим числом. Если 3 - соответственно по каналам.
        :param mode (string): тип заполнения:
        constant: все поля будут заполнены значение fill;
        edge: все поля будут заполнены пикселями на границе;
        reflect: отображение изображения по краям (прим. [1, 2, 3, 4] => [3, 2, 1, 2, 3, 4, 3, 2])
        symmetric: симметричное отображение изображения по краям (прим. [1, 2, 3, 4] => [2, 1, 1, 2, 3, 4, 4, 3])
        """
        self.pad2d = Pad2D(image_size,fill,mode)
        self.image_shape = self.pad2d.out_shape

    def call(self, image):
        res = np.zeros((self.image_shape[0],self.image_shape[1], 3))
        for channel in range(image.shape[2]):
            res[:, :, channel] = self.pad2d.call(image[:, :, channel])
        return res

    def is_train_only(self):
        return False

    def is_mandatory(self):
        return True


class Translate2D(object):
    def __init__(self, shift=10, direction='right', roll=True):
        """
        :param shift (int): количество пикселей, на которое необходимо сдвинуть изображение
        :param direction (string): направление (['right', 'left', 'down', 'up'])
        :param roll (bool): Если False, не заполняем оставшуюся часть. Если True, заполняем оставшимся краем.
        (прим. False: [1, 2, 3]=>[0, 1, 2]; True: [1, 2, 3] => [3, 1, 2])
        """
        self.roll = roll
        self.shift = shift
        if direction in ['right', 'left']:
            if direction == 'left':
                self.shift = - self.shift
            self.call = self.horizontal
        elif direction in ['down', 'up']:
            if direction == 'down':
                self.shift = - self.shift
            self.call = self.vertical

    def is_train_only(self):
        return True

    def is_mandatory(self):
        return False

    def vertical(self, image):
        res = np.roll(image, self.shift, axis=0)
        if not self.roll:
            if self.shift > 0:
                res[0:self.shift, :] = 0
            else:
                res[res.shape[0] + self.shift, :] = 0
        return res

    def horizontal(self, image):
        res = np.roll(image, self.shift, axis=1)
        if not self.roll:
            if self.shift > 0:
                res[:, 0:self.shift] = 0
            else:
                res[:, res.shape[1] + self.shift] = 0
        return res


class Translate3D(object):
    def __init__(self, shift=10, direction='right', roll=True):
        """
        :param shift (int): количество пикселей, на которое необходимо сдвинуть изображение
        :param direction (string): направление (['right', 'left', 'down', 'up'])
        :param roll (bool): Если False, не заполняем оставшуюся часть. Если True, заполняем оставшимся краем.
        (прим. False: [1, 2, 3]=>[0, 1, 2]; True: [1, 2, 3] => [3, 1, 2])
        """
        self.translate2d = Translate2D(shift,direction,roll)

    def is_train_only(self):
        return True

    def is_mandatory(self):
        return False

    def call(self, image):
        res = np.zeros((image.shape[0], image.shape[1], 3))
        for channel in range(image.shape[2]):
            res[:, :, channel] = self.translate2d.call(image[:, :, channel])
        return res


class RandomCrop(object):
    def __init__(self, crop_size):
        """
        :param crop_size (int or tuple): размер вырезанного изображения.
        """
        if isinstance(crop_size, int):
            self.out_shape = (crop_size, crop_size)
        else:
            self.out_shape = crop_size

    def is_train_only(self):
        return False

    def is_mandatory(self):
        return False

    def call(self, image):
        h = np.random.randint(0, image.shape[0] - self.out_shape[0])
        w = np.random.randint(0, image.shape[1] - self.out_shape[1])
        return image[h:h+self.out_shape[0], w:w+self.out_shape[1]]


class CenterCrop(object):
    def __init__(self, crop_size):
        """
        :param crop_size (int or tuple): размер вырезанного изображения (вырезать по центру).
        """
        if isinstance(crop_size, int):
            self.out_shape = (crop_size, crop_size)
        else:
            self.out_shape = crop_size

    def is_train_only(self):
        return False

    def is_mandatory(self):
        return True

    def call(self, image):
        h = (image.shape[0] - self.out_shape[0]) // 2
        w = (image.shape[1] - self.out_shape[1]) // 2
        return image[h:h+self.out_shape[0], w:w+self.out_shape[1]]


class Scale(object):
    def __init__(self, image_size, scale):
        """
        :param image_size (int): размер вырезанного изображения (по центру).
        :param scale (float): во сколько раз увеличить изображение.
        """
        if isinstance(image_size, int):
            self.out_shape = (image_size, image_size)
        else:
            self.out_shape = image_size

        self.scale = scale
        self.crop_obj = CenterCrop(self.out_shape)

    def is_train_only(self):
        return False

    def is_mandatory(self):
        return True

    def call(self, img):
        img = cv2.resize(img, (int(img.shape[1] * self.scale), int(img.shape[0] * self.scale)))
        return self.crop_obj.call(img)


class RandomRotateImage(object):
    def __init__(self, min_angle, max_angle):
        """
        :param min_angle (int): минимальный угол поворота.
        :param max_angle (int): максимальный угол поворота.
        Угол поворота должен быть выбран равномерно из заданного промежутка.
        """
        self.min_angle = min_angle
        self.max_angle = max_angle

    def is_train_only(self):
        return True

    def is_mandatory(self):
        return False

    def call(self, image):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        angle = int(np.random.rand() * (self.max_angle - self.min_angle)) + self.min_angle
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        return cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)


class GaussianNoise2D(object):
    def __init__(self, mean=0, sigma=0.03):
        """
        :param mean (int): среднее значение.
        :param sigma (int): максимальное значение ско. Итоговое значение должно быть выбрано равномерно в промежутке
        [0, sigma].
        :param by_channel (bool): если True, то по каналам независимо. Не понятно как именно при False
        """
        self.mean = mean
        self.sigma = sigma

    def is_train_only(self):
        return True

    def is_mandatory(self):
        return False

    def call(self, image):
        #image = image.astype(np.int16)
        image = (image * np.random.normal(self.mean, np.random.rand() / self.sigma, image.shape) + 1).astype(np.uint8)
        return image


class GaussianNoise3D(object):
    def __init__(self, mean=0, sigma=0.03, by_channel=False):
        """
        :param mean (int): среднее значение.
        :param sigma (int): максимальное значение ско. Итоговое значение должно быть выбрано равномерно в промежутке
        [0, sigma].
        :param by_channel (bool): если True, то по каналам независимо. Не понятно как именно при False
        """
        self.mean = mean
        self.sigma = sigma
        self.by_channel = by_channel

    def is_train_only(self):
        return True

    def is_mandatory(self):
        return False

    def call(self, image):
        image = image.astype(np.int16)
        if self.by_channel:
            for channel in range(image.shape[2]):
                image[:, :, channel] = (image[:, :, channel] *
                                        (np.random.normal(self.mean, np.random.rand() * self.sigma, image.shape) + 1)).astype(np.int16)
        else:
            noise = np.random.normal(self.mean, np.random.rand() * self.sigma, (image.shape[0],image.shape[1])) + 1
            for channel in range(image.shape[2]):
                image[:, :, channel] = (image[:, :, channel] * noise).astype(np.int16)
        image[image > 255] = 255
        image[image < 0] = 0
        return image.astype(np.uint8)


class Salt(object):
    def __init__(self, prob, by_channel=False):
        """
        :param prob (float): вероятность, с которой пиксели будут заполнены белым.
        :param by_channel (bool): если True, то по каналам независимо.
        """
        self.prob = prob
        self.by_channel = by_channel

    def is_train_only(self):
        return True

    def is_mandatory(self):
        return False

    def call(self, image):
        if len(image.shape) == 2:
            white = 255
            return np.where(np.random.rand(image.shape[0], image.shape[1]) < self.prob, white, image).astype(np.uint8)
        else:
            if self.by_channel:
                white = np.ones((image.shape[0], image.shape[1])) * 255
                for channel in range(image.shape[2]):
                    image[:, :, channel] = np.where(np.random.rand(image.shape[0], image.shape[1]) < self.prob,
                                                  white, image[:,:,channel]).astype(np.uint8)
            else:
                white = np.ones((image.shape[0], image.shape[1])) * 255
                noise = np.random.rand(image.shape[0], image.shape[1])
                for channel in range(image.shape[2]):
                    image[:, :, channel] = np.where(noise < self.prob,
                                                    white, image[:, :, channel]).astype(np.uint8)
            return image


class Pepper(object):
    def __init__(self, prob, by_channel=False):
        """
        :param prob (float): вероятность, с которой пиксели будут заполнены черным.
        :param by_channel (bool): если True, то по каналам независимо.
        """
        self.prob = prob
        self.by_channel = by_channel

    def is_train_only(self):
        return True

    def is_mandatory(self):
        return False

    def call(self, image):
        if len(image.shape) == 2:
            black = 0
            return np.where(np.random.rand(image.shape[0], image.shape[1]) < self.prob, black, image).astype(np.uint8)
        else:
            black = 0
            if self.by_channel:
                for channel in range(image.shape[2]):
                    image[:, :, channel] = np.where(np.random.rand(image.shape[0], image.shape[1]) < self.prob,
                                                    black, image[:, :, channel]).astype(np.uint8)
            else:
                black = np.zeros((image.shape[0], image.shape[1]))
                noise = np.random.rand(image.shape[0], image.shape[1])
                for channel in range(image.shape[2]):
                    image[:, :, channel] = np.where(noise < self.prob,
                                                    black, image[:, :, channel]).astype(np.uint8)
            return image


class ChangeChannel(object):
    def __init__(self, value, channel_number=0):
        """
        :param value (int): насколько изменить яркость. hue, contrast, saturation.
        ЭТО И ЕСТЬ ИЗМЕНЕНИЕ ТОГО КАНАЛА, КОТОРЫЙ НУЖЕН
        """
        self.value = value
        self.channel = channel_number

    def is_train_only(self):
        return False

    def is_mandatory(self):
        return False

    def call(self, image):
        if len(image.shape) == 3:
            channel = image[:, :, self.channel]
            if self.value > 0:
                channel[channel > 255 - self.value] = 255 - self.value
            else:
                channel[channel < - self.value] = -self.value
            image[:, :, self.channel] = channel + self.value
        else:
            if self.value > 0:
                image[image > 255 - self.value] = 255 - self.value
            else:
                image[image < - self.value] = -self.value
            image = image + self.value
        return image


class GaussianBlur(object):
    def __init__(self, ksize=(5, 5)):
        """
        :param ksize (tuple): размер фильтра.
        """
        self.ksize = ksize
        self.std = 1

    def is_train_only(self):
        return False

    def is_mandatory(self):
        return False

    def call(self, image):
        return cv2.GaussianBlur(image, self.ksize, self.std)


class Normalize(object):
    def __init__(self, mean=128):
        """
        :param mean (int or tuple): среднее значение (пикселя), которое необходимо вычесть.
        """
        self.mean = mean

    def is_train_only(self):
        return False

    def is_mandatory(self):
        return True

    def call(self, image):
        return (image.astype(np.float32) - 128) / 255