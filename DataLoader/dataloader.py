# numpy, cv2, scipy
# mist , nist, cfar10
# config file как у фирюзы (easy)

class DataLoader:
    def __init__(self, train_data,  shiffle, batch_size, epoch_size=None, type=['train', 'test']):
        pass

    def readData(self):
        pass

    def print_statistic(self):
        # name
        # kol-vo viborki
        # kol-vo klassov
        # kolvo kajdogo klassa
        pass

    def batch_generator(self):
        pass

    #сэмплирование
    # в одной эпохе встречалось 1 раз встречался 1 класс, (вектор какой класс с какой вероятностью встречается в эпохе)

    def preprocess(self, normalize, size):
        pass

    def one_hot_encoding(self, labels):
        pass

    def data_augmentation(self):
        # каждый метод - отдельный класс, после препроцессинга
        # * crop
        # - pixel shift
        # * rotate
        # * scale
        # * noise (white, выбираем среднее от 0 до max-std,)
        # * brightness
        # * hue
        # * saturation
        # * pad
        # * blur
        # * flip
        pass

    def visualize_batch(self):
        pass

    #не забыть main.py

    def normalize(self):
        # (x-128)/255
        #(x - p)/255
        pass


'''
test:   
    уменьшаем до 256 и вырезаем, аугментируем
    224x224  

train:
    
'''