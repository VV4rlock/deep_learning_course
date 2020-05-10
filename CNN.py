from base import BaseModel, BaseActivationFunction
from layers import InputLayer, SoftmaxOutputLayer, FullyConnectedLayer, SimpleOutputLayer,ConvLayer, MaxPooling, AveragePooling, ReshapeLayer
from activation_functions import ReLU, Linear
from config import *
from optimizers import SGD,Momentum
from dataloader_MNIST import DataLoader


class CNN(BaseModel):
    def __init__(self):
        inp = ReshapeLayer((28, 28), (1, 28, 28))
        conv1 = ConvLayer((1, 28, 28), 16, 5, 5, ReLU(), pad=2, stride=1, prev_layer=inp)
        pool1 = MaxPooling((16, 28, 28), 2, 2, stride=2, prev_layer=conv1)
        conv2 = ConvLayer((16, 14, 14), 32, 5, 5, ReLU(), pad=2, stride=1, prev_layer=pool1)
        pool2 = MaxPooling((32, 14, 14), 2, 2, stride=2, prev_layer=conv2)
        reshape = ReshapeLayer((32, 7, 7), (7*7*32,), prev_layer=pool2)
        fc1 = FullyConnectedLayer(7*7*32, 120, ReLU(), prev_layer=reshape)
        fc2 = FullyConnectedLayer(120, 10, Linear(), prev_layer=fc1)
        out = SoftmaxOutputLayer(fc2)

        self.layers = [inp, conv1, pool1, conv2, pool2, reshape, fc1, fc2, out]
        self.out = out
        self.nrof_output = 10
        self.param_count = 0
        for i,l in enumerate(self.layers):
            print(f"Layer {i} have {l.get_nrof_trainable_params()} trainable params")
            self.param_count += l.get_nrof_trainable_params()

    def get_nrof_trainable_params(self):
        return self.param_count

    def __call__(self, x):
        for l in self.layers:
            x = l(x)
        return x

    def validate(self, generator):
        hit, count, loss = 0, 0, 0
        for input, one_hot_vector in generator:
            res = self(input)
            count += one_hot_vector.sum(axis=0)
            loss -= np.log((res * one_hot_vector).sum(axis=1)).sum()
            for i in range(input.shape[0]):
                if one_hot_vector[i].argmax() == res[i].argmax():
                    hit += one_hot_vector[i]
        return loss, hit, count


def loss_func(res, labels):
    return - np.log((res * labels).sum(axis=1)).sum()


if __name__=="__main__":
    dl_train = DataLoader(
        cfg.train_data_path,
        cfg.nrof_classes,
        cfg.dataset_type,
        cfg.shuffle,
        cfg.batch_size,
        cfg.transforms,
        cfg.transform_probs,
        cfg.sample_type,
        cfg.train_labels_path,
        cfg.epoch_size,
        cfg.probabilities
    )
    dl_test = DataLoader(
        cfg.test_data_path,
        cfg.nrof_classes,
        'test',
        cfg.shuffle,
        cfg.batch_size,
        cfg.transforms,
        cfg.transform_probs,
        cfg.sample_type,
        cfg.test_label_path,
        cfg.epoch_size,
        cfg.probabilities
    )
    dl_train.show_statistics()

    # for batch in dl_train.batch_generator():
    #     #dl_train.show_batch()
    #    img, label = batch[0]
    #   show_image(img)
    #   exit(0)

    model = CNN()
    # model = MLP.load_model('test_model.pickle')

    loss_func = lambda res, target: -np.log(res[target.argmax()])
    print(f"model_param_count: {model.get_nrof_trainable_params()}")
    #optimizer = Momentum(model, 0.05, loss_func, label_smoothing=0, gamma=0.8)
    optimizer = SGD(model, 0.005, loss_func, label_smoothing=0)

    iteration = 0
    error_rate, cross_entripy = [], []
    train_accuracy, test_accuracy = [], []
    try:
        accuracy = 0
        accuracy_max = 0
        acuracy_arr = []
        window_len = 5
        while accuracy < 0.99:
            iteration += 1
            print(f"{iteration} training...")
            err_rate, entropy = model.train(dl_train.batch_generator(), optimizer)
            error_rate += err_rate
            print(f"{iteration} train validation...")
            loss, hit, count = model.validate(dl_train.get_full_generator(count=1000))
            train_accuracy.append(hit.sum() / count.sum())
            cross_entripy.append(loss)
            print(f"{iteration} test validation...")
            loss, hit, count = model.validate(dl_test.get_full_generator(count=1000))
            accuracy = hit.sum() / count.sum()

            if accuracy > accuracy_max:
                model.dump_model('best_CNN_model_Momentum.pickle')
                accuracy_max = accuracy
            test_accuracy.append(accuracy)
            print(
                f"\repoch: {iteration} train_acc: {train_accuracy[-1]} test_acc: {test_accuracy[-1]} best: {accuracy_max}")
    except Exception as e:
        print(f"Exception {e}")
        raise e
    finally:
        plt.figure(figsize=(14, 5))
        plt.subplot(1, 2, 1)
        plt.title('Error rate')
        plt.plot(error_rate)
        plt.subplot(1, 2, 2)
        plt.title('Accuracy:')
        plt.plot(test_accuracy, label='test')
        plt.plot(train_accuracy, label='train')
        plt.xlabel('epoch')
        plt.legend()
        plt.show()