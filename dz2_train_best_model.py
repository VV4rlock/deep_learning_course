from config import *
from models import MLP
from activation_functions import ReLU
from optimizers import SGD,Momentum
from dataloader_MNIST import DataLoader

def train_MLP():
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
        "MNIST/t10k-images-idx3-ubyte",
        cfg.nrof_classes,
        'test',
        cfg.shuffle,
        cfg.batch_size,
        cfg.transforms,
        cfg.transform_probs,
        cfg.sample_type,
        "MNIST/t10k-labels-idx1-ubyte",
        cfg.epoch_size,
        cfg.probabilities
    )
    dl_train.show_statistics()

    #for batch in dl_train.batch_generator():
    #     #dl_train.show_batch()
    #    img, label = batch[0]
    #   show_image(img)
    #   exit(0)

    model = MLP(28*28, [200], ReLU(), 10, 'classification')
    #model = MLP.load_model('test_model.pickle')

    loss_func = lambda res, target: -np.log(res[target.argmax()])
    print(f"model_param_count: {model.get_nrof_trainable_params()}")
    optimizer = Momentum(model, 0.05, loss_func, label_smoothing=0.2, gamma=0.8)
    #SGD 0.000051
    #optimizer = Momentum(model, 0.0005, loss_func, label_smoothing=0)
    def one_batch_gen(batch, iter):
        for _ in range(iter):
            yield batch
    def one_batch_validate(batch):
        for i in batch:
            yield i

    overfit_batch = dl_train.batch_generator().__next__()
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

            err_rate, entropy = model.train(dl_train.batch_generator(), optimizer)
            error_rate += err_rate
            cross_entripy += entropy
            loss, hit, count = model.validate(dl_train.get_full_generator(count=10000))
            train_accuracy.append(hit.sum()/count.sum())
            loss, hit, count = model.validate(dl_test.get_full_generator())
            accuracy = hit.sum()/count.sum()
            if iteration > window_len:
                 var= np.array(test_accuracy[-window_len:])
                 if train_accuracy[-2] - train_accuracy[-1] > 1:
                     optimizer.set_learning_rate(optimizer.get_learning_rate() / 2)
                 elif train_accuracy[-3] == train_accuracy[-1]:
                     optimizer.set_learning_rate(optimizer.get_learning_rate() * 3)
                 elif (train_accuracy[-2] - train_accuracy[-1]) > 3 * (var.max() - var.min()):
                     optimizer.set_learning_rate(optimizer.get_learning_rate() / 2)
                     window_len *= 2

            if accuracy > accuracy_max:
                model.dump_model('best_test_model_Momentum.pickle')
                accuracy_max = accuracy
            test_accuracy.append(accuracy)
            print(f"\repoch: {iteration} train_acc: {train_accuracy[-1]} test_acc: {test_accuracy[-1]} best: {accuracy_max} learning_rate: {optimizer.get_learning_rate()}")
    except Exception as e:
        print(f"Exception {e}")
    finally:
        plt.figure(figsize=(14, 5))
        plt.subplot(1, 3, 1)
        plt.title('Error rate')
        plt.plot(error_rate)
        plt.subplot(1, 3, 2)
        plt.title('Entropy')
        plt.plot(cross_entripy)
        plt.subplot(1, 3, 3)
        plt.title('Accuracy:')
        plt.plot(test_accuracy, label='test')
        plt.plot(train_accuracy, label='train')
        plt.xlabel('epoch')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    train_MLP()

