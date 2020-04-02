from config import *
from dataloader_MNIST import DataLoader
from models import MLP
from activation_functions import ReLU
from optimizers import SGD,Momentum

def one_batch_overfit():
    dl_train = DataLoader(
        cfg.train_data_path,
        cfg.nrof_classes,
        cfg.dataset_type,
        cfg.shuffle,
        100,
        [],
        [],
        cfg.sample_type,
        cfg.train_labels_path,
        cfg.epoch_size,
        cfg.probabilities
    )
    model = MLP(28 * 28, [200], ReLU(), 10, 'classification')
    loss_func = lambda res, target: -np.log(res[target.argmax()])
    print(f"model_param_count: {model.get_nrof_trainable_params()}")
    optimizer = Momentum(model, 0.000005, loss_func, label_smoothing=0, gamma=0.8)
    def one_batch_gen(batch, iter):
        for _ in range(iter):
            yield batch
    def one_batch_validate(batch):
        for i in batch:
            yield i
    overfit_batch = dl_train.batch_generator().__next__()
    #dl_train.show_batch()
    iteration = 0
    error_rate, cross_entripy = [], []
    train_accuracy, test_accuracy = [], []

    loss, hit, count = model.validate(one_batch_validate(overfit_batch))
    train_accuracy.append(hit.sum() / count.sum())
    try:
        accuracy = 0
        accuracy_max = 0
        while accuracy < 1:
            iteration += 1

            err_rate, entropy = model.train(one_batch_gen(overfit_batch, 15), optimizer)
            error_rate += err_rate
            cross_entripy += entropy
            loss, hit, count = model.validate(one_batch_validate(overfit_batch))
            train_accuracy.append(hit.sum() / count.sum())
            accuracy = hit.sum() / count.sum()
            print(f"\repoch: {iteration} batch_acc: {train_accuracy[-1]}")
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
        plt.plot(train_accuracy, label='train')
        plt.xlabel('epoch')
        plt.legend()
        plt.show()


if __name__ == "__main__":
    one_batch_overfit()