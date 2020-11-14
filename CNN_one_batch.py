from CNN import CNN, loss_func
from config import *
from dataloader_MNIST import DataLoader
from optimizers import SGD, Momentum


def one_batch_gen(batch, labels, iter):
    for _ in range(iter):
        yield batch, labels


def one_batch_validate(batch, labels):
    yield batch, labels


if __name__=="__main__":
    dl_train = DataLoader(
        cfg.train_data_path,
        cfg.nrof_classes,
        cfg.dataset_type,
        cfg.shuffle,
        64,
        [AdaptiveNormalize()],
        [1],
        cfg.sample_type,
        cfg.train_labels_path,
        cfg.epoch_size,
        cfg.probabilities
    )


    cnn = CNN()

    #optimizer = Momentum(cnn, 0.0005, loss_func, label_smoothing=0, gamma=0.8)
    optimizer = SGD(cnn, 0.01, loss_func, label_smoothing=0)


    overfit_batch, labels = dl_train.batch_generator().__next__()
    # dl_train.show_batch()
    iteration = 0
    error_rate, cross_entripy = [], []
    train_accuracy, test_accuracy = [], []

    loss, hit, count = cnn.validate(one_batch_validate(overfit_batch, labels))
    train_accuracy.append(hit.sum() / count.sum())
    print(f"\repoch: {iteration} batch_acc: {train_accuracy[-1]} loss: {loss}")
    try:
        accuracy = 0
        accuracy_max = 0
        while accuracy < 1:
            iteration += 1

            err_rate, entropy = cnn.train(one_batch_gen(overfit_batch, labels, 2), optimizer)
            error_rate += err_rate
            cross_entripy += entropy
            loss, hit, count = cnn.validate(one_batch_validate(overfit_batch, labels))
            train_accuracy.append(hit.sum() / count.sum())
            accuracy = hit.sum() / count.sum()
            print(f"\repoch: {iteration} batch_acc: {train_accuracy[-1]} loss: {loss}")
    except Exception as e:
        print(f"Exception {e}")
        raise e
    finally:
        plt.figure(figsize=(14, 5))
        plt.subplot(1, 3, 1)
        plt.title('Error rate')
        plt.plot(error_rate)
        plt.subplot(1, 3, 2)
        plt.title('Entropy')
        plt.plot(cross_entripy)
        print(cross_entripy)
        plt.subplot(1, 3, 3)
        plt.title('Accuracy:')
        plt.plot(train_accuracy, label='train')
        plt.xlabel('epoch')
        plt.legend()
        plt.show()