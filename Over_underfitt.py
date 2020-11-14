from config import *
from models import MLP
from activation_functions import ReLU, Hyperbolic, Sigmoid, Linear
from optimizers import SGD,Momentum
from dataloader_MNIST import DataLoader

def validate_MLP():


    test_params_string = "hidden-{}"
    res = []
    for index, param in enumerate([[800], [10]]):
        dl_train = DataLoader(
            cfg.data_path,
            cfg.nrof_classes,
            cfg.dataset_type,
            cfg.shuffle,
            4000,
            [],
            [],
            'balanced',
            cfg.labels_path,
            10,#cfg.epoch_size,
            cfg.probabilities
        )
        dl_test = DataLoader(
            cfg.TEST_PATH,
            cfg.nrof_classes,
            'test',
            cfg.shuffle,
            1000,
            [AdaptiveNormalize()],
            [1],
            cfg.sample_type,
            cfg.TEST_LABEL_PATH,
            cfg.epoch_size,
            cfg.probabilities
        )
        dl_train.show_statistics()

        model = MLP(28*28, param, ReLU(), 10, 'classification')
        loss_func = lambda res, target: -np.log(res[target.argmax()])
        optimizer = SGD(model, 0.5, loss_func, label_smoothing=0)

        error_rate, cross_entripy = [], []
        train_accuracy, test_accuracy = [], []
        for epoch in range(15):

            err_rate, entropy = model.train(dl_train.batch_generator(), optimizer)
            error_rate += err_rate
            cross_entripy += entropy
            loss, hit, count = model.validate(dl_train.get_full_generator(count=10000))
            train_accuracy.append(hit.sum()/count.sum())
            loss, hit, count = model.validate(dl_test.get_full_generator())
            accuracy = hit.sum()/count.sum()

            test_accuracy.append(accuracy)
            print(f"\rparam-{param} epoch: {epoch} train_acc: {train_accuracy[-1]} test_acc: {test_accuracy[-1]}")
        res.append((train_accuracy, "train-{}".format(str(param))))
        res.append((test_accuracy, "test-{}".format(str(param))))

    #plt.figure(figsize=(14, 5))
    plt.title('Overfit and underfit')
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    for r in res:
        plt.plot(r[0], label=r[1])
    plt.legend()
    plt.show()

if __name__ == "__main__":
    validate_MLP()