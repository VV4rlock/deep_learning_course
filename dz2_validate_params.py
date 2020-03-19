from config import *
from models import MLP
from activation_functions import ReLU, Hyperbolic, Sigmoid, Linear
from optimizers import SGD,Momentum
from dataloader_MNIST import DataLoader

def validate_MLP():


    test_params_string = "hidden-{}"
    res = []
    for index, param in enumerate([[200], [300], [100, 100], [200, 100]]):
        dl_train = DataLoader(
            cfg.data_path,
            cfg.nrof_classes,
            cfg.dataset_type,
            cfg.shuffle,
            cfg.batch_size,
            cfg.transforms,
            cfg.transform_probs,
            cfg.sample_type,
            cfg.labels_path,
            5,#cfg.epoch_size,
            cfg.probabilities
        )
        dl_test = DataLoader(
            cfg.TEST_PATH,
            cfg.nrof_classes,
            'test',
            cfg.shuffle,
            cfg.batch_size,
            cfg.transforms,
            cfg.transform_probs,
            cfg.sample_type,
            cfg.TEST_LABEL_PATH,
            cfg.epoch_size,
            cfg.probabilities
        )
        dl_train.show_statistics()

        model = MLP(28*28, param, ReLU(), 10, 'classification')
        loss_func = lambda res, target: -np.log(res[target.argmax()])
        optimizer = Momentum(model, 0.08, loss_func, label_smoothing=0, gamma=0)

        error_rate, cross_entripy = [], []
        train_accuracy, test_accuracy = [], []
        for epoch in range(15):

            err_rate, entropy = model.train(dl_train.batch_generator(), optimizer)
            error_rate += err_rate
            cross_entripy += entropy
            loss, hit, count = model.validate(dl_train.get_full_generator(count=10000))
            train_accuracy.append(hit.sum()/count.sum())
            #loss, hit, count = model.validate(dl_test.get_full_generator())
            accuracy = hit.sum()/count.sum()

            test_accuracy.append(accuracy)
            print(f"\rparam-{param} epoch: {epoch} train_acc: {train_accuracy[-1]} test_acc: {test_accuracy[-1]}")
        res.append((train_accuracy, test_params_string.format(str(param))))

    #plt.figure(figsize=(14, 5))
    plt.title('Hidden configures')
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    for r in res:
        plt.plot(r[0], label=r[1])
    plt.legend()
    plt.show()

if __name__ == "__main__":
    validate_MLP()