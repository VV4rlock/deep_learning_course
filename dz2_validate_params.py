from config import *
from models import MLP
from activation_functions import ReLU, Hyperbolic, Sigmoid, Linear
from optimizers import SGD, Momentum
from dataloader_MNIST import DataLoader

def validate_MLP():


    res = []
    learning_rate = 0.1
    for index, param in enumerate([2, 4, 8, 16, 32, 64, 128]):
        print("testing {}".format(str(param)))
        dl_train = DataLoader(
            cfg.train_data_path,
            cfg.nrof_classes,
            cfg.dataset_type,
            cfg.shuffle,
            param,
            [AdaptiveNormalize()],
            [1],
            'balanced',
            cfg.train_labels_path,
            16 * 200 // param,
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

        model = MLP(28*28, [200], ReLU(), 10, 'classification')
        loss_func = lambda res, target: -np.log(res[target.argmax()])
        optimizer = Momentum(model, learning_rate, loss_func, label_smoothing=0, gamma=0)

        error_rate, cross_entripy = [], []
        train_accuracy, test_accuracy = [], []
        for epoch in range(20):

            err_rate, entropy = model.train(dl_train.batch_generator(), optimizer)
            error_rate += err_rate
            cross_entripy += entropy
            loss, hit, count = model.validate(dl_train.get_full_generator(count=10000))
            accuracy = hit.sum()/count.sum()

            train_accuracy.append(accuracy)
            loss, hit, count = model.validate(dl_test.get_full_generator())
            test_accuracy.append(hit.sum()/count.sum())
            print(f"\rparam-{param} epoch: {epoch} train_acc: {train_accuracy[-1]} test_acc: {test_accuracy[-1]}")
        test_params_string = "batchsize={}"
        res.append((train_accuracy, test_params_string.format(str(param)), test_accuracy))

    #plt.figure(figsize=(14, 5))
    plt.title('Batch_size\n(Momentum, lr=0.1, (3200 samples in epoch), ReLU, 200 hidden neurons)\nsolid line - test, dashed line - train')
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    colors='mrgbkcy'
    for i, r in enumerate(res):
        plt.plot(range(len(r[0])), r[0], '--{}'.format(colors[i]), label=r[1])
        plt.plot(range(len(r[0])), r[2], '-{}'.format(colors[i]))
    plt.legend()
    plt.show()

if __name__ == "__main__":
    validate_MLP()