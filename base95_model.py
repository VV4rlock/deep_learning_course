from config import *
from models import MLP
from activation_functions import ReLU
from optimizers import SGD, Momentum
from dataloader_MNIST import DataLoader

def train_MLP():
    dl_train = DataLoader(
        cfg.train_data_path,
        cfg.nrof_classes,
        'train',
        cfg.shuffle,
        16, # чем меньше батч, тем лучше
        [AdaptiveNormalize()],
        [1],
        'default',
        cfg.train_labels_path,
        None,
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
    print(f"model_param_count: {model.get_nrof_trainable_params()}")
    optimizer = SGD(model, 0.01, loss_func)

    iteration = 0
    error_rate, cross_entripy = [], []
    train_accuracy, test_accuracy = [], []

    try:
        accuracy = 0
        accuracy_max = 0
        while accuracy < 0.99:
            iteration += 1

            err_rate, entropy = model.train(dl_train.batch_generator(), optimizer)
            error_rate += err_rate
            cross_entripy += entropy
            loss, hit, count = model.validate(dl_train.get_full_generator(count=10000))
            train_accuracy.append(hit.sum()/count.sum())
            loss, hit, count = model.validate(dl_test.get_full_generator())
            accuracy = hit.sum()/count.sum()

            if accuracy > accuracy_max:
                #model.dump_model('best_test_model_Momentum.pickle')
                accuracy_max = accuracy
            test_accuracy.append(accuracy)
            print(f"\repoch: {iteration} train_acc: {train_accuracy[-1]} test_acc: {test_accuracy[-1]} best: {accuracy_max}")
    except Exception as e:
        print(f"Exception {e}")
    finally:
        plt.title('Accuracy:')
        plt.plot(test_accuracy, label='test')
        plt.plot(train_accuracy, label='train')
        plt.xlabel('epoch')
        plt.legend()
        plt.show()


if __name__ == "__main__":
    train_MLP()