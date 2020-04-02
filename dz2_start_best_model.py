from config import *
from models import MLP
from activation_functions import ReLU
from optimizers import SGD,Momentum
from dataloader_MNIST import DataLoader


def validate_best_model():
    dl_test = DataLoader(
        "MNIST/t10k-images-idx3-ubyte",
        cfg.nrof_classes, #большинство параметров не важны, т к просто перебираем выборку
        'test',
        cfg.shuffle,
        cfg.batch_size,
        [AdaptiveNormalize()], # важная трансформация
        [1],
        cfg.sample_type,
        "MNIST/t10k-labels-idx1-ubyte",
        cfg.epoch_size,
        cfg.probabilities
    )
    dl_test.show_statistics()
    model = MLP.load_model('model_97850.pickle')
    loss, hit, count = model.validate(dl_test.get_full_generator())
    acc = hit.sum()/count.sum()
    print(f"BEST MODEL TRAIN RESULT: accuracy={acc} error_rate={(1-acc)*100}%\n"
          f"accuracy_by_classes:\n\t{hit/count}")


if __name__ == "__main__":
    validate_best_model()