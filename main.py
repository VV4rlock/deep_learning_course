from config import *


def batch_generate_check():
    if cfg.DATASET == 'MNIST':
        from dataloader_MNIST import DataLoader
        dl = DataLoader(
            cfg.data_path,
            cfg.nrof_classes,
            cfg.dataset_type,
            cfg.shuffle,
            cfg.batch_size,
            cfg.transforms,
            cfg.transform_probs,
            cfg.sample_type,
            cfg.labels_path,
            cfg.epoch_size,
            cfg.probabilities
        )
        dl.show_statistics()
        generator = dl.batch_generator()
        generator.__next__()
        dl.show_batch()
        generator.__next__()
        dl.show_batch()
    else:
        from dataloader_CIFAR import DataLoader

        dl = DataLoader(
            cfg.data_path,
            cfg.nrof_classes,
            cfg.dataset_type,
            cfg.shuffle,
            cfg.batch_size,
            cfg.transforms,
            cfg.transform_probs,
            cfg.sample_type,
            cfg.labels_path,
            cfg.epoch_size,
            cfg.probabilities
        )
        dl.show_statistics()
        generator = dl.batch_generator()
        generator.__next__()
        dl.show_batch()
        generator.__next__()
        dl.show_batch()

def test_MLP():
    from dataloader_MNIST import DataLoader
    dl = DataLoader(
        cfg.data_path,
        cfg.nrof_classes,
        cfg.dataset_type,
        cfg.shuffle,
        cfg.batch_size,
        cfg.transforms,
        cfg.transform_probs,
        cfg.sample_type,
        cfg.labels_path,
        cfg.epoch_size,
        cfg.probabilities
    )
    dl.show_statistics()
    generator = dl.batch_generator()
    from models import MLP
    from activation_functions import ReLU
    model = MLP(28*28, [100], ReLU(), 10, 'classification')
    batch = generator.__next__()
    #print(batch)
    #dl.show_batch()
    res = model(batch[3][0].reshape(28*28))

    print(res)

if __name__ == "__main__":
    test_MLP()
