from config import *
from dataloader import DataLoader


if __name__ == "__main__":
    dl = DataLoader(
        cfg.data_path,
        cfg.nrof_classes,
        cfg.dataset_type,
        cfg.shuffle,
        cfg.batch_size,
        cfg.transforms,
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