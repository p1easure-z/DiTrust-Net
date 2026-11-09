from .dataset_loader import *

datasets = {
    'brats': BraTSDatasetLoader,
}

def get_segmentation_dataset(name, **kwargs):
    if name.lower() not in datasets:
        raise ValueError(f"Dataset '{name}' not found. Available: {list(datasets.keys())}")
    return datasets[name.lower()](**kwargs)