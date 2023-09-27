import os
import multiprocessing
from utils.utils import get_device, set_seed

device = get_device()

# use seed to ensure reproducibility
set_seed(1337)

classes = [
    "Lung Opacity",
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Diaphragmatic Disfunction",
    "Emphysema",
    "Fracture",
    "Hilar/Mediastinal Disease",
    "Interstitial Disease",
    "Lung Lesion",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumothorax",
    "Cifo-scoliosis",
    "Support Devices",
    "Tuberculosis",
]

ood_gap_ptxd = ["Pneumothorax", "Support Devices"]
top = [
    "Pneumothorax",
    "Support Devices",
    "Emphysema",
    "Interstitial Disease",
    "Tuberculosis",
]

ROOT = "/home/xvision/alex/"

config = {
    "metadata_path": ROOT + "domain_generalization/exp1_final.csv",
    "train_data_path": ROOT + "xray_data/xray_labeling_npy_256/",
    "valid_data_path": ROOT + "xray_data/xray_labeling_npy_256/",
    "imagenet_path": ROOT + "domain_generalization/mini_imagenet_train/",
    "checkpoint_path": None,

    "image_size": (256, 256),
    "batch_size": 32,
    "num_workers": multiprocessing.cpu_count(),
    "num_epochs": 45,
    
    "lr": 1e-4,
    "patience": 5,
    "wc": 1e-4,
    "SEED": 1337,

    "augmentation": True,
    "only_test": False,
    
    "split_type": "split_ID",
    "strategy": "LISA",
    "LISA_psel": 0.5,
    "topk_LISA": ood_gap_ptxd,
}
