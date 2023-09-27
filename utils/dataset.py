import torch
import os
import numpy as np
import pandas as pd
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms
from utils.config import classes
from utils.mixup import *
import albumentations as A


def get_augmentations(image_size, mode="train"):
    augs = [A.Resize(*image_size)]

    if mode == "train":
        augs += [
            A.RandomBrightnessContrast(),
            A.HorizontalFlip(),
            A.GaussianBlur(),
            A.ShiftScaleRotate(border_mode=0, value=0)
            # A.CoarseDropout(max_height=config['image_size'][0] // 16, max_width=config['image_size'][1] // 16)
            # A.ElasticTransform(border_mode=0, value=0, p=1, sigma=40, alpha=1024, alpha_affine=40)
            # A.PixelDropout(p=0.3)
            # A.GaussNoise(var_limit=0.001)
        ]

    return A.Compose(augs)


class XRayClassificationDataset(Dataset):
    def __init__(self, data_path, metadata_path, mode, augmentations, config):
        self.metadata = pd.read_csv(metadata_path, index_col=0)
        self.metadata = self.metadata[self.metadata[config["split_type"]] == mode]
        self.metadata["Abnormal"] = 1 - self.metadata["Normal"]
        self.mode = mode

        self.file_names = []
        self.remove_files = []
        for file_name in self.metadata.index.values:
            file_path = os.path.join(
                data_path, file_name.replace(".jpeg", "").replace(".dcm", "") + ".npy"
            )

            if os.path.exists(file_path):
                self.file_names.append(file_name)
            else:
                self.remove_files.append(file_name)

        if len(self.remove_files):
            print(f"Files not found: {self.remove_files}")

        self.metadata = self.metadata.drop(self.remove_files)

        self.data_path = data_path
        self.augmentations = augmentations
        self.normalize = transforms.Normalize(
            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        )
        self.imagenet_transforms = transforms.Compose(
            [transforms.ToTensor(), self.normalize]
        )

        if mode == "train":
            self.strategy = config["strategy"]

            if config["strategy"] == "LISA":
                self.ratio = config["LISA_psel"]
                self.topk_LISA_indices = [classes.index(c) for c in config["topk_LISA"]]

                # Intra-label candidates
                topk_targets = self.metadata[config["topk_LISA"]].values
                self.intra_label = {
                    tuple(target.astype(bool)): list(
                        np.where((topk_targets == target).all(-1))[0]
                    )
                    for target in np.unique(topk_targets, axis=0)
                }

                for k, v in self.intra_label.items():
                    print(f"Target: {k}, frequency: {len(v)}")

                # Intra-domain candidates
                all_domains = self.metadata["domain"].values
                self.intra_domain = {
                    domain: list(np.where(all_domains == domain)[0])
                    for domain in np.unique(all_domains)
                }
            elif config["strategy"] == "CrossDomain":
                # Cross-domain candidates
                all_domains = self.metadata["domain"].values
                self.cross_domain = {
                    domain: list(np.where(all_domains != domain)[0])
                    for domain in np.unique(all_domains)
                }
            elif config["strategy"] == "StyleDensenet":
                self.style_imgs_fpaths = np.array(
                    [
                        config["imagenet_path"] + x
                        for x in os.listdir(config["imagenet_path"])
                    ]
                )
        else:
            self.strategy = None

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = (
            self.file_names[idx].replace(".jpeg", "").replace(".dcm", "") + ".npy"
        )
        file_path = os.path.join(self.data_path, file_name)

        img = (np.load(file_path) / 2**16).astype(np.float32)
        img = self.augmentations(image=img)["image"]
        img = np.repeat(img[np.newaxis, ...], 3, axis=0)
        img = self.normalize(torch.from_numpy(img))

        target = self.metadata.loc[self.file_names[idx], classes]
        target = torch.from_numpy(target.values.astype(np.float32))

        domain = self.metadata.loc[self.file_names[idx], "domain"]

        if self.strategy == "StyleDensenet" and self.mode == "train":
            img_no_style_transfer = img.clone()
            x = img.clone()
            y = self._get_random_style_image()

            m1 = torch.mean(x, dim=[1, 2], keepdim=True)
            v1 = torch.var(x, dim=[1, 2], keepdim=True)
            x = (x - m1) / (v1 + 1e-7).sqrt()

            m2 = torch.mean(y, dim=[1, 2], keepdim=True)
            v2 = torch.var(y, dim=[1, 2], keepdim=True)
            y = (y - m2) / (v2 + 1e-7).sqrt()

            mf = m2
            vf = v2
            x = x * (vf + 1e-7).sqrt() + mf
            img = x

            return img, img_no_style_transfer, target, domain, idx

        return img, target, domain, idx

    def LISA(self, img1, target1, domain1, idx1):
        """ """
        do_intra_label = np.random.rand() < self.ratio
        img2, target2 = [], []
        for i in range(img1.shape[0]):
            label = tuple(target1[i][self.topk_LISA_indices].numpy().astype(bool))
            domain = domain1[i]

            if do_intra_label:
                idx2 = np.random.choice(self.intra_label[label])
            else:
                idx2 = np.random.choice(self.intra_domain[domain])

            img, target, _, _ = self[idx2]
            img2.append(img)
            target2.append(target)

        img2 = torch.stack(img2)
        target2 = torch.stack(target2)

        img, target = mix_up(img1, target1, img2, target2)

        return img, target

    def CrossDomain(self, img1, target1, domain1, idx1):
        """ """
        img2, target2 = [], []
        for i in range(img1.shape[0]):
            idx2 = np.random.choice(self.cross_domain[domain1[i]])
            img, target, _, _ = self[idx2]
            img2.append(img)
            target2.append(target)

        img2 = torch.stack(img2)
        target2 = torch.stack(target2)

        img, target = mix_up(img1, target1, img2, target2)

        return img, target

    def _get_random_style_image(self):
        random_index = np.random.choice(np.arange(self.style_imgs_fpaths.shape[0]))
        image = Image.open(self.style_imgs_fpaths[random_index]).convert("RGB")

        image = self.imagenet_transforms(image)

        return image
