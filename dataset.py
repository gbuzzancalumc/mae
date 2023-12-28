import typing

import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from custom_transforms import Transform


class WSIDataset(Dataset):
    def __init__(self, h5_files: typing.List[str], full_dataset_epochs: bool, imgs_per_epoch: int = 0):
        self.h5_files = h5_files
        self.n_files = len(self.h5_files)
        print(f"In total {self.n_files} files will be used for training")

        if full_dataset_epochs:
            print("Since working in full dataset mode, now reading the h5 files for info...")
            self.mode = "full"
            self.imgs = []
            for file in self.h5_files:
                try:
                    with h5py.File(file, "r") as f:
                        imgs_number = f["imgs"].shape[0]
                except Exception as e:
                    print(f"File {file} is probably corrupted, not using it")
                    print("The error while trying to load it was: ", e)
                    self.h5_files.remove(file)
                    self.n_files -= 1
                    continue

                for i in range(imgs_number):
                    self.imgs.append((file, i))

            # Shuffle the images
            np.random.shuffle(self.imgs)

            self.imgs_per_epoch = len(self.imgs)
            print(f"Working in full dataset mode, {self.imgs_per_epoch} images will be used for each epoch")
        else:
            self.mode = "partial"
            self.imgs_per_epoch = imgs_per_epoch
            print(f"Working in partial dataset mode, {self.imgs_per_epoch} images will be used for each epoch")

        self.transform = Transform("./stain_norm_stats.yaml")

    def __len__(self):
        return self.imgs_per_epoch

    def __getitem__(self, idx):
        if self.mode == "full":
            if idx == self.imgs_per_epoch - 1:
                print("Epoch finished, shuffling the images")
                np.random.shuffle(self.imgs)
            file, img_idx = self.imgs[idx]
            try:
                with h5py.File(file, "r") as f:
                    img = Image.fromarray(f["imgs"][img_idx])
            except Exception as e:
                print(f"File {file} is probably corrupted, skipping it")
                print("The error was: ", e)
                return self.__getitem__(idx + 1)
        else:
            file_idx = np.random.randint(0, self.n_files)
            file = self.h5_files[file_idx]
            try:
                with h5py.File(file, "r") as f:
                    img_idx = np.random.randint(0, f["imgs"].shape[0])
                    img = Image.fromarray(f["imgs"][img_idx])
            except Exception as e:
                print(f"File {file} is probably corrupted, skipping it")
                print("The error was: ", e)
                return self.__getitem__(idx)

        return self.transform(img)
