import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from data_proc.augmentation import Augmentations
from data_proc.augmentation import RemoveEyeBackground
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import convolve1d
from scipy.signal.windows import triang


class Retinal_Dataset(Dataset):
    def __init__(self, df, data_root=None, size=320, is_train=True, test_mode=False, train_val_split='patient',
                 age_norm=False, debug=False, reweight='none', lds=False, lds_kernel='gaussian', lds_ks=5, lds_sigma=2):
        """
        train_val_split: "simple" or "patient"

        """
        # self.paths = list(df.filename)

        # self.label = list(df.label)
        if debug:
            df = df[:1000]
        self.df = df
        self.data_root = data_root
        self.size = size
        self.is_train = is_train
        self.age_norm = age_norm
        self.norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # imagenet stats
        self.weights_dict = np.ones(100)
        self.weights = self._prepare_weights(reweight=reweight, lds=lds, lds_kernel=lds_kernel, lds_ks=lds_ks,
                                             lds_sigma=lds_sigma)
        self.test_mode = test_mode
        print('total samples number is: ', len(df.filename))

        if self.is_train:
            self.transform = Augmentations(size=size)

        else:
            self.transform = transforms.Compose([# RemoveEyeBackground(),
                                                 transforms.Resize((size, size)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                 ])

        if not test_mode:
            if train_val_split == 'simple':
                val_index = np.linspace(0, len(df.filename), len(df.filename) // 5, endpoint=False, dtype=np.int)
                train_index = np.setdiff1d(np.arange(len(df.filename)), val_index)

            elif train_val_split == 'patient':
                print('Split the training and validation by patient_id!!')
                patient_id = df.id.unique()
                print('total train patint: ', len(patient_id))
                val_id = np.linspace(0, len(patient_id), len(patient_id) // 10, endpoint=False, dtype=int)
                print('total train patint: ', len(patient_id), 'validation patint: ', len(val_id))

                for id in patient_id[val_id]:
                    df.loc[df.id == id, 'is_train'] = 0

                df = df.reset_index(drop=True)
                train_index = df.index[df.is_train == 1].array
                val_index = df.index[df.is_train == 0].array

                print('train ', len(train_index), 'val ', len(val_index))

            else:
                raise ValueError('train_val_split should be either  \'simple\' or \'patient\'!!!')

            if is_train:
                self.img_dirs = np.array(df.filename)[train_index]
                self.labels = np.array(df.label)[train_index]
                self.data_source = np.array(df.data_source)[train_index]
                print('training samples number is: ', len(self.img_dirs))
            else:
                self.img_dirs = np.array(df.filename)[val_index]
                self.labels = np.array(df.label)[val_index]
                self.data_source = np.array(df.data_source)[val_index]
                print('validation samples number is: ', len(self.img_dirs))
        else:
            self.img_dirs = np.array(df.filename)
            self.labels = np.array(df.label)
            self.data_source = np.array(df.data_source)

    def __len__(self):
        return len(self.img_dirs)

    def __getitem__(self, idx):
        print(len(self.weights), len(self.labels))
        # dealing with the image
        if self.data_root is not None:
            img_dir = os.path.join(self.data_root, self.img_dirs[idx])
        else:
            img_dir = self.img_dirs[idx]

        image = Image.open(img_dir).convert('RGB')
        image = self.transform(image)

        if self.age_norm:
            target = (torch.tensor(self.labels[idx], dtype=torch.float) - 30.) / 5.
        else:
            target = torch.tensor(self.labels[idx], dtype=torch.float) - 15 

        if self.test_mode:
            return image, target, self.img_dirs[idx], data_source
        else:
            return image, target, self.img_dirs[idx] #, data_source


    def show(self, idx):
        image, target, _, __ = self.__getitem__(idx)
        print(_)
        stds = np.array([0.229, 0.224, 0.225])
        means = np.array([0.485, 0.456, 0.406])
        img = ((image.numpy().transpose((1, 2, 0)) * stds + means) * 255).astype(np.uint8)
        plt.imshow(img)
        plt.title("The retinal age is {}!".format(target.item()))


if __name__ == '__main__':
    csv = pd.read_csv('/media/zyi/080c2d74-aa6d-4851-acdc-c9588854e17c/retinal_age_projects/mixed_data_final.csv')
    csv = csv[csv.is_train==0]
    data = Retinal_Dataset(df=csv, data_root='/media/zyi/litao/retinal_age_projects/',
                           is_train=True, test_mode=False, reweight='sqrt_inv', lds=True, debug=False)