import cv2
import PIL
import torch
import numpy as np
import PIL.Image as Image
import albumentations as A
import torchvision.transforms as transforms
from _collections import OrderedDict
import data_proc.eye_preprocess.fundus_prep as prep


class RemoveEyeBackground:
    def __init__(self, size=None):
        self.size = size

    def __call__(self, image):
        image = np.array(image)
        image, borders, mask, _, _ = prep.process_without_gb(image)
        if self.size is not None:
            image = cv2.resize(image, (self.size, self.size))

        image = PIL.Image.fromarray(image)
        return image


class RandomMaskOut(torch.nn.Module):
    def __init__(self, max_holes=8, max_height=8, max_width=8, min_holes=None, min_height=None,
                 min_width=None, fill_value=0, mask_fill_value=None, always_apply=False, p=0.5):
        super().__init__()
        self.transform = A.CoarseDropout(max_holes, max_height, max_width, min_holes, min_height, min_width,
                                         fill_value, mask_fill_value, always_apply, p)

    def __call__(self, image):
        image = np.array(image)
        image = self.transform(image=image)['image']
        image = PIL.Image.fromarray(image)

        return image


class RandomGaussianBlur(torch.nn.Module):
    def __init__(self, blur_limit=(3, 7), sigma_limit=0.5, always_apply=False, p=0.5):
        super().__init__()
        self.transform = A.GaussianBlur(blur_limit, sigma_limit, always_apply, p)

    def __call__(self, image):
        image = np.array(image)
        image = self.transform(image=image)['image']
        image = PIL.Image.fromarray(image)

        return image


class RandomCLAHE(torch.nn.Module):
    def __init__(self, clipLimit=2.0, tileGridSize=(8, 8), p=0.5):
        super().__init__()
        self.clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
        self.p = p
    def __call__(self, image):
        img = np.array(image)
        img_1 = self.clahe.apply(img[:, :, 0]) * 1
        img_2 = self.clahe.apply(img[:, :, 1]) * 1
        img_3 = self.clahe.apply(img[:, :, 2]) * 1
        if np.random.rand() < self.p:
            image = cv2.merge([img_1, img_2, img_3]).astype(np.uint8)
        else:
            image = img
        image = PIL.Image.fromarray(image)

        return image


class RandomGaussianNoise(torch.nn.Module):
    def __init__(self, var_limit=(10.0, 30.0), mean=0, per_channel=True, always_apply=False, p=0.5):
        super().__init__()
        self.transform = A.GaussNoise(var_limit, mean, per_channel, always_apply, p)

    def __call__(self, image):
        image = np.array(image)
        image = self.transform(image=image)['image']
        image = PIL.Image.fromarray(image)

        return image


class Augmentations:
    def __init__(self, size=224):

        transform_list = []

        # random resize and crop
        transform_list.append(transforms.Resize((size, size)))

        # random flip
        transform_list.append(transforms.RandomHorizontalFlip())
        transform_list.append(transforms.RandomVerticalFlip())

        transform_list.append(RandomGaussianNoise(p=0.3))

        transform_list.append(RandomCLAHE(p=0.5))

        # random color trans
        transform_list.append(transforms.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2),
                                                     saturation=(0.8, 1.2),
                                                     hue=(-0.03, 0.03))
                              )

        transform_list.append(RandomMaskOut(p=0.3))
        #transform_list.append(RandomGaussianBlur(p=0.3))

        # normalize to tensor
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))

        self.transform = transforms.Compose(transform_list)  # output: T, H, W, C

    def __call__(self, img):
        img = self.transform(img)
        return img
    #     self.ta_transform = self._get_crop_transform(tta)
    #
    # def _get_crop_transform(self, method='ten'):
    #
    #     if method == 'ten':
    #         crop_tf = transforms.Compose([
    #             transforms.Resize((self.size + 32, self.size + 32)),
    #             transforms.TenCrop((self.size, self.size))
    #         ])
    #
    #     if method == 'inception':
    #         crop_tf = InceptionCrop(
    #             self.size,
    #             resizes=tuple(range(self.size + 32, self.size + 129, 32))
    #         )
    #
    #     after_crop = transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize(self.mean, self.std),
    #     ])
    #     return transforms.Compose([
    #         crop_tf,
    #         transforms.Lambda(
    #             lambda crops: torch.stack(
    #                 [after_crop(crop) for crop in crops]))
    #     ])


if __name__ == "__main__":
    img = Image.open('/media/zyi/080c2d74-aa6d-4851-acdc-c9588854e17c/Acemid/data_proc/00a1b74a-6bd7-5408-ab71-d1aab685bb90.png')

    augmentor = Augmentations()
    imt = augmentor(img)
    imt = transforms.ToPILImage()(imt)
    imt.show()
