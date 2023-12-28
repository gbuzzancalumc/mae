import random

from PIL import ImageFilter, Image
import torchvision.transforms as transforms
from pathlib import Path
import cv2 as cv
import numpy as np
from skimage import color
import yaml

COLOR_SPACES = ["LAB", "HSV", "HED"]


def convert_rgb_to_color_space(img_rgb: np.ndarray, color_space: str) -> np.ndarray:
    if color_space == "LAB":
        return cv.cvtColor(img_rgb, cv.COLOR_RGB2LAB)
    elif color_space == "HSV":
        return cv.cvtColor(img_rgb, cv.COLOR_RGB2HSV)
    elif color_space == "HED":
        return color.rgb2hed(img_rgb)
    else:
        raise ValueError(f"Color space {color_space} not supported")


def convert_color_space_to_rgb(img: np.ndarray, color_space: str) -> np.ndarray:
    if color_space == "LAB":
        return cv.cvtColor(img, cv.COLOR_LAB2RGB)
    elif color_space == "HSV":
        return cv.cvtColor(img, cv.COLOR_HSV2RGB)
    elif color_space == "HED":
        img = color.hed2rgb(img)
        imin = img.min()
        imax = img.max()

        return (255 * (img - imin) / (imax - imin)).astype("uint8")
    else:
        raise ValueError(f"Color space {color_space} not supported")


def get_image_stats(img: np.ndarray):
    stats = np.zeros((img.shape[2], 2))
    for channel_idx in range(img.shape[2]):
        stats[channel_idx, 0] = np.mean(img[:, :, channel_idx])
        stats[channel_idx, 1] = np.std(img[:, :, channel_idx])

    return stats


def normalize_image(img: np.ndarray, img_stats: np.ndarray, tar_stats: np.ndarray, color_space: str) -> np.ndarray:
    img = img.astype(np.float32)

    for channel_idx in range(img.shape[2]):
        img[:, :, channel_idx] = (((img[:, :, channel_idx] - img_stats[channel_idx, 0]) / img_stats[channel_idx, 1])
                                  * tar_stats[channel_idx, 1] + tar_stats[channel_idx, 0])

    if color_space == "LAB" or color_space == "HSV":
        img = np.clip(img, 0, 255).astype(np.uint8)

    return img


class RandStainNA(object):
    def __init__(self, dataset_statistics_filepath: str, p: float = 0.5):
        with open(dataset_statistics_filepath, "r") as f:
            self.dataset_statistics = yaml.load(f, Loader=yaml.FullLoader)

        self.p = p

    def augment(self, img: Image) -> Image:
        # convert to numpy
        img = np.array(img)

        # pick a random color space
        color_space = np.random.choice(COLOR_SPACES)
        img = convert_rgb_to_color_space(img, color_space)

        tar_stats = np.zeros((3, 2))

        for channel in range(3):
            channel_avg_mean = self.dataset_statistics[f"{color_space}_{channel}"]["avg"]["mean"]
            channel_avg_std = self.dataset_statistics[f"{color_space}_{channel}"]["avg"]["std"]
            channel_std_mean = self.dataset_statistics[f"{color_space}_{channel}"]["std"]["mean"]
            channel_std_std = self.dataset_statistics[f"{color_space}_{channel}"]["std"]["std"]

            channel_avg_distribution = self.dataset_statistics[f"{color_space}_{channel}"]["avg"]["distribution"]
            channel_avg_distribution = np.random.normal if channel_avg_distribution == "norm" else np.random.laplace

            channel_std_distribution = self.dataset_statistics[f"{color_space}_{channel}"]["std"]["distribution"]
            channel_std_distribution = np.random.normal if channel_std_distribution == "norm" else np.random.laplace

            tar_stats[channel, 0] = channel_avg_distribution(loc=channel_avg_mean, scale=channel_avg_std, )
            tar_stats[channel, 1] = channel_std_distribution(loc=channel_std_mean, scale=channel_std_std)

        img_stats = get_image_stats(img)
        img = normalize_image(img, img_stats, tar_stats, color_space)
        img = convert_color_space_to_rgb(img, color_space)

        return Image.fromarray(img)

    def __call__(self, img):
        if np.random.rand(1) < self.p:
            return self.augment(img)
        else:
            return img
        

class Transform:
    def __init__(self, rand_stain_yaml: str):
        self.transform = transforms.Compose([
            RandStainNA(rand_stain_yaml, p=0.25),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.25),
            transforms.RandomVerticalFlip(p=0.25),
            transforms.RandomRotation(degrees=(-20, 20)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        return self.transform(x)
