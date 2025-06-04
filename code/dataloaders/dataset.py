import os
import cv2
import torch
import random
import numpy as np
from glob import glob
from torch.utils.data import Dataset
from scipy.ndimage.interpolation import zoom
from torchvision import transforms
import itertools
from scipy import ndimage
import h5py
from torch.utils.data.sampler import Sampler
# import augmentations
from torchvision.transforms import functional as F
from torchvision import transforms as T
# from augmentations.ctaugment import OPS
from PIL import Image, ImageOps, ImageFilter

import matplotlib.pyplot as plt

class BaseDataSets_acdc(Dataset):
    def __init__(self, base_dir=None, split='train', num=None, transform=None):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        if self.split == 'train':
            with open(self._base_dir + '/train_slices.list', 'r') as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace('\n', '') for item in self.sample_list]

        elif self.split == 'val':
            with open(self._base_dir + '/val.list', 'r') as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace('\n', '') for item in self.sample_list]
        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(self._base_dir + "/data/slices/{}.h5".format(case), 'r')
        else:
            h5f = h5py.File(self._base_dir + "/data/{}.h5".format(case), 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        sample = {'image': image, 'label': label}
        if self.split == "train":
            sample = self.transform(sample)
        sample["idx"] = idx
        return sample


class BaseDataSets(Dataset):
    def __init__(self,
        imgs_dir,
        masks_dir,
        split="train",
        scale=0.5,
        transform=None, # 设置随机裁剪为256x256    # transform:Compose(in<dataloaders.dataset.RandomGenerator object at x7fad8df120b0>\n.
        ops_weak=None,  # 设置数据弱增强   # None
        ops_strong=None,    # 设置数据强增强   # None
    ):
        self.img_dir = imgs_dir
        self.mask_dir = masks_dir
        self.split = split
        self.scale = scale
        self.transform = transform
        self.ops_weak = ops_weak
        self.ops_strong = ops_strong
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        assert bool(ops_weak) == bool(
            ops_strong
        ), "For using CTAugment learned policies, provide both weak and strong batch augmentation policy"

    def __len__(self):
        return len(self.img_dir)

    @classmethod
    def preprocess(cls, pil_img, scale):
        # pil_img = pil_img.resize((224, 224))
        pil_img = pil_img.resize((256, 256))
        # w, h = pil_img.size
        # newW, newH = int(scale * w), int(scale * h)
        # assert newW > 0 and newH > 0, 'Scale is too small'
        # pil_img = pil_img.resize((newW, newH))



        img_nd = np.array(pil_img)  # img(256, 256, 3),mask(256, 256, 1)
        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)  # axis=2加入一个维度

        # HWC to CHW
        # img_trans = img_nd.transpose((2, 0, 1))  # 将(256,256,3) 转到 (3，256,256)
        # img_trans = img_trans / 255
        # if img_trans.max() > 1:
        # img_trans = img_trans / 255
        img_trans = img_nd / 255
        return img_trans

    def __getitem__(self, idx):

        img_path = self.img_dir[idx]
        mask_path = self.mask_dir[idx]

        img = Image.open(img_path)
        mask = Image.open(mask_path)
        img = img.convert("RGB")  # 转三通道
        mask = mask.convert("L")  # 转一通道

        # assert img.size == mask.size, \
        #     f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'


        img = self.preprocess(img, self.scale)  # 如果不转三通道，就是一通道进行传输(1, 256, 256)
        mask = self.preprocess(mask, self.scale)
        sample = {"image": torch.from_numpy(img), "label": torch.from_numpy(mask)}

        # print(sample["image"].shape)  # brainMRI:torch.Size([256, 256, 3])

        if self.split == "train":
            if None not in (self.ops_weak, self.ops_strong):
                sample = self.transform(sample, self.ops_weak, self.ops_strong)
            else:
                sample = self.transform(sample)
        # print(sample["image"].shape)  # torch.Size([1, 256, 256])
        # print(sample["label"].shape)  # torch.Size([256, 256])
        #  print(len(sample))
        return sample

class ImageToImage2D(Dataset):

    def __init__(self, dataset_path: str, joint_transform: None, one_hot_mask: int = False, image_size: int =224) -> None:

        self.image_size = image_size
        self.input_path = os.path.join(dataset_path, 'imgs')
        self.output_path = os.path.join(dataset_path, 'masks')
        self.images_list = os.listdir(self.input_path)
        self.one_hot_mask = one_hot_mask

        if joint_transform:
            self.joint_transform = joint_transform
        else:
            to_tensor = T.ToTensor()
            self.joint_transform = lambda x, y: (to_tensor(x), to_tensor(y))

    def __len__(self):
        return len(os.listdir(self.input_path))

    def __getitem__(self, idx):

        image_filename = self.images_list[idx]
        #print(image_filename[: -3])
        # read image
        # print(os.path.join(self.input_path, image_filename))
        # print(os.path.join(self.output_path, image_filename[: -3] + "png"))
        # print(os.path.join(self.input_path, image_filename))
        image = cv2.imread(os.path.join(self.input_path, image_filename))
        # print("img",image_filename)
        # print("1",image.shape)
        image = cv2.resize(image,(self.image_size,self.image_size))
        # print(np.max(image), np.min(image))
        # print("2",image.shape)
        # read mask image
        mask = cv2.imread(os.path.join(self.output_path, image_filename[: -3] + "png"),0)
        # print("mask",image_filename[: -3] + "png")
        # print(np.max(mask), np.min(mask))
        mask = cv2.resize(mask,(self.image_size,self.image_size))
        # print(np.max(mask), np.min(mask))
        mask[mask<=0] = 0
        # (mask == 35).astype(int)
        mask[mask>0] = 1
        # print("11111",np.max(mask), np.min(mask))

        # correct dimensions if needed
        image, mask = correct_dims(image, mask)
        # image, mask = F.to_pil_image(image), F.to_pil_image(mask)
        # print("11",image.shape)
        # print("22",mask.shape)
        sample = {'image': image, 'label': mask}

        if self.joint_transform:
            sample = self.joint_transform(sample)
        # sample = {'image': image, 'label': mask}
        # print("2222",np.max(mask), np.min(mask))

        if self.one_hot_mask:
            assert self.one_hot_mask > 0, 'one_hot_mask must be nonnegative'
            mask = torch.zeros((self.one_hot_mask, mask.shape[1], mask.shape[2])).scatter_(0, mask.long(), 1)
        # mask = np.swapaxes(mask,2,0)
        # print(image.shape)
        # print("mask",mask)
        # mask = np.transpose(mask,(2,0,1))
        # image = np.transpose(image,(2,0,1))
        # print(image.shape)
        # print(mask.shape)
        # print(sample['image'].shape)

        return sample


def random_rot_flip(image, label=None):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    if label is not None:
        label = np.rot90(label, k)
        label = np.flip(label, axis=axis).copy()
        return image, label
    else:
        return image


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


def color_jitter(image):
    if not torch.is_tensor(image):
        np_to_tensor = transforms.ToTensor()
        image = np_to_tensor(image)

    # s is the strength of color distortion.
    s = 1.0
    jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    return jitter(image)

def color_jitter_2d(image, p=1.0):
    # if not torch.is_tensor(image):
    #     np_to_tensor = transforms.ToTensor()
    #     image = np_to_tensor(image)
    # s is the strength of color distortion.
    # s = 1.0
    # jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    jitter = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)
    if np.random.random() < p:
        image = jitter(image)
    return image

def to_long_tensor(pic):
    # handle numpy array
    img = torch.from_numpy(np.array(pic, np.uint8))
    # backward compatibility
    return img.long()

def correct_dims(*images):
    corr_images = []
    # print(images)
    for img in images:
        if len(img.shape) == 2:
            corr_images.append(np.expand_dims(img, axis=2))
        else:
            corr_images.append(img)

    if len(corr_images) == 1:
        return corr_images[0]
    else:
        return corr_images

# class CTATransform(object):
#     def __init__(self, output_size, cta):
#         self.output_size = output_size
#         self.cta = cta
#
#     def __call__(self, sample, ops_weak, ops_strong):
#         image, label = sample["image"], sample["label"]
#         image = self.resize(image)
#         label = self.resize(label)
#         to_tensor = transforms.ToTensor()
#
#         # fix dimensions
#         image = torch.from_numpy(image.astype(np.float32))
#         label = torch.from_numpy(label.astype(np.uint8))
#
#         # apply augmentations
#         image_weak = augmentations.cta_apply(transforms.ToPILImage()(image), ops_weak)
#         image_strong = augmentations.cta_apply(image_weak, ops_strong)
#         label_aug = augmentations.cta_apply(transforms.ToPILImage()(label), ops_weak)
#         label_aug = to_tensor(label_aug).squeeze(0)
#         label_aug = torch.round(255 * label_aug).int()
#
#
#         sample = {
#             "image_weak": to_tensor(image_weak),
#             "image_strong": to_tensor(image_strong),
#             "label_aug": label_aug,
#         }
#         return sample
#
#     def cta_apply(self, pil_img, ops):
#         if ops is None:
#             return pil_img
#         for op, args in ops:
#             pil_img = OPS[op].f(pil_img, *args)
#         return pil_img
#
#     def resize(self, image):
#         # x, y = image.shape
#         # return zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
#         x, y, z = image.shape
#         image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y, 3 / z), order=0)
#         image = image.transpose((2, 0, 1))
#         return image


class RandomGenerator(object):

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]  # image:([256, 256, 3]),label:([256, 256, 1])

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        # image:([256, 256, 3]),label:([256, 256, 1])
        x, y, z = image.shape
        image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y, 3 / z), order=0)  # 对每一个轴进行缩放，参数在parameter中修改
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y, 3 / z), order=0)
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image.astype(np.float32))
        label = label.transpose((2, 0, 1))
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {"image": image, "label": label}
        return sample

class RandomGenerator_2D(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image, label = F.to_pil_image(image), F.to_pil_image(label)
        x, y = image.size
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() < 0.5:
            image, label = random_rotate(image, label)

        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = F.to_tensor(image)
        label = to_long_tensor(label)
        sample = {'image': image, 'label': label}
        return sample

class RandomGenerator_acdc(object):

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # ind = random.randrange(0, img.shape[0])
        # image = img[ind, ...]
        # label = lab[ind, ...]
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {'image': image, 'label': label}
        return sample

class ValGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image, label = F.to_pil_image(image), F.to_pil_image(label)
        x, y = image.size
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = F.to_tensor(image)
        label = to_long_tensor(label)
        sample = {'image': image, 'label': label}
        return sample

class WeakStrongAugment(object):
    """returns weakly and strongly augmented images
    Args:
        object (tuple): output size of network
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        x, y, z = image.shape
        image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y, 3 / z),
                     order=0)  # 对每一个轴进行缩放，参数在parameter中修改
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y, 3 / z), order=0)
        # image
        # if random.random() > 0.5:
        #     image, label = random_rot_flip(image, label)
        # elif random.random() > 0.5:
        #     image, label = random_rotate(image, label)
        # weak augmentation is rotation / flip
        image_weak, label = random_rot_flip(image, label)
        # strong augmentation is color jitter
        image_strong = color_jitter(image_weak).type("torch.FloatTensor")
        # fix dimensions
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image.astype(np.float32))
        image_weak = image_weak.transpose((2, 0, 1))
        image_weak = torch.from_numpy(image_weak.astype(np.float32))
        label = label.transpose((2, 0, 1))
        label = torch.from_numpy(label.astype(np.uint8))

        sample = {
            "image": image,
            "image_weak": image_weak,
            "image_strong": image_strong,
            "label_aug": label,
        }
        return sample

    def resize(self, image):
        x, y, z = image.shape
        return zoom(image, (self.output_size[0] / x, self.output_size[1] / y, 3 / z), order=0)


class WeakStrongAugmentMore(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        x, y, z = image.shape
        image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y, 3 / z),
                     order=0)  # 对每一个轴进行缩放，参数在parameter中修改
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y, 3 / z), order=0)
        # image
        # if random.random() > 0.5:
        #     image, label = random_rot_flip(image, label)
        # elif random.random() > 0.5:
        #     image, label = random_rotate(image, label)

        # weak augmentation
        image_weak, label = random_rot_flip(image, label)

        # strong augmentation is color jitter
        image_strong = func_strong_augs(image, p_color=0.5, p_blur=0.2)
        image_strong_more = func_strong_augs(image, p_color=1.0, p_blur=0.2)

        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image.astype(np.float32))
        image_weak = image_weak.transpose((2, 0, 1))
        image_weak = torch.from_numpy(image_weak.astype(np.float32))
        label = label.transpose((2, 0, 1))
        label = torch.from_numpy(label.astype(np.uint8))

        sample = {
            "image": image,
            "image_weak": image_weak,
            "image_strong": image_strong,
            "image_strong_more": image_strong_more,
            "label_aug": label,
        }
        return sample

    def resize(self, image):
        x, y, z = image.shape
        return zoom(image, (self.output_size[0] / x, self.output_size[1] / y, 3 / z), order=0)

def func_strong_augs(image, p_color=0.8, p_blur=0.5):
    img = Image.fromarray((image * 255).astype(np.uint8))
    img = color_jitter_2d(img, p_color)
    img = blur(img, p_blur)

    img = torch.from_numpy(np.array(img).transpose((2,0,1))).float() / 255.0

    return img

def blur(img, p=0.5):
    if random.random() < p:
        sigma = np.random.uniform(0.1, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return img

class WeakStrongAugment_acdc(object):
    """returns weakly and strongly augmented images
    Args:
        object (tuple): output size of network
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        x, y = image.shape
        image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y),
                     order=0)  # 对每一个轴进行缩放，参数在parameter中修改
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        # weak augmentation is rotation / flip
        image_weak, label = random_rot_flip(image, label)
        # strong augmentation is color jitter
        image_strong = color_jitter(image_weak).type("torch.FloatTensor")
        # fix dimensions
        image = torch.from_numpy(image.astype(np.float32))
        image_weak = torch.from_numpy(image_weak.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))

        sample = {
            "image": image,
            "image_weak": image_weak,
            "image_strong": image_strong,
            "label_aug": label,
        }
        return sample

    def resize(self, image):
        x, y = image.shape
        return zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)

class Flip_Color_Augment(object):

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        x, y, z = image.shape
        image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y, 3 / z),
                     order=0)  # 对每一个轴进行缩放，参数在parameter中修改
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y, 3 / z), order=0)
        # weak augmentation is rotation / flip
        # image, label = RandomCrop(image, label)
        image, label = random_rot_flip(image, label)
        # strong augmentation is color jitter
        image_strong = color_jitter(image).type("torch.FloatTensor")
        # fix dimensions
        label = label.transpose((2, 0, 1))
        label = torch.from_numpy(label.astype(np.uint8))

        sample = {
            "image": image_strong,
            "label": label,
        }
        return sample

    def resize(self, image):
        x, y, z = image.shape
        return zoom(image, (self.output_size[0] / x, self.output_size[1] / y, 3 / z), order=0)


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size    # 无标签bs
        self.primary_batch_size = batch_size - secondary_batch_size # 有标签bs

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch) in zip(
                grouper(primary_iter, self.primary_batch_size),
                grouper(secondary_iter, self.secondary_batch_size),
            )
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    #  grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
