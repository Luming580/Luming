import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import ImageEnhance


# several data augumentation strategies
def cv_random_flip(img, target_img, label):
    flip_flag = random.randint(0, 1)

    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        target_img = target_img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)

    return img, target_img, label


def randomCrop(image, target_image, label):
    border = 30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), target_image.crop(random_region), label.crop(
        random_region)


def randomRotation(image, target_image, label):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, mode)
        target_image = target_image.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
    return image, target_image, label


def colorEnhance(image):
    bright_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image


def randomGaussian(image, mean=0.1, sigma=0.35):
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

    img = np.asarray(image)
    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])
    return Image.fromarray(np.uint8(img))


def randomPeper(img):
    img = np.array(img)  # 将图像转换为NumPy数组
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
    for i in range(noiseNum):

        randX = random.randint(0, img.shape[0] - 1)
        randY = random.randint(0, img.shape[1] - 1)
        if random.randint(0, 1) == 0:
            img[randX, randY] = 0
        else:
            img[randX, randY] = 255

    return Image.fromarray(img)


# dataset for training
class SalObjDataset(data.Dataset):
    def __init__(self, image_root, target_image_root, gt_root, trainsize):
        self.trainsize = trainsize
        self.original_images = [image_root + f for f in os.listdir(image_root) if
                                f.endswith('.jpg') or f.endswith('.png')]
        self.target_images = [target_image_root + f for f in os.listdir(target_image_root) if
                              f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]

        self.original_images = sorted(self.original_images)
        self.target_images = sorted(self.target_images)
        self.gts = sorted(self.gts)
        self.size = len(self.original_images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.4975], [0.5])
        ])
        self.target_img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.4975], [0.5])
            # transforms.Normalize([0.4975], [0.09887])
        ])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])

    def __getitem__(self, index):
        original_images = self.binary_loader(self.original_images[index])
        target_image = self.binary_loader(self.target_images[index])
        gt = self.binary_loader(self.gts[index])

        original_images, target_image, gt = cv_random_flip(original_images, target_image, gt)
        original_images, target_image, gt = randomCrop(original_images, target_image, gt)

        original_images = self.img_transform(original_images)
        target_image = self.target_img_transform(target_image)
        gt = self.gt_transform(gt)
        filename = os.path.basename(self.target_images[index % len(self.target_images)])

        return original_images, target_image, gt, filename

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size


# dataloader for training
def get_loader(image_root, target_image_root, gt_root, batchsize, trainsize, shuffle=True, num_workers=0,
               pin_memory=True):  # 创建数据集实例
    dataset = SalObjDataset(image_root, target_image_root, gt_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


def get_test_loader(image_root, target_image_root, gt_root, batchsize, testsize, shuffle=True, num_workers=0,
                    pin_memory=True):
    datasets = test_dataset(image_root, target_image_root, gt_root, testsize)
    data_loader = data.DataLoader(dataset=datasets,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


# test dataset and loader
class test_dataset:
    def __init__(self, image_root, target_image_root, gt_root, testsize):
        self.testsize = testsize
        self.original_images = [image_root + f for f in os.listdir(image_root) if
                                f.endswith('.jpg') or f.endswith('.png')]  # 获取图片文件路径列表
        self.target_images = [target_image_root + f for f in os.listdir(target_image_root) if
                              f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]

        self.original_images = sorted(self.original_images)
        self.target_images = sorted(self.target_images)
        self.gts = sorted(self.gts)
        self.size = len(self.original_images)
        self.image_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.4975], [0.09887])
        ])

        self.gt_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor()])
        self.size = len(self.original_images)
        self.index = 0

    def load_data(self):
        original_images = self.binary_loader(self.original_images[self.index])
        original_images = self.image_transform(original_images).unsqueeze(0)
        target_images = self.binary_loader(self.target_images[self.index])
        tar = target_images
        target_images = self.image_transform(target_images).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        g = gt
        g = self.gt_transform(g)
        name = self.original_images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        self.index = self.index % self.size

        return original_images, target_images, tar, gt, name, g

    def __getitem__(self, index):
        original_image = self.binary_loader(self.original_images[index])
        target_image = self.binary_loader(self.target_images[index])
        gt = self.binary_loader(self.gts[index])

        original_image = self.image_transform(original_image)
        target_image = self.image_transform(target_image)
        gt = self.gt_transform(gt)
        filename = os.path.basename(self.target_images[index % len(self.target_images)])

        return {'A': target_image, 'B': original_image, "C": gt, 'filename': filename}

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size
