import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torchsampler import ImbalancedDatasetSampler
from PIL import Image
from imgaug import augmenters as iaa

from src import utils

CLASSES_LABELS = ['piste-cyclable', 'route', 'sentier', 'trottoir', 'voie-partagee']

DATASET_RESOLUTION_SMALL = {
    'train_dir': '../data/classified_images_80x45/train_images',
    'test_dir': '../data/classified_images_80x45/test_images',
    'padding': (74, 90, 73, 89),
    'resolution': '80x45'
}
DATASET_RESOLUTION_MEDIUM = {
    'train_dir': '../data/classified_images_160x90/train_images',
    'test_dir': '../data/classified_images_160x90/test_images',
    'padding': (32, 67),
    'resolution': '160x90'
}
DATASET_RESOLUTION_LARGE = {
    'train_dir': '../data/classified_images_320x180/train_images',
    'test_dir': '../data/classified_images_320x180/test_images',
    'padding': (0, 22),
    'resolution': '320x180'
}


def image_loader(loader, image):
    image = loader(image).float()
    image = image.clone().detach().requires_grad_(True)
    image = image.unsqueeze(0)
    return image


class ImgAugTransform:
    def __init__(self):
        self.aug = iaa.Sequential([
            iaa.Fliplr(0.2),
            iaa.Sometimes(0.2, iaa.GaussianBlur(sigma=(0, 1.0))),
            iaa.Affine(rotate=(-20, 20), mode='symmetric'),
            iaa.Sometimes(0.2, iaa.OneOf([iaa.Dropout(), iaa.CoarseDropout(0.02, size_percent=0.5)])),
        ])

    def __call__(self, img):
        img = np.array(img)
        aug_img = self.aug.augment_image(img)
        return Image.fromarray(aug_img)


class Data:

    def __init__(self, batch_size=4, valid_size=0.2, dataset_resolution=None):

        if dataset_resolution is None:
            dataset_resolution = DATASET_RESOLUTION_MEDIUM

        train_dir = dataset_resolution.get('train_dir')
        test_dir = dataset_resolution.get('test_dir')

        train_transform = transforms.Compose([
            transforms.Pad(dataset_resolution.get('padding')),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        test_transform = train_transform

        train_data = torchvision.datasets.ImageFolder(train_dir, transform=train_transform)
        test_data = torchvision.datasets.ImageFolder(test_dir, transform=test_transform)

        num_train = len(train_data)
        indices_train = list(range(num_train))
        np.random.shuffle(indices_train)
        split_tv = int(np.floor(valid_size * num_train))
        train_idx, valid_idx = indices_train[split_tv:], indices_train[:split_tv]

        train_sampler = ImbalancedDatasetSampler(train_data, indices=train_idx)
        valid_sampler = ImbalancedDatasetSampler(train_data, indices=valid_idx)

        self.train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=train_sampler,
                                                        num_workers=2)
        self.valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler,
                                                        num_workers=2)
        self.test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=2)


if __name__ == '__main__':
    number_images = 10

    data_loader = Data(number_images)
    data_iter = iter(data_loader.train_loader)
    images, labels = data_iter.next()
    utils.imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % CLASSES_LABELS[labels[j]] for j in range(number_images)))

    data_iter = iter(data_loader.test_loader)
    images, labels = data_iter.next()
    utils.imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % CLASSES_LABELS[labels[j]] for j in range(number_images)))

    print('Distribution of classes in train dataset:')
    _, ax = plt.subplots()
    labels = [label for _, label in data_loader.train_loader.dataset.imgs]
    class_labels, counts = np.unique(labels, return_counts=True)
    ax.bar(class_labels, counts)
    ax.set_xticks(class_labels)
    plt.show()

    print('Distribution of classes in test dataset:')
    _, ax = plt.subplots()
    labels = [label for _, label in data_loader.test_loader.dataset.imgs]
    class_labels, counts = np.unique(labels, return_counts=True)
    ax.bar(class_labels, counts)
    ax.set_xticks(class_labels)
    plt.show()
