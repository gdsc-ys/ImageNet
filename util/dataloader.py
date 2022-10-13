#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename : dataloader
# @Date : 2021-09-01-09-06
# @Project : pytorch_basic
# @Author : seungmin

import os

from PIL import Image
import numpy as np

import torch
from torchvision.datasets import ImageFolder  # Load dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as T

DATA_TRAIN_PATH = os.path.join(os.getcwd(), "data", "imagenette2-320")  # use 절대경로 for train data
DATA_TEST_PATH = os.path.join(os.getcwd(), "data", "imagenetwoof2-320")  # use 절대경로 for test data
MINI_BATCH_SIZE = 128  # numbers of images in a mini-batch
VALID_BATCH_SIZE = 64  # numbers of images in a mini-batch
LEARNING_RATE = 1e-4
EPOCHS = 100
RESOLUTION = (224, 224)

np.random.seed(0)


def make_datapath_list(root_path):
    txt_list = os.listdir(root_path)

    data_list = []
    for idx, txt in enumerate(txt_list):
        with open(os.path.join(root_path, txt)) as f:
            file_list = [line.rstrip() for line in f]
            file_list = [line for line in file_list if line]
            data_list.extend(file_list)

    print("\nNumber of classes: {}".format(len(txt_list)))
    print("Number of training data: {}".format(len(data_list)))
    return data_list, txt_list


class MyDataset(object):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img_path = self.file_list[index]

        img = Image.open(img_path)  # 필요한 경우, .resize((224,224)), .convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if img_path.split("/")[-2].split("_")[-1][:1] == "a":  # 파일명, 또는 폴더명에 a 가 들어가면 비정상으로 라벨링 함.
            label = 1  ## 비정상
        else:
            label = 0  ## 정상

        return img, label


# 학습 데이터셋 wrapper 클래스
class MyTrainSetWrapper(object):
    def __init__(self, batch_size, num_workers, valid_size, train_path):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.path = train_path

    def get_data_loaders(self):
        data_augment = self._get_train_transform()
        # print("flag: ", make_datapath_list(self.path)[1])
        train_dataset = MyDataset(make_datapath_list(self.path)[0], transform=data_augment)

        train_loader, valid_loader = self.get_train_validation_data_loaders(
            train_dataset, transform=data_augment
        )

        return train_loader, valid_loader

    def _get_train_transform(self):
        imagenet_transforms = T.Compose(
            [
                T.Resize(RESOLUTION),  # 3(channel) x 224(width) x 224(height) Resizing
                T.ToTensor(),  # np.array or float -> torch.Tensor()
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        return imagenet_transforms

    def get_train_validation_data_loaders(self, train_dataset, transform=None):
        # # obtain training indices that will be used for validation
        # num_train = len(train_dataset)
        # indices = list(range(num_train))
        # # print(indices)
        # np.random.shuffle(indices)

        # split = int(np.floor(self.valid_size * num_train))
        # train_idx, valid_idx = indices[split:], indices[:split]

        # # define samplers for obtaining training and validation batches
        # train_sampler = SubsetRandomSampler(train_idx)
        # valid_sampler = SubsetRandomSampler(valid_idx)

        train_data = ImageFolder(root=f"{DATA_TRAIN_PATH}/train", transform=transform)
        print("number of training data: ", len(train_data))

        train_loader = torch.utils.data.DataLoader(
            dataset=train_data, batch_size=self.batch_size, shuffle=True
        )

        ## 입력 텐서를 직접 확인하고 싶은 경우, 이하를 실행함.

        # print('-----')
        # for x, y in train_loader:
        #    print("x_len:{0}, x_shape:{1}, x_type:{2}, y:{3}".format(len(x), x[0].shape, type(x[0]), y))
        # print('-----')

        valid_data = ImageFolder(root=f"{DATA_TRAIN_PATH}/val", transform=transform)
        print("number of test data: ", len(valid_data))

        valid_loader = torch.utils.data.DataLoader(
            dataset=valid_data, batch_size=self.batch_size, shuffle=True
        )

        print("\nValidation size: {}%".format(self.valid_size * 100))
        print("Train set: {} / Validation set: {}".format(len(train_loader), len(valid_loader)))
        return train_loader, valid_loader


# 테스트 데이터셋 wrapper 클래스
class MyTestSetWrapper(object):
    def __init__(self, batch_size, num_workers, test_path):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.path = test_path

    def _get_test_transform(self):
        imagenet_transforms = T.Compose(
            [
                T.Resize(RESOLUTION),  # 3(channel) x 224(width) x 224(height) Resizing
                T.ToTensor(),  # np.array or float -> torch.Tensor()
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        return imagenet_transforms

    def get_test_loaders(self):
        data_augment = self._get_test_transform()

        test_data = ImageFolder(root=f"{DATA_TEST_PATH}/val", transform=data_augment)
        print("number of test data: ", len(test_data))

        test_loader = torch.utils.data.DataLoader(
            dataset=test_data, batch_size=self.batch_size, shuffle=True
        )

        return test_loader
