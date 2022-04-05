# Copyright (C) 2018 Elvis Yu-Jing Lin <elvisyjlin@gmail.com>
# 
# This work is licensed under the MIT License. To view a copy of this license,
# visit https://opensource.org/licenses/MIT.

"""Custom datasets for CelebA and CelebA-HQ."""

import numpy as np
import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import torchio as tio
import random


class Custom(data.Dataset):
    def __init__(self, data_path, attr_path, image_size, selected_attrs):
        self.data_path = data_path
        att_list = open(attr_path, 'r', encoding='utf-8').readlines()[1].split()
        atts = [att_list.index(att) + 1 for att in selected_attrs]
        self.images = np.loadtxt(attr_path, skiprows=2, usecols=[0], dtype=np.str)
        self.labels = np.loadtxt(attr_path, skiprows=2, usecols=atts, dtype=np.int)
        
        self.tf = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    def __getitem__(self, index):
        img = self.tf(Image.open(os.path.join(self.data_path, self.images[index])))
        att = torch.tensor((self.labels[index] + 1) // 2)
        return img, att
    
    def __len__(self):
        return len(self.images)

class CelebA(data.Dataset):
    def __init__(self, data_path, attr_path, image_size, mode, selected_attrs):
        super(CelebA, self).__init__()
        self.data_path = data_path
        att_list = open(attr_path, 'r', encoding='utf-8').readlines()[1].split()
        atts = [att_list.index(att) + 1 for att in selected_attrs]
        images = np.loadtxt(attr_path, skiprows=2, usecols=[0], dtype=np.str)
        labels = np.loadtxt(attr_path, skiprows=2, usecols=atts, dtype=np.int)
        
        if mode == 'train':
            self.images = images[:182000]
            self.labels = labels[:182000]
        if mode == 'valid':
            self.images = images[182000:182637]
            self.labels = labels[182000:182637]
        if mode == 'test':
            self.images = images[182637:]
            self.labels = labels[182637:]
        
        self.tf = transforms.Compose([
            transforms.CenterCrop(170),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
                                       
        self.length = len(self.images)
    def __getitem__(self, index):
        img = self.tf(Image.open(os.path.join(self.data_path, self.images[index])))
        att = torch.tensor((self.labels[index] + 1) // 2)
        return img, att
    def __len__(self):
        return self.length


class WBC(data.Dataset):
    def __init__(self, data_path, attr_path, image_size, mode, selected_attrs, aug):
        super(WBC, self).__init__()
        self.aug = aug
        self.data_path = data_path
        att_list = open(attr_path, 'r', encoding='utf-8').readlines()[1].split()
        atts = [att_list.index(att) + 1 for att in selected_attrs]
        images = np.loadtxt(attr_path, skiprows=2, usecols=[0], dtype=np.str)
        labels = np.loadtxt(attr_path, skiprows=2, usecols=atts, dtype=np.int)

        if mode == 'train':
            self.images = images[:2820]
            self.labels = labels[:2820]
        if mode == 'valid':
            self.images = images[-24:-12]
            self.labels = labels[-24:-12]
        if mode == 'test':
            self.images = images[-12:]
            self.labels = labels[-12:]

        self.tf = transforms.Compose([
            transforms.RandomRotation(degrees=(0, 20)),
            transforms.Resize(int(240 * random.uniform(1.15,1.25))),
            transforms.ColorJitter(brightness=(0.8, 1.2),contrast=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor()
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.length = len(self.images)


    def __getitem__(self, index):
        img = self.tf(Image.open(os.path.join(self.data_path, self.images[index]))) # (b, h, w)
        att = torch.tensor((self.labels[index] + 1) // 2)

        osize = img.shape[2]
        w_offset = random.randint(0, max(0, osize - 256 - 1))
        h_offset = random.randint(0, max(0, osize - 256 - 1))
        img = img[:, h_offset:h_offset + 256, w_offset:w_offset + 256]
        img = img / img.max()
        img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img)

        return img, att

    def __len__(self):
        return self.length

    def get_labels(self):
        label = [int(np.where(self.labels[i]==1)[0]) for i in range(len(self.labels))]
        return label


class WBC_all(data.Dataset):
    def __init__(self, data_path, attr_path, image_size, mode, selected_attrs, aug):
        super(WBC_all, self).__init__()
        self.aug = aug
        self.data_path = data_path
        att_list = open(attr_path, 'r', encoding='utf-8').readlines()[1].split()
        atts = [att_list.index(att) + 1 for att in selected_attrs]
        images = np.loadtxt(attr_path, skiprows=2, usecols=[0], dtype=np.str)
        labels = np.loadtxt(attr_path, skiprows=2, usecols=atts, dtype=np.int)

        if mode == 'train':
            self.images = images[:27453]
            self.labels = labels[:27453]
        if mode == 'valid':
            self.images = images[-192:-180]
            self.labels = labels[-192:-180]
        if mode == 'test':
            self.images = images[-180:]
            self.labels = labels[-180:]

        if mode == 'train':
            self.tf = transforms.Compose([
                transforms.RandomRotation(degrees=(0, 20)),
                transforms.Resize(int(240 * random.uniform(1.15,1.25))),
                transforms.ColorJitter(brightness=(0.8, 1.2),contrast=(0.9, 1.1)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ToTensor()
                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize(int(240 * random.uniform(1.15, 1.25))),
                transforms.ToTensor()
            ])
        self.length = len(self.images)


    def __getitem__(self, index):
        img = self.tf(Image.open(os.path.join(self.data_path, self.images[index]))) # (b, h, w)
        att = torch.tensor((self.labels[index] + 1) // 2)

        osize = img.shape[2]
        w_offset = random.randint(0, max(0, osize - 256 - 1))
        h_offset = random.randint(0, max(0, osize - 256 - 1))
        img = img[:, h_offset:h_offset + 256, w_offset:w_offset + 256]
        img = img / img.max()
        img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img)

        return img, att, self.images[index]

    def __len__(self):
        return self.length

    def get_labels(self):
        label = [int(np.where(self.labels[i]==1)[0])+1 for i in range(len(self.labels))]
        return label


class CelebA_HQ(data.Dataset):
    def __init__(self, data_path, attr_path, image_list_path, image_size, mode, selected_attrs):
        super(CelebA_HQ, self).__init__()
        self.data_path = data_path
        att_list = open(attr_path, 'r', encoding='utf-8').readlines()[1].split()
        atts = [att_list.index(att) + 1 for att in selected_attrs]
        orig_images = np.loadtxt(attr_path, skiprows=2, usecols=[0], dtype=np.str)
        orig_labels = np.loadtxt(attr_path, skiprows=2, usecols=atts, dtype=np.int)
        indices = np.loadtxt(image_list_path, skiprows=1, usecols=[1], dtype=np.int)
        
        images = ['{:d}.jpg'.format(i) for i in range(30000)]
        labels = orig_labels[indices]
        
        if mode == 'train':
            self.images = images[:28000]
            self.labels = labels[:28000]
        if mode == 'valid':
            self.images = images[28000:28500]
            self.labels = labels[28000:28500]
        if mode == 'test':
            self.images = images[28500:]
            self.labels = labels[28500:]
        
        self.tf = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
                                       
        self.length = len(self.images)
    def __getitem__(self, index):
        img = self.tf(Image.open(os.path.join(self.data_path, self.images[index])))
        att = torch.tensor((self.labels[index] + 1) // 2)
        return img, att
    def __len__(self):
        return self.length

def check_attribute_conflict(att_batch, att_name, att_names):# att_names = ['band', 'blast', 'meta','myelo', 'promyelo', 'seg']
    def _get(att, att_name):
        if att_name in att_names:
            return att[att_names.index(att_name)]
        return None
    def _set(att, value, att_name):
        if att_name in att_names:
            att[att_names.index(att_name)] = value
    att_id = att_names.index(att_name) # first time 0 second time 1....
    for att in att_batch:
        if att_name in ['Bald', 'Receding_Hairline'] and att[att_id] != 0:
            if _get(att, 'Bangs') != 0: # if this type is not bangs
                _set(att, 1-att[att_id], 'Bangs')
        elif att_name == 'Bangs' and att[att_id] != 0:
            for n in ['Bald', 'Receding_Hairline']:
                if _get(att, n) != 0:
                    _set(att, 1-att[att_id], n)
                    _set(att, 1-att[att_id], n)
        elif att_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair'] and att[att_id] != 0:
            for n in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                if n != att_name and _get(att, n) != 0:
                    _set(att, 1-att[att_id], n)
        elif att_name in ['Straight_Hair', 'Wavy_Hair'] and att[att_id] != 0:
            for n in ['Straight_Hair', 'Wavy_Hair']:
                if n != att_name and _get(att, n) != 0:
                    _set(att, 1-att[att_id], n)
        elif att_name in ['Mustache', 'No_Beard'] and att[att_id] != 0:
            for n in ['Mustache', 'No_Beard']:
                if n != att_name and _get(att, n) != 0:
                    _set(att, 1-att[att_id], n)
    return att_batch


if __name__ == '__main__':
    import argparse
    import matplotlib.pyplot as plt
    import torchvision.utils as vutils

    attrs_default = [
        'band', 'blast', 'meta',
        'myelo', 'promyelo', 'seg'
    ]
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--attrs', dest='attrs', default=attrs_default, nargs='+', help='attributes to test')
    parser.add_argument('--data_path', dest='data_path', type=str, default='data/wbc')
    parser.add_argument('--attr_path', dest='attr_path', type=str, default='data/WBC_train.txt')
    args = parser.parse_args()
    
    dataset = WBC(args.data_path, args.attr_path, 256, 'valid', args.attrs)
    dataloader = data.DataLoader(
        dataset, batch_size=64, shuffle=False, drop_last=False
    )

    print('Attributes:')
    print(args.attrs)
    for x, y in dataloader:
        vutils.save_image(x, 'test.png', nrow=8, normalize=True, range=(-1., 1.))
        print(y)
        break
    del x, y
    
    dataset = CelebA(args.data_path, args.attr_path, 128, 'valid', args.attrs)
    dataloader = data.DataLoader(
        dataset, batch_size=16, shuffle=False, drop_last=False
    )