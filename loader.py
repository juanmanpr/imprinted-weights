import torch
from PIL import Image
import os
import pandas as pd
import math
from torch.utils.data.sampler import WeightedRandomSampler
import numpy as np
import random

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class ImageLoader(torch.utils.data.Dataset):

    def __init__(self, root, transform=None, target_transform=None, train=False, num_classes=100, num_train_sample=0, novel_only=False, aug=False,
                 loader=pil_loader):
        img_folder = os.path.join(root, "images")
        img_paths = pd.read_csv(os.path.join(root, "images.txt"), sep=" ", header=None, names=['idx', 'path'])
        img_labels = pd.read_csv(os.path.join(root, "image_class_labels.txt"), sep=" ", header=None,  names=['idx', 'label'])
        train_test_split = pd.read_csv(os.path.join(root, "train_test_split.txt"), sep=" ", header=None,  names=['idx', 'train_flag'])
        data = pd.concat([img_paths, img_labels, train_test_split], axis=1)
        data = data[data['train_flag'] == train]
        data['label'] = data['label'] - 1

        # split dataset
        data = data[data['label'] < num_classes]
        base_data = data[data['label'] < 100]
        novel_data = data[data['label'] >= 100]
        
        # sampling from novel classes
        if num_train_sample != 0:
            novel_data = novel_data.groupby('label', group_keys=False).apply(lambda x: x.iloc[:num_train_sample])

        # whether only return data of novel classes
        if novel_only:
            data = novel_data
        else:
            data = pd.concat([base_data, novel_data])

        # repeat 5 times for data augmentation
        if aug:
            tmp_data = pd.DataFrame()
            for i in range(5):
                tmp_data = pd.concat([tmp_data, data])
            data = tmp_data
        imgs = data.reset_index(drop=True)
        
        if len(imgs) == 0:
            raise(RuntimeError("no csv file"))
        self.root = img_folder
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.train = train

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        item = self.imgs.iloc[int(index)]
        file_path = item['path']
        target = item['label']

        img = self.loader(os.path.join(self.root, file_path))
        
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)

    def get_balanced_sampler(self):
        img_labels = np.array(self.imgs['label'].tolist())
        class_sample_count = np.array([len(np.where(img_labels==t)[0]) for t in np.unique(img_labels)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in img_labels])
        samples_weight = torch.from_numpy(samples_weight)
        sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))
        return sampler
         
class EpisodicLoader(torch.utils.data.Dataset):

    def __init__(self, root, transform=None, target_transform=None, num_train_sample=0, aug=False):
        img_folder = os.path.join(root, "images")
        img_paths  = pd.read_csv(os.path.join(root, "images.txt"), sep=" ", header=None, names=['idx', 'path'])
        img_labels = pd.read_csv(os.path.join(root, "image_class_labels.txt"), sep=" ", header=None,  names=['idx', 'label'])
        original_train_test_split = pd.read_csv(os.path.join(root, "train_test_split.txt"), sep=" ", header=None,  names=['idx', 'train_flag'])
        
        data = pd.concat([img_paths, img_labels, original_train_test_split], axis=1)
        data = data[data['train_flag'] == True]
        data['label'] = data['label'] - 1

        # split dataset
        # we will only simulate novel classes from the original training categories
        all_data = data[data['label'] < 100] 

        # from the training set, we will use 80 categories for episodic training
        tra_data = all_data[all_data['label'] <  80]
        # a sample in here is in fact an episode
        # every episode will simulate a new problem with a different 
        # set of base classes (40 base classes) and a different set of novel classes
        # (other 40)


        # repeat 5 times for data augmentation
        if aug:
            tmp_data = pd.DataFrame()
            for i in range(5):
                tmp_data = pd.concat([tmp_data, tra_data])
            tra_data = tmp_data
        imgs = tra_data.reset_index(drop=True)
        
        if len(imgs) == 0:
            raise(RuntimeError("no csv file"))
        self.root = img_folder
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = pil_loader
        self.img_labels = np.array(self.imgs['label'].tolist())
        
    def __len__(self):
        return 1000
                
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (support_images, query_images, base_images, target) where target is class_index of the target class.
        """

        class_indices_all = list(range(80))
        random.shuffle(class_indices_all)
        
        class_indices_base = class_indices_all[:5]
        class_indices_test = class_indices_all[5:10]
        
        per_query_class_imgs = []
        per_novel_class_imgs = []
        for test_idx in class_indices_test:
            c_i        = self.imgs[self.imgs['label']==test_idx].sample(n=10)#5 for support, 5 for query
            per_query_class_imgs.append(c_i.iloc[:5])
            per_novel_class_imgs.append(c_i.iloc[5:])
            
                        
        per_base_class_imgs = []
        for base_idx in class_indices_base:
            c_i        = self.imgs[self.imgs['label']==base_idx].sample(n=5)#5 for base classes
            per_base_class_imgs.append(c_i)        
            
        fake_novel_query = [] 
        fake_novel_query_labels  = [] 
        for c_i in per_query_class_imgs:
            for ii, item in c_i.iterrows():
                file_path = item['path']
                target = item['label']                    
                img = self.loader(os.path.join(self.root, file_path))
                fake_novel_query.append(img)
                fake_novel_query_labels.append(target)
        
        fake_novel_samples = [] 
        fake_novel_labels  = [] 
        for c_i in per_novel_class_imgs:
            for ii, item in c_i.iterrows():
                file_path = item['path']
                target = item['label']                    
                img = self.loader(os.path.join(self.root, file_path))
                fake_novel_samples.append(img)
                fake_novel_labels.append(target)        
        
        base_samples = []
        base_labels  = [] 
        for c_i in per_base_class_imgs:
            for ii, item in c_i.iterrows():
                file_path = item['path']
                target = item['label']                    
                img = self.loader(os.path.join(self.root, file_path))
                base_samples.append(img)
                base_labels.append(target)        
                
        if self.transform is not None:
            for i,img in enumerate(fake_novel_samples):
                fake_novel_samples[i] = self.transform(img)
            for i,img in enumerate(fake_novel_query):
                fake_novel_query[i] = self.transform(img)                
            for i,img in enumerate(base_samples):
                base_samples[i] = self.transform(img)       
                
        return base_samples, base_labels, fake_novel_samples, fake_novel_query, fake_novel_labels
        
class EpisodicValLoader(torch.utils.data.Dataset):

    def __init__(self, root, transform=None, target_transform=None, num_train_sample=0, novel_only=False, aug=False,
                 loader=pil_loader):
        img_folder = os.path.join(root, "images")
        img_paths = pd.read_csv(os.path.join(root, "images.txt"), sep=" ", header=None, names=['idx', 'path'])
        img_labels = pd.read_csv(os.path.join(root, "image_class_labels.txt"), sep=" ", header=None,  names=['idx', 'label'])
        train_test_split = pd.read_csv(os.path.join(root, "train_test_split.txt"), sep=" ", header=None,  names=['idx', 'train_flag'])
        data = pd.concat([img_paths, img_labels, train_test_split], axis=1)
        data = data[data['train_flag'] == True]
        data['label'] = data['label'] - 1

        # split dataset
        data = data[data['label'] < 100]
        base_data = data[(data['label']<90) & (data['label']>=80)]
        novel_data = data[data['label'] >= 90]
        
        # sampling from novel classes
        if num_train_sample != 0:
            novel_data = novel_data.groupby('label', group_keys=False).apply(lambda x: x.iloc[:num_train_sample])

        # whether only return data of novel classes
        if novel_only:
            data = novel_data
        else:
            data = pd.concat([base_data, novel_data])

        # repeat 5 times for data augmentation
        if aug:
            tmp_data = pd.DataFrame()
            for i in range(5):
                tmp_data = pd.concat([tmp_data, data])
            data = tmp_data
        imgs = data.reset_index(drop=True)
        
        if len(imgs) == 0:
            raise(RuntimeError("no csv file"))
        self.root = img_folder
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        item = self.imgs.iloc[int(index)]
        file_path = item['path']
        target = item['label']

        img = self.loader(os.path.join(self.root, file_path))
        
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)

    def get_balanced_sampler(self):
        img_labels = np.array(self.imgs['label'].tolist())
        class_sample_count = np.array([len(np.where(img_labels==t)[0]) for t in np.unique(img_labels)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in img_labels])
        samples_weight = torch.from_numpy(samples_weight)
        sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))
        return sampler       
