from io import BytesIO

from PIL import Image
from torch.utils.data import Dataset
import random
import data.util as Util
import numpy as np
import os
from glob import glob
import re

class LIHIDataset(Dataset):
    def __init__(self, dataroot, patientsID, datatype, l_resolution=16, r_resolution=128, split='train', data_len=-1, need_LI=False):
        self.datatype = datatype
        self.l_res = l_resolution
        self.r_res = r_resolution
        self.data_len = data_len
        self.need_LI = need_LI
        self.split = split

        if datatype == 'img':
            # self.ri_path = Util.get_paths_from_images(
            #     '{}/ri_{}_{}'.format(dataroot, l_resolution, r_resolution))
            # self.hi_path = Util.get_paths_from_images(
            #     '{}/hi_{}'.format(dataroot, r_resolution))
            # if self.need_LI:
            #     self.li_path = Util.get_paths_from_images(
            #         '{}/li_{}'.format(dataroot, l_resolution))
            self.rip = sorted(glob(os.path.join(dataroot,'*_limited.npy')))
            self.hip = sorted(glob(os.path.join(dataroot,'*_full.npy')))
            self.ri_path = [f for f in self.rip if re.search(r"(C|N|L)\d{3}", f)[0] in patientsID]
            self.hi_path = [f for f in self.hip if re.search(r"(C|N|L)\d{3}", f)[0] in patientsID]
            if self.need_LI:
                self.lip = sorted(glob(os.path.join(dataroot,'*_limited.npy')))
                self.li_path = [f for f in self.rip if re.search(r"(C|N|L)\d{3}", f)[0] in patientsID]
            self.dataset_len = len(self.hi_path)
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        else:
            raise NotImplementedError(
                'data_type [{:s}] is not recognized.'.format(datatype))

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        img_HI = None
        img_LI = None

        # img_HI = Image.open(self.hi_path[index]).convert("RGB")
        # print(self.hi_path[index])
        img_HI = np.load(self.hi_path[index])
        # print(img_HI.astype('uint8').shape)
        # img_HI = np.concatenate((img_HI,img_HI,img_HI),axis=2)
        # img_HI = Image.fromarray(img_HI[0]*600).convert('L')
        # print(self.ri_path[index])
        img_RI = np.load(self.ri_path[index])
        # img_RI = np.concatenate((img_RI,img_RI,img_RI),axis=2)
        # img_RI = Image.fromarray(img_RI[0]*1000).convert('L')
        # img_RI = Image.open(self.ri_path[index]).convert("RGB")
        if self.need_LI:
            # img_LI = Image.open(self.li_path[index]).convert("RGB")
            img_LI = np.load(self.li_path[index])
            # img_LI = np.concatenate((img_LI,img_LI,img_LI),axis=2)
            # img_LI = Image.fromarray(img_LI[0]*1000).convert('L')
        if self.need_LI:
            [img_LI, img_RI, img_HI] = Util.transform_augment(
                [img_LI, img_RI, img_HI], split=self.split, min_max=(-1, 1))
            return {'LI': img_LI, 'HI': img_HI, 'RI': img_RI, 'Index': index}
        else:
            [img_RI, img_HI] = Util.transform_augment(
                [img_RI, img_HI], split=self.split, min_max=(-1, 1))
            return {'HI': img_HI, 'RI': img_RI, 'Index': index}
