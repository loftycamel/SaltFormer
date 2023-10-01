from typing import Any, Dict, List, Optional, Type, Sequence
from pathlib import Path
import csv
import torch
from torch.nn import functional as F
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from skimage import io
import pandas as pd
from skimage.morphology import convex_hull_image
from sklearn.model_selection import train_test_split
import torchvision.transforms as T
from transforms.unet_transforms import *
from .salt_dataset import SaltDataset
import pytorch_lightning as pl


class SaltDataModule(pl.LightningDataModule):
    orig_img_size = 101
    img_size = 128
    mean = [0.485, 0.456, 0.406] 
    std = [0.229, 0.224, 0.225]
    def __init__(self
        , data_dir: str = "path/to/dir"
        , splits: Sequence = [.8, .1, .1]
        , stratify: str = None
        , batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.splits = splits
        self.stratify = stratify
        self.batch_size = batch_size
        #
        self._load_data()
        self._setup_transforms()

    def _setup_transforms(self):
        padding = compute_padding(self.orig_img_size, self.orig_img_size, self.img_size)
        geometric_transform_prob = 0.5 * 0.25
        geometric_transform = T.Compose([T.RandomApply([CropAndRescale(max_scale=0.2)], p=geometric_transform_prob),
                                    T.RandomApply([HorizontalShear(max_scale=0.07)], p=geometric_transform_prob),
                                    T.RandomApply([Rotation(max_angle=15)], p=geometric_transform_prob),
                                    T.RandomApply([ElasticDeformation(max_distort=0.15)], p=geometric_transform_prob)])
        brightness_transform_prob = 0.5 * 0.33
        brightness_transform = T.Compose([T.RandomApply([BrightnessShift(max_value=0.1)], p=brightness_transform_prob),
                                        T.RandomApply([BrightnessScaling(max_value=0.08)], p=brightness_transform_prob),
                                        T.RandomApply([GammaChange(max_value=0.08)], p=brightness_transform_prob)])
        self.train_transform = T.Compose([PrepareImageAndMask(),
                                T.RandomApply([Cutout(1, 30)], p=0.0),
                                T.RandomApply([HorizontalFlip()]),
                                geometric_transform,
                                brightness_transform,
                                ResizeToNxN(self.img_size),HWCtoCHW(),CHWNormalize(self.mean,self.std)])
        self.valid_transform = T.Compose([PrepareImageAndMask(),
                                ResizeToNxN(self.img_size),HWCtoCHW(),CHWNormalize(self.mean,self.std)])

    def _load_data(self):
        train_csv = Path(self.data_dir)/'train.csv'
        depths_csv = Path(self.data_dir)/'depths.csv'
        lines = []
        depths = pd.read_csv(depths_csv)
        with open(train_csv) as f:
            reader = csv.reader(f)
            for row in reader:
                image = '{0}.png'.format(row[0])
                img_path = Path(self.data_dir)/'train'/'images'/image
                mask_path = Path(self.data_dir)/'train'/'masks'/image
                qstr="id=='%s'"%row[0]
                z=0
                depth=depths.query(qstr)
                if not depth.empty:
                    z = int(depth['z'])
                lines.append([img_path,mask_path,z])
            lines.pop(0) #head line
        #load images and masks
        all_data = []
        pbar = tqdm(lines, desc="Load datasets", total=len(lines), unit="images")
        for item in pbar:
            img = io.imread(item[0])
            mask = io.imread(item[1], as_gray=True) #self._rle2mask(self.data[idx][1])
            mask[mask>0]=255
            data = {'input':img,'mask':mask,'z':item[2]}
            all_data.append(data)
        # split
        stratify=None
        if self.stratify == 'coverage':
            stratify = list(map(self.cov_to_class,all_data))
        elif self.stratify=='shape':
            stratify = list(map(self.shape_to_class,all_data))
        elif self.stratify=='depth':
            stratify = list(map(self.depth_to_class,all_data))
        # lengths=len(all_data)*np.array(self.splits)
        train_size = float(self.splits[0])
        test_size = float(self.splits[1])
        if len(self.splits) > 2:
            test_size=float(self.splits[2])
        train_ds,test_ds = train_test_split(
            all_data, test_size=test_size,random_state=42,stratify=stratify)
        self.datasets=[train_ds, test_ds]

        if len(self.splits) > 2:
            valid_size = float(self.splits[1])
            train_ds,valid_ds = train_test_split(
                train_ds, test_size=valid_size,random_state=42,stratify=stratify)
            self.datasets=[train_ds, valid_ds, test_ds]
    def depth_to_class(self, val):
        dc = val['z']/100
        for i in range(0,11):
            if dc <= i :
                return i
    def cov_to_class(self,val):
        cov = np.sum(val['mask']/255)/(self.orig_img_size*self.orig_img_size)
        for i in range(0, 11):
            if cov * 10 <= i :
                return i
    def shape_to_class(self, val):
        sc = self.shape_complexity(val['mask']/255)
        for i in range(0,11):
            if sc * 10 <= i :
                return i
    def shape_complexity(self,mask):
        mask=mask.astype(bool)
        chull = convex_hull_image(mask)
        mask_v=sum(sum(mask))
        chull_v=sum(sum(chull))
        if chull_v==0:
            v=1
        else:
            v=mask_v/chull_v
        return 1-v    
    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = SaltDataset(
                self.datasets[0], self.orig_img_size, self.orig_img_size, self.train_transform)
            self.valid_dataset = SaltDataset(
                self.datasets[1], self.orig_img_size, self.orig_img_size, self.valid_transform)
        if stage == "test":
            i = 2 if len(self.splits) > 2 else 1
            self.test_dataset = SaltDataset(
                self.datasets[i], self.orig_img_size, self.orig_img_size, self.valid_transform)

    def train_dataloader(self,num_workers:int=8):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=num_workers)

    def val_dataloader(self,num_workers=8):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size,num_workers=num_workers)

    def test_dataloader(self,num_workers=8):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,num_workers=num_workers)

    def teardown(self, stage: str):
        pass
