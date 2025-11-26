from typing import Tuple
import random

import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from torchvision.io import read_image
from functools import partial
from utils.utils import grab_hard_eval_image_augmentations, grab_soft_eval_image_augmentations, grab_image_augmentations
import torch.nn.functional as F
import numpy as np
import albumentations as A

def resize_tensor(tensor, size):
    if tensor.dtype == torch.uint8:
        tensor = tensor.float() / 255.0  # 转为 float 并归一化（可选）
    tensor = tensor.unsqueeze(0) if tensor.dim() == 3 else tensor
    return F.interpolate(tensor, size=size, mode='bilinear', align_corners=False).squeeze(0)

def convert_to_float(x):
  return x.float()
def resize_tensor_fixed(x, size):
    return resize_tensor(x, size)
def convert_to_ts(x, **kwargs):
  x = np.clip(x, 0, 255) / 255
  x = torch.from_numpy(x).float()
  x = x.permute(2,0,1)
  return x

def convert_to_ts_01(x, **kwargs):
  x = torch.from_numpy(x).float()
  x = x.permute(2,0,1)
  return x


class ImageDataset(Dataset):
  """
  Dataset for the evaluation of images
  """
  def __init__(self, data: str, labels: str, delete_segmentation: bool, eval_train_augment_rate: float, img_size: int, target: str, train: bool, live_loading: bool, task: str,
               dataset_name:str='dvm', augmentation_speedup:bool=False) -> None:
    super(ImageDataset, self).__init__()
    self.train = train
    self.eval_train_augment_rate = eval_train_augment_rate
    self.live_loading = live_loading
    self.task = task

    self.dataset_name = dataset_name
    self.augmentation_speedup = augmentation_speedup

    self.data = torch.load(data)
    self.labels = torch.load(labels)

    if delete_segmentation:
      for im in self.data:
        im[0,:,:] = 0

    self.transform_train = grab_hard_eval_image_augmentations(img_size, target, augmentation_speedup=self.augmentation_speedup)

    if self.augmentation_speedup:
      if self.dataset_name == 'dvm':
        self.transform_val = A.Compose([
          A.Resize(height=img_size, width=img_size),
          A.Lambda(name='convert2tensor', image=convert_to_ts)
        ])
        print('Using dvm transform for val transform in ImageDataset')
      elif self.dataset_name == 'cardiac':
        self.transform_val = A.Compose([
          A.Resize(height=img_size, width=img_size),
          A.Lambda(name='convert2tensor', image=convert_to_ts_01)
        ])
        print('Using cardiac transform for val transform in ImageDataset')
      elif self.dataset_name == 'adni':
        self.transform_val= A.Compose([
          A.Resize(height=img_size, width=img_size),
          A.Lambda(name='convert2tensor', image=convert_to_ts_01)
        ])
        print(f'Using adni transform for default transform in ContrastiveReconstructImagingAndTabularDataset')
      else:
        raise ValueError('Only support dvm and cardiac/adni datasets')
    else:
      resize_fn = partial(resize_tensor_fixed, size=(img_size, img_size))
      self.transform_val = transforms.Compose([
        transforms.Lambda(resize_fn),
        transforms.Lambda(convert_to_float)
      ])



  def __getitem__(self, indx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns an image for evaluation purposes.
    If training, has {eval_train_augment_rate} chance of being augmented.
    If val, never augmented.
    """
    im = self.data[indx]
    if self.live_loading:
      if self.augmentation_speedup:
        im = np.load(im[:-4]+'.npy', allow_pickle=True)
      else:
        im = read_image(im)
        im = im[:3, :, :] if im.shape[0] > 3 else im
        im = im / 255 if self.dataset_name == 'dvm' else im

    if self.train and (random.random() <= self.eval_train_augment_rate):
      im = self.transform_train(image=im)['image'] if self.augmentation_speedup else self.transform_train(im)
    else:
      im = self.transform_val(image=im)['image'] if self.augmentation_speedup else self.transform_val(im)
    
    label = self.labels[indx]
    # 将标签转换为一个 float 类型的 Tensor
    label_tensor = torch.tensor(label, dtype=torch.long) # 注意：分类任务通常使用 torch.long
    
    return im, label_tensor

  def __len__(self) -> int:
    return len(self.labels)
