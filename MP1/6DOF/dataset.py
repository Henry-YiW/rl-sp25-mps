import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
import json
from PIL import Image
from tqdm import tqdm
from absl import flags
import logging
from io import BytesIO

FLAGS = flags.FLAGS


class YCBVDataset(Dataset):
    def __init__(self, data_dir='./data/ycbv/v1/', split='train', transform=None, 
                 preload_images=False):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.data = json.load(open(os.path.join(self.data_dir, 
                                                f'./ycbv_{split}.json')))
        for k, v in self.data.items():
            v['key_name'] = k
        self.data = list(self.data.values())
        self.num_classes = 13
        self.preload_images = preload_images
        if self.preload_images:
            self._preload_images()

    def _preload_images(self,):
        logging.info(f'Preloading {self.split} Images Strings into memory')
        self.image_strings = {}
        for i in range(len(self.data)):
            img_path = self._get_image_path(i)
            if img_path not in self.image_strings.keys():
                image_string = open(img_path, 'rb').read()
                self.image_strings[img_path] = image_string
        logging.info(f'Finished Preloading {self.split} Images.')
    
    def __len__(self):
        return len(self.data)
    
    def _get_image(self, idx):
        if self.preload_images:
            image_string = self.image_strings[self._get_image_path(idx)]
        else:
            img_path = self._get_image_path(idx)
            image_string = open(img_path, 'rb').read()
        img = Image.open(BytesIO(image_string)).convert('RGB') 
        return img
    
    def _get_image_path(self, idx):
        img_name = self.data[idx]['img_name']
        return os.path.join(self.data_dir, 'rgb', img_name)

    def crop_batched_tensor(self,image_tensor, bbox):
        """
        Crops a batched image tensor based on the bounding box.

        Args:
            image_tensor (torch.Tensor): Image tensor of shape (B, C, H, W).
            bbox (tuple): Bounding box as (x, y, width, height).

        Returns:
            torch.Tensor: Cropped image tensor of shape (B, C, h, w).
        """
        x, y, w, h = bbox
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)
        # logging.error(f'x, y, w, h, {x}, {y}, {w}, {h}, bbox')
        return image_tensor[:, y:y+h, x:x+w]  # Keep batch & channels
        
    def pad_images_torchvision(self, cropped_image, target_size=(480, 640), pad_value=0):
        """
        Pads images to a fixed size without distorting aspect ratio using torchvision.transforms.Pad.

        Args:
            cropped_images (list of torch.Tensor): List of cropped images (C, H, W).
            target_size (tuple): Desired (H, W) after padding.
            pad_value (int): Padding color (0 for black, 255 for white).

        Returns:
            torch.Tensor: Batched padded images of shape (B, C, target_H, target_W).
        """

        # logging.error(f'target_size, {target_size}, cropped_image, {cropped_image.shape}')
        target_h, target_w = target_size
        c, h, w = cropped_image.shape  # Get original image size

        # Compute required padding
        pad_h = max(target_h - h, 0)
        pad_w = max(target_w - w, 0)

        # Compute symmetric padding (left, right, top, bottom)
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        # Apply padding
        pad_transform = transforms.Pad((pad_left, pad_top, pad_right, pad_bottom), fill=pad_value)
        padded_img = pad_transform(cropped_image)  # Apply padding transformation


        return padded_img  # Convert list to batch tensor

    def __getitem__(self, idx, crop=True):
        item = self.data[idx].copy()
        obj_class = item['obj_id']
        R = torch.tensor(item['cam_R_m2c'], dtype=torch.float32).reshape(3, 3)
        t = torch.tensor(item['cam_t_m2c'], dtype=torch.float32).reshape(3, 1) / 1000.
        img = self._get_image(idx)
        bbox = torch.tensor(item['bbox_visib'], dtype=torch.float32).reshape(4)
        if self.transform:
            img = self.transform(img)
        original_size = img.shape[1:]
        if crop:
            img = self.crop_batched_tensor(img, bbox)
            img = self.pad_images_torchvision(img, original_size)
        return img, bbox, obj_class, R, t, item['key_name']


