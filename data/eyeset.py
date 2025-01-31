# -*- encoding:utf-8 -*-
import os
import glob
import numpy as np
import math
import random
from PIL import Image
from albumentations import (
    HorizontalFlip, VerticalFlip, Transpose, RandomRotate90, PadIfNeeded, 
    OneOf, Compose, CLAHE, RandomGamma, CropNonEmptyMaskIfExists
)
import torch
from torch.utils.data import Dataset

class EyeSetResource:
    def __init__(self, folder, dbname='stare', split_ratios=(0.7, 0.2, 0.1)):
        """
        Initialize EyeSetResource to handle dataset loading and splitting.

        Args:
            folder (str): Path to the dataset directory.
            dbname (str): Name of the dataset (default: 'stare').
            split_ratios (tuple): Ratios for train, validation, and test splits.
        """
        self.folder = folder
        self.dbname = dbname  # Identifier for the dataset
        self.imgs, self.labs = self.get_data()
        self.split_data(split_ratios)

    def get_data(self):
        """Load image and label file paths."""
        images_dir = os.path.join(self.folder, 'images')
        labels_dir = os.path.join(self.folder, '1st_manual')

        imgs = sorted(glob.glob(os.path.join(images_dir, '*.tif')))  # Assuming .ppm format for STARE
        labs = sorted(glob.glob(os.path.join(labels_dir, '*.gif')))  # Assuming .ppm format for labels
        
        if len(imgs) != len(labs):
            raise ValueError("Mismatch between images and labels count!")
        
        return imgs, labs
	
    def split_data(self, split_ratios):
        """Split the dataset into train, val, and test sets."""
        indices = list(range(len(self.imgs)))
        random.shuffle(indices)

        train_end = int(len(indices) * split_ratios[0])
        val_end = train_end + int(len(indices) * split_ratios[1])

        self.imgs_train = [self.imgs[i] for i in indices[:train_end]]
        self.labs_train = [self.labs[i] for i in indices[:train_end]]

        self.imgs_val = [self.imgs[i] for i in indices[train_end:val_end]]
        self.labs_val = [self.labs[i] for i in indices[train_end:val_end]]

        self.imgs_test = [self.imgs[i] for i in indices[val_end:]]
        self.labs_test = [self.labs[i] for i in indices[val_end:]]
	
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from albumentations import Compose, HorizontalFlip, RandomRotate90, CLAHE, RandomGamma, CropNonEmptyMaskIfExists



import os
from PIL import Image
import numpy as np
import torch
from albumentations import Compose, HorizontalFlip, RandomRotate90, CLAHE, RandomGamma, CropNonEmptyMaskIfExists


class EyeSetGenerator(Dataset):
    # Existing code...

    def parse(self, imgs):
        """
        Parse a batch of images and return their corresponding images, labels, FOV, and auxiliary data.
        
        Args:
            imgs (list or Tensor): A batch of images (file paths or tensors).

        Returns:
            tuple: (images, labels, fov, aux).
        """
        img_paths = []
        lab_paths = []
        fov = []  # Placeholder for FOV data
        aux = []  # Placeholder for auxiliary data

        for img_path in imgs:
            if isinstance(img_path, str):
                img = np.array(Image.open(img_path))
                label_path = img_path.replace("images", "1st_manual").replace("_training.tif", "_manual1.gif")
                if not os.path.exists(label_path):
                    raise FileNotFoundError(f"Label file not found for {img_path}")
                label = np.array(Image.open(label_path))
            else:
                raise ValueError(f"Unsupported image format: {type(img_path)}")
            
            img_paths.append(img)
            lab_paths.append(label)
            fov.append(None)  # Replace with actual FOV logic if needed
            aux.append(None)  # Replace with actual auxiliary data logic if needed

        return img_paths, lab_paths, fov, aux

    def __init__(self, folder, datasize=128, dbname='stare', split_ratios=(0.7, 0.2, 0.1)):
        """
        Dataset generator for training, validation, and testing.

        Args:
            folder (str): Path to the dataset directory.
            datasize (int): Size for cropped images.
            dbname (str): Name of the dataset (default: 'stare').
            split_ratios (tuple): Ratios for train, validation, and test splits.
        """
        self.datasize = datasize
        self.mode = 'train'
        self.dbname = dbname

        # Load and split the dataset
        self.img_paths, self.label_paths = self._load_dataset(folder)
        self.imgs_train, self.labs_train, self.imgs_val, self.labs_val, self.imgs_test, self.labs_test = self._split_data(
            self.img_paths, self.label_paths, split_ratios)

        # Define transformations for different modes
        self.transforms = {
            'train': Compose([
                HorizontalFlip(p=0.7),
                RandomRotate90(p=0.7),
                CLAHE(p=1),
                RandomGamma(p=1),
                CropNonEmptyMaskIfExists(p=1, height=self.datasize, width=self.datasize)
            ]),
            'val': Compose([
                CLAHE(p=1),
                RandomGamma(p=1)
            ]),
            'test': Compose([
                CLAHE(p=1),
                RandomGamma(p=1)
            ])
        }

    def _load_dataset(self, folder):
        """
        Load all image and label file paths, ensuring strict correspondence.

        Args:
            folder (str): Path to the dataset directory.

        Returns:
            tuple: (image_paths, label_paths)
        """
        image_dir = os.path.join(folder, "images")
        label_dir = os.path.join(folder, "1st_manual")

        image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.tif'))])
        label_files = sorted([f for f in os.listdir(label_dir) if f.endswith(('.gif'))])
        # print(image_files,label_files)
        # print(label_dir)
        image_paths, label_paths = [], []
        for img_file in image_files:
            base_name = os.path.splitext(img_file)[0][:2]
            # print(base_name) # Get the base name without extension
            label_file = f"{base_name}_manual1.gif" 

            label_path = os.path.join(label_dir, label_file)
            img_path = os.path.join(image_dir, img_file)

            if os.path.exists(label_path):
                image_paths.append(img_path)
                label_paths.append(label_path)
            else:
                print(f"Warningg error: No label found for {label_path} . Skipping.....")
        # print(len(image_paths),len(label_paths))
        return image_paths, label_paths


    def _split_data(self, img_paths, label_paths, split_ratios):
        """
        Split the dataset into train, validation, and test sets.

        Args:
            img_paths (list): List of image paths.
            label_paths (list): List of label paths.
            split_ratios (tuple): Ratios for train, validation, and test splits.

        Returns:
            tuple: Split datasets (train_imgs, train_labels, val_imgs, val_labels, test_imgs, test_labels)
        """
        total = len(img_paths)
        train_end = int(total * split_ratios[0])
        val_end = train_end + int(total * split_ratios[1])

        imgs_train = img_paths[:train_end]
        labs_train = label_paths[:train_end]
        imgs_val = img_paths[train_end:val_end]
        labs_val = label_paths[train_end:val_end]
        imgs_test = img_paths[val_end:]
        labs_test = label_paths[val_end:]

        return imgs_train, labs_train, imgs_val, labs_val, imgs_test, labs_test

    def set_mode(self, mode):
        """Set the mode for the dataset."""
        if mode not in ['train', 'val', 'test']:
            raise ValueError("Mode must be 'train', 'val', or 'test'.")
        self.mode = mode

    def __len__(self):
        """Return the size of the dataset for the current mode."""
        if self.mode == 'train':
            return len(self.imgs_train)
        elif self.mode == 'val':
            return len(self.imgs_val)
        elif self.mode == 'test':
            return len(self.imgs_test)

    def __getitem__(self, idx):
        """Get an item (image and label) from the dataset."""
        if self.mode == 'train':
            img_path = self.imgs_train[idx]
            lab_path = self.labs_train[idx]
        elif self.mode == 'val':
            img_path = self.imgs_val[idx]
            lab_path = self.labs_val[idx]
        elif self.mode == 'test':
            img_path = self.imgs_test[idx]
            lab_path = self.labs_test[idx]

        img = np.array(Image.open(img_path))
        lab = np.array(Image.open(lab_path))

        # Apply transformations based on the mode
        transformed = self.transforms[self.mode](image=img, mask=lab)
        img = transformed['image']
        lab = transformed['mask']

        # Convert image and label to tensors
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).div(255)
        lab = torch.tensor(lab, dtype=torch.float32).unsqueeze(0).div(255)

        return img, lab
    def valSet(self):
        self.set_mode('val')  # Switch to validation mode
        return self




# Main Function
# if __name__ == '__main__':
#     dataset_path = '/home/umerfarooq/Downloads/STARE/training/'
#     dataset = EyeSetGenerator(folder=dataset_path, datasize=128)

#     # Example usage
#     dataset.set_mode('train')
#     train_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

#     for img, lab in train_loader:
#         print(f"Image shape: {img.shape}, Label shape: {lab.shape}")
#         break
