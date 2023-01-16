import matplotlib.pyplot as plt
import random
import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
from skimage.io import imread
from skimage.transform import rescale
import matplotlib
matplotlib.use('Agg')

# DEFAULT_PANOPTICS_DIR = '/home/getch/DATA/VAE/data'  # "../data"

class PanopticsTrainDataset(Dataset):
    def __init__(self, category, img_size, fake_dataset_size, rec_loss="xent", UNDEFINE=None,blurr='yes'):
        self.blurr = blurr
        if self.blurr=='yes':
            print('Data reading with downscaling and blurring of original image!')

        DEFAULT_PANOPTICS_DIR = '/DATA/VAE/data'
        if category == 'all':
            if os.path.isdir(DEFAULT_PANOPTICS_DIR):
                self.img_dir = [DEFAULT_PANOPTICS_DIR + '/{}/train/images/'.format(folder) for folder in os.listdir(DEFAULT_PANOPTICS_DIR)]
            else:
                self.img_dir = UNDEFINE
            self.img_files = []
            for path in self.img_dir:
                self.img_files += list(np.random.choice([os.path.join(path, img) for img in os.listdir(path) if (os.path.isfile(os.path.join(path,img)) and imread(os.path.join(path,img)).shape[0] == 256 and imread(os.path.join(path,img)).shape[1] == 256)], size=fake_dataset_size))

        else:
            if os.path.isdir(DEFAULT_PANOPTICS_DIR):
                self.img_dir = os.path.join(DEFAULT_PANOPTICS_DIR,category,"train/images/")
            #print(self.img_dir)
            else:
                self.img_dir = UNDEFINE
            self.img_files = list(
                            np.random.choice(
                                [os.path.join(self.img_dir, img)
                            for img in os.listdir(self.img_dir)
                            if (os.path.isfile(os.path.join(self.img_dir,
                            img)) and
                            imread(os.path.join(self.img_dir,
                            img)).shape[0] == 256
                            and imread(os.path.join(self.img_dir,
                            img)).shape[1] == 256)],
                            size=fake_dataset_size)
                            )
        print(f'The length of image files is {len(self.img_files)}')

        #self.gt_files = [s.replace("/images/", "/labels/") for s in self.img_files]
        #self.gt_files = [s.replace("images", "labels") for s in self.img_files]
        self.fake_dataset_size = fake_dataset_size # needed otherwise there are
        # 125000 images, and this is too much
        #selected_img_idx = np.random.choice(
        #                np.arange(self.fake_dataset_size),
        #                size=self.fake_dataset_size,
        #                replace=False
        #                )
        self.transform = transforms.Compose([
            transforms.ToTensor(),
         #   transforms.ToPILImage(),
         #   transforms.Resize(size=(img_size, img_size)),
         #   transforms.PILToTensor(),
            transforms.Lambda(lambda img: img.float())
        ]) 
        #if rec_loss == "xent" or rec_loss == "mse":
        #    self.transform.transforms.append(transforms.Lambda(lambda img: img
        #    / 255.))
        self.nb_img = len(self.img_files)

    def im_blurr(self, img, factor=16):
        img_down = rescale(img, 1/factor, multichannel=True, anti_aliasing=True)
        blur = rescale(img_down, factor, multichannel=True, anti_aliasing=True)
        assert img.shape == blur.shape
        return blur

    def __len__(self):
        return max(self.nb_img, self.fake_dataset_size)

    def __getitem__(self, index):
        # NOTE take all 4 channels (B / G / R / NIR)
        index = index % self.nb_img
        img = imread(self.img_files[index])[..., :].astype(float)
        img = (img - np.amin(img)) / (np.amax(img) - np.amin(img))

        try:
            assert img.shape[0] == 256 and img.shape[1] == 256
        except:
            raise RuntimeError("Expected (256, 256) images, got",
                img.shape[0], img.shape[1])
        #img = Image.fromarray((img * 255).astype(np.uint8))
        # some manipulation before visualization: choose BGR and reverse order
        # and renormalize
        #img_ = np.flip(img[..., :3], axis=-1)
        #img_ = (img_ - np.amin(img_)) / (np.amax(img_) - np.amin(img_))
        #plt.imshow(img_)
        #plt.show()
        if self.blurr=='yes':
            old_shape = img.shape
            img = self.im_blurr(img)
            new_shape = img.shape
            assert old_shape == new_shape, 'original and blurred image shape not matching'
        
        return self.transform(img), 1 # one if the ground truth if there is one

class PanopticsTestDataset(Dataset):
    def __init__(self, img_size, fake_dataset_size,category, rec_loss="xent", with_loc=False, UNDEFINE=None, blurr='yes', scale=16):
        DEFAULT_PANOPTICS_DIR = '/DATA/VAE/data'
        print(DEFAULT_PANOPTICS_DIR)
        #if os.path.isdir(DEFAULT_PANOPTICS_DIR):
        if category == 'all':
            self.img_dir = os.path.join(DEFAULT_PANOPTICS_DIR,"Minawao_june_2016/test/images/")    # small validation data to save test time
            print('Minawao_june_2016 as train test')
        else:
            self.img_dir = os.path.join(DEFAULT_PANOPTICS_DIR, f"{category}/test/images/")
            print(f'Test data set to dir: {category}')
        print(self.img_dir)
        # print(os.listdir(self.img_dir))
        #else:
         #   self.img_dir = UNDEFINE
        self.blurr= blurr
        self.scale=scale
        self.img_files = list(
                            np.random.choice(
                                [os.path.join(self.img_dir, img)
                            for img in os.listdir(self.img_dir)
                            if (os.path.isfile(os.path.join(self.img_dir, img))
                            and imread(os.path.join(self.img_dir,
                            img)).shape[0] == 256
                            and imread(os.path.join(self.img_dir,
                            img)).shape[1] == 256)],
                            size=fake_dataset_size)
                            )
        self.fake_dataset_size = fake_dataset_size # needed otherwise there are
        #self.gt_files = [s.replace("/images/", "/labels/") for s in self.img_files]
        self.gt_files = [s.replace("images", "labels") for s in self.img_files]   # why repetitative
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Resize(size=(img_size, img_size)),
            #transforms.PILToTensor(),
            transforms.Lambda(lambda img: img.float())
        ]) 
        self.transform_gt = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda img: img.long())
        ]) 
        #if rec_loss == "xent" or rec_loss == "mse":
        #    self.transform.transforms.append(transforms.Lambda(lambda img: img
        #    / 255.))
        self.nb_img = len(self.img_files) # recompute the size,
        # fake_dataset_size may have changed it
    def im_blurr(self, img, factor=16):
        img_down = rescale(img, 1/factor, multichannel=True, anti_aliasing=True)
        blur = rescale(img_down, factor, multichannel=True, anti_aliasing=True)
        #print(f'shape of blurred image: {blur},  dataype of blurred image: {blur.dtype}')
        assert img.shape == blur.shape
        return blur

    def __len__(self):
        return self.fake_dataset_size

    def __getitem__(self, index):
        # NOTE take all 4 channels (RGB + NIR)
        # NOTE take all 4 channels (B / G / R / NIR)
        img = imread(self.img_files[index])[..., :].astype(float)
        img = (img - np.amin(img)) / (np.amax(img) - np.amin(img))
        #img = Image.fromarray((img * 255).astype(np.uint8))
        gt = imread(self.gt_files[index]).astype(float)
        try:
            assert img.shape[0] == 256 and img.shape[1] == 256
        except:
            raise RuntimeError("Expected (256, 256) images, got",
                img.shape[0], img.shape[1])
        try:
            assert gt.shape[0] == 256 and gt.shape[1] == 256
        except:
            raise RuntimeError("Expected (256, 256) ground truths, got",
                gt.shape[0], gt.shape[1])
        try:
            assert gt.ndim == 2
        except:
            raise RuntimeError("Ground truth is not a binary image with ndim"
                + " == 2, got shape", gt.shape)
        #gt = Image.fromarray(gt)

        if self.blurr=='yes':
            img_blurr = self.im_blurr(img=img,scale=self.scale)
            return self.transform(img), self.transform_gt(gt), self.transform(img_blurr)
        else:
            return self.transform(img), self.transform_gt(gt)
