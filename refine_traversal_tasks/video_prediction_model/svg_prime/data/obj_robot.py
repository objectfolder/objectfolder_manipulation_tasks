import socket
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset
import h5py
from torchvision import transforms
import os
import torch
from torch.utils.data import DataLoader
import cv2
from tqdm import tqdm

import time
class BaseDataset(Dataset):
    
    """Data Handler that creates objfolder robotic task data."""

    def __init__(self, data_root, train, seq_len=20, image_size=72, croped_size=64, ordered = False):
        super().__init__()
        if data_root[-3:] == 'txt':
            with open(data_root, "r") as f:
                self.path = f.read().splitlines()
        else:
            self.path = [data_root]
        self.seq_len = seq_len
        self.image_size = image_size
        self.croped_size = croped_size
        self.step_length = 0.1
        self.seed_is_set = False # multi threaded loading
        self.ordered = ordered

    def set_seed(self, seed):
        raise NotImplementedError
          
    def __len__(self):
        raise NotImplementedError


    def __getitem__(self, index):
        raise NotImplementedError


class TactileRefine(BaseDataset):
    def __init__(self, data_root, train=True, seq_len=14, image_size=72, croped_size = 64, modality="v"):
        super().__init__(data_root, train, seq_len, image_size, croped_size)
        self.modality = modality
        self.action_chunks = None
        self.frame_chunks = None
        self.action_chunks = None
        for d in self.path:
            data = h5py.File(os.path.join(d, "data.hdf5"), 'r')
            if "t" in self.modality:
                self.frame_chunks = data["tactile"][:] if self.frame_chunks is None else np.vstack((self.frame_chunks, data["tactile"][:]))
            elif "v" in self.modality:
                self.frame_chunks = data["visionRGB"][:] if self.frame_chunks is None else np.vstack((self.frame_chunks, data["visionRGB"][:]))
            self.action_chunks = data['action'][:] if self.action_chunks is None else np.vstack((self.action_chunks, data["action"][:]))
        self.d = 0
        # TODO: based on the generated dataset, write the data loading
        self.N = self.frame_chunks.shape[0]
        if train: 
            self.trainTransform = transforms.Compose(
                [
                    transforms.Resize(self.image_size),
                    transforms.RandomCrop(self.croped_size),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
                ]
            )
        else:
            self.trainTransform = transforms.Compose(
                [
                    transforms.Resize(self.image_size),
                    transforms.CenterCrop(self.croped_size),
                ]
            )
        # cv2.imshow("1", (self.frame_chunks[-1].transpose(0,1,3,4,2).numpy()[0][-4] * 255).astype(np.uint8))
        # cv2.waitKey(1)
        
    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)
          
    def __len__(self):
        return self.N

    def __getitem__(self, index):
        self.set_seed(index)
        if self.ordered:
            video = self.frame_chunks[self.d]
            action = self.action_chunks[self.d]
            if self.d == self.N - 1:
                self.d = 0
            else:
                self.d +=1
        else:
            tmp = np.random.randint(self.N)
            video = self.frame_chunks[tmp]
            action = self.action_chunks[tmp]
        video = torch.from_numpy(video).permute(0, 3, 1, 2).type(torch.FloatTensor)
        action = torch.from_numpy(action).type(torch.FloatTensor)
        
        start = np.random.randint(0, video.shape[0] - self.seq_len - 1)
        end = start + self.seq_len
        video = video[start:end]        
        action = action[start:(end-1)]        
        
        transformed_video = self.trainTransform(video/255.0)
        return transformed_video, action

class SurfaceTraversal(BaseDataset):
    def __init__(self, data_root, train=True, seq_len=14, image_size=72,croped_size = 64,  modality="v"):
        super().__init__(data_root, train, seq_len, image_size, croped_size)
        self.modality = modality
        self.action_chunks = None
        self.frame_chunks = None
        self.action_chunks = None
        for d in tqdm(self.path):
            data = h5py.File(os.path.join(d, "data.hdf5"), 'r')
            if "t" in self.modality:
                self.frame_chunks = data["tactile"][:] if self.frame_chunks is None else np.vstack((self.frame_chunks, data["tactile"][:]))
            elif "v" in self.modality:
                self.frame_chunks = data["visionRGB"][:] if self.frame_chunks is None else np.vstack((self.frame_chunks, data["visionRGB"][:]))
            self.action_chunks = data['action'][:] if self.action_chunks is None else np.vstack((self.action_chunks, data["action"][:]))
        self.d = 0
        # TODO: based on the generated dataset, write the data loading
        self.N = self.frame_chunks.shape[0]
        if train: 
            self.trainTransform = transforms.Compose(
                [
                    transforms.Resize(self.image_size),
                    transforms.RandomCrop(self.croped_size),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
                ]
            )
        else:
            self.trainTransform = transforms.Compose(
                [
                    transforms.Resize(self.image_size),
                    transforms.CenterCrop(self.croped_size),
                ]
            )
        # cv2.imshow("1", (self.frame_chunks[-1].transpose(0,1,3,4,2).numpy()[0][-4] * 255).astype(np.uint8))
        # cv2.waitKey(1)
        
    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)
          
    def __len__(self):
        return self.N

    def __getitem__(self, index):
        self.set_seed(index)
        if self.ordered:
            video = self.frame_chunks[self.d]
            action = self.action_chunks[self.d]
            if self.d == self.N - 1:
                self.d = 0
            else:
                self.d +=1
        else:
            tmp = np.random.randint(self.N)
            video = self.frame_chunks[tmp]
            action = self.action_chunks[tmp]
        video = torch.from_numpy(video).permute(0, 3, 1, 2).type(torch.FloatTensor)
        action = torch.from_numpy(action).type(torch.FloatTensor)
        
        start = np.random.randint(0, video.shape[0] - self.seq_len - 1)
        end = start + self.seq_len
        video = video[start:end]        
        action = action[start:(end-1)]        
        
        transformed_video = self.trainTransform(video/255.0)
        return transformed_video, action
    
def main():
    path = "/viscam/projects/objectfolder_benchmark/Robot_related/Taxim_obj/experiments/data/surface/train_data.txt"
    dataset = SurfaceTraversal(path)
    loader = DataLoader(dataset)
    for d in loader:
        v, a = d
        # for i in range(14):
        #     cv2.imshow("1", (v.permute(0,1,3,4,2).numpy()[0][i] * 255).astype(np.uint8))
        #     cv2.waitKey(1000)
        #     print(a)
        
if __name__ == "__main__":
    main()