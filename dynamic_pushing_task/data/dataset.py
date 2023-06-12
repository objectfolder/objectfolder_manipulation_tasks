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
import time
import random
from tqdm import tqdm
from PIL import Image
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


class DynamicPush(BaseDataset):
    def __init__(self, data_root, train=True, seq_len=14, image_size=72, croped_size = 64, modality="v"):
        super().__init__(data_root, train, seq_len, image_size, croped_size)
        self.modality = modality
        self.action_chunks = None
        self.tactile_chunks = None
        self.vision_chunks = None
        self.pose_chunks = None
        for d in tqdm(self.path):
            data = h5py.File(os.path.join(d, "data.hdf5"), 'r')
            if "t" in self.modality:
                self.tactile_chunks = np.expand_dims(data["tactile"][:, 13:43], axis=0) if self.tactile_chunks is None else np.vstack((self.tactile_chunks, np.expand_dims(data["tactile"][:, 13:43], axis=0)))
            if "v" in self.modality:
                self.vision_chunks = np.expand_dims(data["visionRGB"][:], axis=0)[:, :, ::2] if self.vision_chunks is None else np.vstack((self.vision_chunks, np.expand_dims(data["visionRGB"][:], axis=0)[:, :, ::2]))
            self.action_chunks = np.expand_dims(data['action'][:], axis=0) if self.action_chunks is None else np.vstack((self.action_chunks, np.expand_dims(data['action'][:], axis=0)))
            self.pose_chunks = np.expand_dims(data['obj_pose'][:], axis=0)[:, :, ::2] if self.pose_chunks is None else np.vstack((self.pose_chunks, np.expand_dims(data['obj_pose'][:], axis=0)[:, :, ::2]))
        self.trail_num = self.action_chunks.shape[0]
        self.traj_num = self.action_chunks.shape[1]
        self.N = self.trail_num * self.traj_num
        if train: 
            self.trainTransform = transforms.Compose(
                [
                    transforms.Resize(self.image_size),
                    transforms.RandomCrop(self.croped_size),
                ]
            )
        else:
            self.trainTransform = transforms.Compose(
                [
                    transforms.Resize(self.image_size),
                    transforms.CenterCrop(self.croped_size),
                ]
            )
        
    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)
          
    def __len__(self):
        return self.N

    def __getitem__(self, index):
        self.set_seed(index)
        obj_idx = np.random.randint(self.trail_num)
        traj_idx = random.sample(range(self.traj_num), 4)
        feat_list_1 = []
        feat_list_2 = []
        feat_list_3 = []
        if "t" in self.modality:
            # tac shape [30 * 3, 64, 64]
            tac_1 = torch.from_numpy(self.tactile_chunks[obj_idx][traj_idx[0]]).permute(0, 3, 1, 2).type(torch.FloatTensor)
            tac_2 = torch.from_numpy(self.tactile_chunks[obj_idx][traj_idx[1]]).permute(0, 3, 1, 2).type(torch.FloatTensor)
            tac_3 = torch.from_numpy(self.tactile_chunks[obj_idx][traj_idx[2]]).permute(0, 3, 1, 2).type(torch.FloatTensor)
            tac_1 = self.trainTransform(tac_1).reshape(-1, self.croped_size, self.croped_size)
            tac_2 = self.trainTransform(tac_2).reshape(-1, self.croped_size, self.croped_size)
            tac_3 = self.trainTransform(tac_3).reshape(-1, self.croped_size, self.croped_size)
            feat_list_1.append(tac_1)
            feat_list_2.append(tac_2)
            feat_list_3.append(tac_3)
        if 'v' in self.modality:
            # vis shape (11*3, 64, 64]
            vis_1 = torch.from_numpy(self.vision_chunks[obj_idx][traj_idx[0]]).permute(0, 3, 1, 2).type(torch.FloatTensor)
            vis_2 = torch.from_numpy(self.vision_chunks[obj_idx][traj_idx[1]]).permute(0, 3, 1, 2).type(torch.FloatTensor)
            vis_3 = torch.from_numpy(self.vision_chunks[obj_idx][traj_idx[2]]).permute(0, 3, 1, 2).type(torch.FloatTensor)
            vis_1 = self.trainTransform(vis_1).reshape(-1, self.croped_size, self.croped_size)
            vis_2 = self.trainTransform(vis_2).reshape(-1, self.croped_size, self.croped_size)
            vis_3 = self.trainTransform(vis_3).reshape(-1, self.croped_size, self.croped_size)
            feat_list_1.append(vis_1)
            feat_list_2.append(vis_2)
            feat_list_3.append(vis_3)
        
        # (2, 64, 64)
        act_1 = torch.from_numpy(self.action_chunks[obj_idx][traj_idx[0]][0]).type(torch.FloatTensor).squeeze(0).repeat(64,64,1).permute(2,0,1)
        act_2 = torch.from_numpy(self.action_chunks[obj_idx][traj_idx[1]][0]).type(torch.FloatTensor).squeeze(0).repeat(64,64,1).permute(2,0,1)
        act_3 = torch.from_numpy(self.action_chunks[obj_idx][traj_idx[2]][0]).type(torch.FloatTensor).squeeze(0).repeat(64,64,1).permute(2,0,1)
        feat_list_1.append(act_1)
        feat_list_2.append(act_2)
        feat_list_3.append(act_3)
        # action (2)
        act_in = torch.from_numpy(self.action_chunks[obj_idx][traj_idx[3]][-1]).type(torch.FloatTensor).squeeze(0)
        # (7*11, 64, 64)  
        pose_1 = torch.from_numpy(self.pose_chunks[obj_idx][traj_idx[0]]).type(torch.FloatTensor).view(-1).repeat(64,64,1).permute(2,0,1)
        pose_2 = torch.from_numpy(self.pose_chunks[obj_idx][traj_idx[1]]).type(torch.FloatTensor).view(-1).repeat(64,64,1).permute(2,0,1)
        pose_3 = torch.from_numpy(self.pose_chunks[obj_idx][traj_idx[2]]).type(torch.FloatTensor).view(-1).repeat(64,64,1).permute(2,0,1)
        feat_list_1.append(pose_1)
        feat_list_2.append(pose_2)
        feat_list_3.append(pose_3)
        # initial/final pose
        pose_4_init = torch.from_numpy(self.pose_chunks[obj_idx][traj_idx[3]][0]).type(torch.FloatTensor).squeeze(0)
        pose_4_final = torch.from_numpy(self.pose_chunks[obj_idx][traj_idx[3]][-1]).type(torch.FloatTensor).squeeze(0)[:2]
        
        x_1 = torch.cat(feat_list_1, dim=0)
        x_2 = torch.cat(feat_list_2, dim=0)
        x_3 = torch.cat(feat_list_3, dim=0)
        return x_1, x_2, x_3, act_in, pose_4_init, pose_4_final
    
def main():
    path = "/viscam/projects/objectfolder_benchmark/Robot_related/Taxim_obj/dynamic_pushing/config/debug.txt"
    dataset = DynamicPush(path)
    testLoader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=12, pin_memory=True
    )
    for d in testLoader:
        x_1, x_2, x_3, act_in, pose_4_init, pose_4_final = d
        print(pose_4_final)
        
if __name__ == "__main__":
    main()