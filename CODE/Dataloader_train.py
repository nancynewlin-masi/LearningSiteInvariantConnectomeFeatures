import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import random

class TrainingDataset(Dataset):
        def __init__(self):
                directory = "/home-local/LearningConnectomeInvariance/Data/Training_Data_freesurfer/"

                subj_list = glob.glob(directory + "*")
                #print(subj_list)
                self.data = []
                for subject_path in subj_list:
                        subject = subject_path.split("/")[-1]
                        img_path = glob.glob(directory + "/" + subject +"/*CONNECTOME*NUMSTREAM*.csv")
                        label_path = glob.glob(directory + "/" + subject + "/graphmeasures.json")
                        site_path = glob.glob(directory + "/" + subject + "/*SITE.csv")
                        age_path = glob.glob(directory + "/" + subject + "/*AGE.csv")
                        sex_path = glob.glob(directory + "/" + subject + "/*SEX.csv")
                        self.data.append([img_path, label_path, site_path, age_path, sex_path])

                self.img_dim = (84, 84)

        def __len__(self):
                return len(self.data)

        def __getitem__(self, idx):
                img_path, label_path, site_path, age_path, sex_path = self.data[idx]
                #print(img_path)
                img = pd.read_csv(img_path[0], sep=',', header=None)
                img = img.to_numpy()
                img = img/100000
                #np.fill_diagonal(img, 0)
                img = np.float32(img)
                

                label = pd.read_json(label_path[0])
                label = label.to_numpy()

                site = pd.read_csv(site_path[0], sep=',', header=None)
                site = site.to_numpy()
                site = np.float32(site)

                age = pd.read_csv(age_path[0], sep=',', header=None)
                age = age.to_numpy()
                age = np.float32(age)

                sex = pd.read_csv(sex_path[0], sep=',', header=None)
                sex = sex.to_numpy()
                sex = np.int32(sex)
                #sex=sex-1
                sex_onehot = np.eye(2)[sex - 1]
                sex_onehot = np.float32(sex_onehot)
                #print(sex[0:5], sex_onehot[0:5,:])
        
                img_tensor = torch.from_numpy(img)
                site_tensor = torch.tensor(site)
                age_tensor = torch.tensor(age)
                sex_tensor = torch.tensor(sex)
                sex_tensor = torch.tensor(sex_onehot)
                
                return img_tensor, label, site_tensor, age_tensor, sex_tensor, img_path

if __name__ == "__main__":
    dataset = TrainingDataset()
    data_loader = DataLoader(dataset, batch_size=1500, shuffle=True)
    for imgs, predictionlabels, sitelabels, contrastivegroups, img_paths in data_loader:
        print("Batch of images has shape: ", imgs.shape)
        print("Batch of labels has shape: ", sitelabels.shape)
        print("Batch of labels has shape: ", predictionlabels.shape)
