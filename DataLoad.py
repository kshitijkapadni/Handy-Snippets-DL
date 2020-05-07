from torch.utils.data import Dataset
from PIL import Image
import torch
from imutils import paths
import random
import numpy as np

class DataLoad(Dataset):
    def __init__(self,path):
    ############# Uncomment this if You dont have validation set ###############
    # def __init__(self,path,flag):
    ############################################################################
        self.imagepath = list(paths.list_files(path))
        # self.transforms = transforms
        random.shuffle(self.imagepath)
        ############# Uncomment this if You dont have validation set ###############
        # if flag == 'train':
        #     print(int(len(self.imagepath)))
        #     self.imagepath = self.imagepath[:int(-len(self.imagepath)*0.1)]
        # elif flag == 'val':
        #     self.imagepath = self.imagepath[int(-len(self.imagepath)*0.1):]
        ############################################################################

    def __len__(self):
        return len(self.imagepath)

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_name = self.imagepath[idx]
        image = Image.open(image_name)
        image = image.resize((50,50),resample = Image.LANCZOS)
        image = np.array(image)
        image = image.reshape((3,50,50))
        label = None
        if image_name.split('\\')[-1].split('.')[0] == 'cat':
            label = [0,1]
        elif image_name.split('\\')[-1].split('.')[0] == 'dog':
            label = [1,0]

        image = torch.tensor(image).type(torch.FloatTensor)
        label = torch.tensor(label).type(torch.FloatTensor)
        return image,label

########### Uncomment this to check if your Custom DataLoader Works ###############
# train = DataLoad("Path to images Folder")

# for i in train:
#     print(i)
#     break
####################################################################################