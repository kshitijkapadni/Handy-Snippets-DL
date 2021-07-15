import torch
from torch.utils.data import Dataset
from PIL import Image
from xml.etree import cElementTree as ET
import os

label_dict = {1:'class1'}

class PascalVOCDataset(Dataset):

    def __init__(self, img_folder, ann_folder,transform):
        self.img_folder = img_folder
        self.ann_folder = ann_folder
        self.transforms = transform
        self.ann = sorted(os.listdir(ann_folder))

    def __getitem__(self, i):
        # Read image
        image = Image.open(os.path.join(self.img_folder,self.ann[i].split('.')[0]+'.jpg'), mode='r')
        image = image.convert('RGB')

        # Read objects in this image (bounding boxes, labels)
        root = ET.parse(os.path.join(self.ann_folder,self.ann[i]))
        labels = []
        boxes = []
        for obj in root.findall('object'):
            labels.append(label_dict[obj.find('name').text])
            xmin = int(obj.find('bndbox/xmin').text)
            ymin = int(obj.find('bndbox/ymin').text)
            xmax = int(obj.find('bndbox/xmax').text)
            ymax = int(obj.find('bndbox/ymax').text)
            boxes.append( [ xmin, ymin, xmax, ymax] )
        boxes = torch.FloatTensor(boxes)  # (n_objects, 4)
        labels = torch.tensor(labels)  # (n_objects)
        num_objs = len(boxes)
        
        image_id = torch.tensor([i])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:,0])

        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(image, target)

        return img, target

    def __len__(self):
        return len(self.ann)
