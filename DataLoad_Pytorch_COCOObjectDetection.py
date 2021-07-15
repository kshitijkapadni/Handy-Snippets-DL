import torch
from torch.utils.data import Dataset
from PIL import Image
import json
import os

class COCODataset(Dataset):

    def __init__(self, img_folder, ann_file,transform):
        self.img_folder = img_folder
        self.transforms = transform
        dataset = json.load(open(ann_file))
      
        self.ann = []
        self.img = []
        self.labels = []

        for data in dataset['images']:
            tmp_ann = []
            tmp_labels = []
            for for_ann in dataset['annotations']:
                if data['id'] == for_ann['image_id'] and for_ann['bbox'][1] != 0:
                    tmp_ann.append([for_ann['bbox'][0],for_ann['bbox'][1],for_ann['bbox'][0]+for_ann['bbox'][2],for_ann['bbox'][1]+for_ann['bbox'][3]])
                    tmp_labels.append(for_ann['category_id']-1)

            if len(tmp_ann) != 0 :
              self.ann.append(tmp_ann)
              self.labels.append(tmp_labels)
              self.img.append(data['file_name'])



    def __getitem__(self, i):
        # Read image
        image = Image.open(os.path.join(self.img_folder,self.img[i]), mode='r')
        image = image.convert('RGB')

        # Read objects in this image (bounding boxes, labels)
        labels = self.labels[i]
        boxes = self.ann[i]
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
