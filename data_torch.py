import os
import json
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

class LIVCellDataset(Dataset):
    def __init__(self, images_dir, annotation_file, img_size=512, transform=None):
        
        self.images_dir = images_dir
        self.img_size = img_size
        self.transform = transform

        with open(annotation_file, 'r') as f:
            self.data = json.load(f)

        self.image_id_to_filename = {img['id']: img['file_name'] for img in self.data['images']}
        self.annotations_by_image_id = {}
        for annotation in self.data['annotations']:
            if annotation['image_id'] not in self.annotations_by_image_id:
                self.annotations_by_image_id[annotation['image_id']] = []
            self.annotations_by_image_id[annotation['image_id']].append(annotation)

    def __len__(self):
        return len(self.data['images'])

    def __getitem__(self, idx):

        image_info = self.data['images'][idx]
        image_id = image_info['id']
        image_path = os.path.join(self.images_dir, image_info['file_name'])

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (self.img_size, self.img_size))

        image = image.astype(np.float32) / 255.0

        mask = np.zeros((image_info['height'], image_info['width']), dtype=np.float32)
        annotations = self.annotations_by_image_id.get(image_id, [])
        for annotation in annotations:
            segmentation = annotation['segmentation']
            for polygon in segmentation:
                polygon = np.array(polygon).reshape(-1, 2)
                cv2.fillPoly(mask, [polygon.astype(np.int32)], 1)

        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"].unsqueeze(0)
        else:
            image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  
            mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        return image, mask