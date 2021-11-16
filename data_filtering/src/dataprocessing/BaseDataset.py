import os
import cv2
import json
import torch
import numpy as np
from tqdm import tqdm
import albumentations as A


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self,
                 img_folder,
                 markup_path,
                 augs,
                 train = False,
                 ):
        super(BaseDataset, self).__init__()
        
        self.train = train
        self.augs = augs
    
        self.markup = self.load_markup(img_folder,markup_path)
        self.length = len(self.markup)
        self.print_summary()

    def print_summary(self):
        print('*'*50)
        if self.train:
            print('Training dataset')
        else:
            print('Testing dataset')
        print('Dataset length: ', self.length) 
        print('Class number: ', self.num_classes)
        print('*'*50)

    def load_markup(self, img_folder, markup_path):
        
        with open(markup_path,'r') as f:
            data = json.load(f)
        
        if self.train:
            self.num_classes = len(set(data.values()))

            translation = {i: num for num,i in enumerate(set(data.values()))}
            with open('translation_dictionary.json','w') as f:
                json.dump(translation,f,indent=4,ensure_ascii=False)
        else:
            with open('translation_dictionary.json','r') as f:
                translation = json.load(f)    
            self.num_classes = min(len(translation),len(set(data.values())))      

        return [(os.path.join(img_folder,path),translation[cl]) for path, cl in data.items() if cl in translation]

    def __getitem__(self, item):
        
        img_path, img_class = self.markup[item]
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            
        augmented = self.augs(image=img)['image']
        return augmented, torch.tensor(int(img_class),dtype=torch.long)

    def __len__(self):
        return self.length
