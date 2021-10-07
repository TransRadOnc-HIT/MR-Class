  
from mrclass_resnet.utils import get_label,extract_middleSlice, crop_background3D
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
import os
import albumentations as A
import torch

class MRClassifierDataset(Dataset):

    def __init__(self,list_images='', transform=None, augmentations=None, 
                 class_names = '', run_3d = False, scan = 0, 
                 remove_corrupt = False, subclasses = False,
                 parentclass = False, inference = False, spatial_size=224, 
                 nr_slices = 50, slices= None, crop = None,infer = False):
 
        self.transform = transform
        self.list_images = list_images
        self.class_names = class_names
        self.augmentations = augmentations
        self.run_3d = run_3d
        self.scan = scan
        self.remove_corrupt = remove_corrupt
        self.subclasses = subclasses
        self.parentclass = parentclass
        self.nr_slices = nr_slices
        self.spatial_size = spatial_size
        self.slices = slices
        self.crop = crop
        self.infer = infer
        
    def __len__(self):
        return len(self.list_images)
    
    def get_random(self,fa = False):
        
        if self.run_3d or fa:
            image = np.random.randn(self.spatial_size, self.spatial_size,self.nr_slices).astype('f')
        else:
            image = np.random.randn(self.spatial_size, self.spatial_size).astype('f')
        class_cat = 'random'
        return image, class_cat
    
    def __getitem__(self, idx):
        
        #modify the collate_fn from the dataloader so that it filters out None elements.
        img_name = self.list_images[idx]

        try:
            image = nib.load(img_name)
            imgs_nib = nib.as_closest_canonical(image)  
            image = np.float32(imgs_nib.get_fdata())          
            if len(image.shape)>3:
                image = image[:,:,:,0]
            if self.crop:
                image,mask = crop_background3D(image)
            #image = np.float32(load_reorient(img_name)[1])
            if self.subclasses and self.parentclass:
                class_cat = img_name.split('/')[-2].split('_')[0]
            elif self.subclasses and not self.parentclass:
                class_cat = img_name.split('/')[-2].split('_')[-1]
            else:
                class_cat = img_name.split('/')[-2]
        except:
            print ('error loading {0}'.format(img_name))
            if self.remove_corrupt and os.path.isfile(img_name):
                os.remove(img_name)
            image, class_cat = self.get_random(fa=True)
            
        if not np.any(image):
            image, class_cat = self.get_random(fa=True)
        
        if len(image.shape)>3:
            image = image[:,:,:,0]
      
        if not self.run_3d and self.infer:
            try:
                image_ms = extract_middleSlice(image, 3)  
            except:
                print ('error loading {0}'.format(img_name))
                if self.remove_corrupt and os.path.isfile(img_name):
                    os.remove(img_name)
                image, class_cat = self.get_random()
        spacing = image.shape
        if not self.infer:
            label = get_label(class_cat, self.class_names)
        else:
            label  =0
        
        if not self.infer:
            image_ms = image
        if 'ADC' not in self.class_names[0] and 'T2' not in self.class_names[0]:
            pre_transform = A.Compose([
                    A.CenterCrop(height=image_ms.shape[0]//2, width=image_ms.shape[1]//2,always_apply=True),
                    A.Resize(height=224,width=224)])    
        else:
            pre_transform = A.Compose([
                    A.Resize(height=224,width=224)])                
    
        image = pre_transform(image=image_ms)['image']  
        image -= image.mean() 
        image /= image.std() 
        image = image + abs(np.min(image))
       
        if np.sum(image==0) > 0.3*np.sum(image>0) and 'ADC' not in self.class_names[0] and 'T2' not in self.class_names[0]:
            pre_transform = A.Compose([
                    A.CenterCrop(height=image_ms.shape[0]//3, width=image_ms.shape[1]//3,always_apply=True),
                    A.Resize(height=224,width=224)])            
            image = pre_transform(image=image_ms)['image']  
            np.seterr(invalid='ignore')
            image -= image.min() 
            image /= image.max()
            image = image + abs(np.min(image))
                             
        if self.augmentations is not None:
            image = self.augmentations.augment_image(image)        

        sample = {'image': torch.from_numpy(np.float32(image.copy())).unsqueeze(dim=0), 'label': np.array(label), 'spacing': spacing, 'fn': img_name}


        return sample
