
import glob
import numpy
import nibabel as nib
import numpy as np
from multiprocessing import Pool
from mrclass_resnet.utils import smallest

def split_slices(i):
    
    new_dir = ''
    image = nib.load(i)
    imgs_nib = nib.as_closest_canonical(image)  
    image = imgs_nib.get_fdata()
    
    x,y,z = image.shape
    s = smallest(x,y,z)
    if len(image.shape) > 3:
        image = image[:,:,:,0]  
    if s==x:
        ms = image.shape[0]//2
        for j in range(ms-3, ms+3):
            img = nib.Nifti1Image(image[j,:,:], np.eye(4))
            new_name =  new_dir + i.split('/')[-2] +'/'+ i.split('/')[-1].split('.nii.gz')[0]+'_c'+str(j)+'.nii.gz'            
            nib.save(img, new_name)            
    elif s==y:
        ms = image.shape[1]//2
        for j in range(ms-3, ms+3):
            img = nib.Nifti1Image(image[:,j,:], np.eye(4))
            new_name = new_dir  + i.split('/')[-2] +'/'+ i.split('/')[-1].split('.nii.gz')[0]+'_t'+str(j)+'.nii.gz'
            nib.save(img, new_name)
    else:
        ms = image.shape[2]//2
        for j in range(ms-3, ms+3):
            img = nib.Nifti1Image(image[:,:,j], np.eye(4))
            new_name = new_dir  + i.split('/')[-2] +'/'+ i.split('/')[-1].split('.nii.gz')[0]+'_t'+str(j)+'.nii.gz'
            nib.save(img, new_name)               

def run(args):
    
    image = args
    try:
       split_slices(image)
    except:
        print('FAILED!')

root_dir = ''
images = glob.glob(root_dir +'/*.nii.gz')
failed = []
p = Pool(processes=12)
p.map(run, images)
p.close()
p.join()
