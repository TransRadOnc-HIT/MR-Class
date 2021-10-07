
import numpy as np
from medpy.filter import otsu
from scipy import ndimage
try:
    import SimpleITK as sitk
except ImportError:
    print("You need to have SimpleITK installed to run this example!")
    raise ImportError("SimpleITK not found")

from multiprocessing import Pool
import nibabel as nib

import os
from mrclass_resnet.utils import extract_middleSlice
import albumentations as A
from PIL import Image
from pyhelpers.store import save_pickle
import glob

def crop_background3D(image_np):
    
    ms = extract_middleSlice(image_np,3)
    threshold = otsu(ms)
    output_data = image_np > threshold
    output_data = output_data.astype(int)
    img = image_np*output_data
    x,y,z = np.where(img>0)
    if image_np.shape[2] > 1:
        new_image = image_np[x.min():x.max(), y.min():y.max(),z.min():z.max()]
    else:
        new_image = image_np[x.min():x.max(), y.min():y.max(), :]
    return new_image, output_data[x.min():x.max(), y.min():y.max(),z.min():z.max()]

def get_list_of_files(basedir, class_names):

    list_of_lists = []
    for glioma_type in class_names:
        current_directory = os.path.join(basedir, glioma_type)
        niftis = glob(current_directory+'/*.nii*')
        for n in niftis:
            list_of_lists.append([os.path.join(current_directory, n)])
    print("Found %d patients" % len(list_of_lists))
    return list_of_lists


def load_and_preprocess(case, patient_name, output_folder, savetofile, modality='mr'):
    """
    loads, preprocesses and saves a case
    This is what happens here:
    1) load all images and stack them to a 4d array
    2) crop to nonzero region, this removes unnecessary zero-valued regions and reduces computation time
    3) normalize the nonzero region with its mean and standard deviation
    4) save 4d tensor as numpy array. Also save metadata required to create niftis again (required for export
    of predictions)
    :param case:
    :param patient_name:
    :return:
    """

    print('Processing: {}'.format(case[0]))
    imgs_sitk = [sitk.ReadImage(i) for i in case]

    # get pixel arrays from SimpleITK images
#     imgs_npy = [sitk.GetArrayFromImage(i) for i in imgs_sitk]

    # get some metadata
    spacing = imgs_sitk[0].GetSpacing()
    spacing = np.array(spacing)[::-1]

    direction = imgs_sitk[0].GetDirection()
    origin = imgs_sitk[0].GetOrigin()
    
    imgs_nib = [nib.load(i) for i in case]
    imgs_nib = [nib.as_closest_canonical(i) for i in imgs_nib]
    imgs_npy = [i.get_fdata() for i in imgs_nib]

    original_shape = imgs_nib[0].get_fdata().shape

    tmp = []
    for im in imgs_npy:
        if len(im.shape) == 4:
            tmp.append(im[:, :, :, 0])
        elif len(im.shape) == 3:
            tmp.append(im)
        else:
            raise Exception('Image has {} dimensions'.format(len(im.shape)))
    imgs_npy = tmp

    crp = [crop_background3D(i) for i in imgs_npy]
    imgs_npy = [x[0] for x in crp]
    nonzero_masks = [x[1] for x in crp]
#     imgs_npy = [window_image(i, window_levels[class_name]) for i in imgs_npy]
#     removed_noise = [remove_noise(i) for i in imgs_npy]
#     imgs_npy = [x[0] for x in removed_noise]
#     nonzero_masks = [x[1] for x in removed_noise]
    
    # now stack the images into one 4d array, cast to float because we will get rounding problems if we don't
    imgs_npy = np.concatenate([i[None] for i in imgs_npy]).astype(np.float32)

    # now find the nonzero region and crop to that
    nonzero = [np.array(np.where(i > 0)) for i in nonzero_masks]
    nonzero = [[np.min(i, 1), np.max(i, 1)] for i in nonzero]
    nonzero = np.array([np.min([i[0] for i in nonzero], 0), np.max([i[1] for i in nonzero], 0)]).T
    # nonzero now has shape 3, 2. It contains the (min, max) coordinate of nonzero voxels for each axis

    # now crop to nonzero
    imgs_npy = imgs_npy[:,
               nonzero[0, 0] : nonzero[0, 1] + 1,
               nonzero[1, 0]: nonzero[1, 1] + 1,
               nonzero[2, 0]: nonzero[2, 1] + 1,
               ]

    image = extract_middleSlice(imgs_npy[0,:,:,:], 3)    
    pre_transform = A.Compose([
                A.CenterCrop(height=image.shape[0]//2, width=image.shape[1]//2,always_apply=True),
                A.Resize(height=224,width=224)])
    image = pre_transform(image=image)['image']  
    image -= image.mean() 
    image /= image.std() 
        
    image = image + abs(np.min(image))    
   
    if savetofile:
        im2save = nib.Nifti1Image(imgs_npy[0,:,:,:], affine=np.eye(4))
        nib.save(im2save, os.path.join(output_folder, patient_name + ".nii.gz"))
        # now save as npz
        np.save(os.path.join(output_folder, patient_name + ".npy"), imgs_npy)
        metadata = {
            'spacing': spacing,
            'direction': direction,
            'origin': origin,
            'original_shape': original_shape,
            'nonzero_region': nonzero
        }
    
        save_pickle(metadata, os.path.join(output_folder, patient_name + ".pkl"))
        rescaled = (255.0 / image.max() * (image - image.min())).astype(np.uint8)
        rescaled = ndimage.rotate(rescaled ,90)            
        name = os.path.join(output_folder, patient_name + ".png")
        im = Image.fromarray(rescaled)      
        im.save(name + '.png') 
    else:
        return imgs_npy, metadata


if __name__ == "__main__":

    basedir = ''
    outdir = ''
    class_names = ['ADC']
    list_of_lists = get_list_of_files(basedir, class_names)
    class_name = class_names[0].split('_')[0]
    outdir = os.path.join(outdir,class_names[0])
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
        
    patient_names = [i[0].split("/")[-2]+'_'+i[0].split("/")[-1].split('.')[0] for i in list_of_lists]

    p = Pool(processes=10)
    p.starmap(load_and_preprocess, zip(list_of_lists, patient_names, [outdir] * len(list_of_lists),
                                       [True] * len(list_of_lists), [class_name] * len(list_of_lists)))
    p.close()
    p.join()
    

    print('Done!')
    
    