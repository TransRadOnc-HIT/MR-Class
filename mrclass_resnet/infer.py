# -*- coding: utf-8 -*-

import torch 
from mrclass_resnet.utils import load_checkpoint
from mrclass_resnet.MRClassiferDataset import MRClassifierDataset
from torch.utils.data import DataLoader
from collections import defaultdict
import glob
import os
import csv

def infer(config):
    
    CP_DIR =  config['checkpoint_dir']
    save_csv =  config['save_csv']
    checkpoints = {'T1': CP_DIR + '/T1_other.pth',
                   'T2': CP_DIR +'/T2_other.pth',
                   'FLAIR': CP_DIR+'/FLAIR_other.pth',
                   'SWI': CP_DIR+'/SWI_other.pth',
                   'ADC': CP_DIR+'/ADC_other.pth',
                   }
    sub_checkpoints = { 'T1': CP_DIR  + '/T1_T1KM.pth'}
    root_dir =   config['root_dir']
    for_inference = [x for x in glob.glob(root_dir+'/*/*.nii.gz') if not os.path.isdir(x)]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # iteration through the different models
    labeled = defaultdict(list)
    for cl in checkpoints.keys():
        model,class_names,scan = load_checkpoint(checkpoints[cl])
        print('Classifying {0} MR scans'.format(class_names[0]))
        class_names[1] = '@lL'
       
        test_dataset = MRClassifierDataset(list_images = for_inference, class_names = class_names,
                                           scan = scan,infer = True,remove_corrupt= True)
        test_dataloader = DataLoader(test_dataset, batch_size = 1, shuffle=False, num_workers=8)
        for step, data in enumerate(test_dataloader):
            inputs = data['image']
            img_name = data['fn']
            inputs = inputs.to(device)
            output = model(inputs)
            prob = output.data.cpu().numpy()
            actRange = abs(prob[0][0])+abs(prob[0][1])
            index = output.data.cpu().numpy().argmax()
            if index == 1:
                labeled[img_name[0]].append([cl,actRange])

#check double classification and compare the activation value of class 0
    labeled_cleaned = defaultdict(list)
    for key in labeled.keys():
        r = 0
        j = 0
        for i in range(len(labeled[key])):
            if labeled[key][i][1] > r:
                r = labeled[key][i][1]
                j = i
        labeled_cleaned[key] = labeled[key][j]

    labeled_images = defaultdict(list)
    for key in labeled_cleaned.keys():
        labeled_images[labeled_cleaned[key][0]].append(key)
        
    # subclassification        
    labeled_sub= defaultdict(list)
    for cl in sub_checkpoints.keys():
        
        model,class_names,scan = load_checkpoint(sub_checkpoints[cl])
        test_dataset = MRClassifierDataset(list_images = labeled_images[cl],
                                           class_names = class_names, scan = scan, 
                                           subclasses = True,infer = True)
        print('Checking if contrast agent was administered for T1w scans' )
        test_dataloader = DataLoader(test_dataset, batch_size = 1, shuffle=False, num_workers=8)
        for step, data in enumerate(test_dataloader):
            inputs = data['image']
            img_name = data['fn']
            inputs = inputs.to(device)
            output = model(inputs)
            prob = output.data.cpu().numpy()
            actRange = abs(prob[0][0])+abs(prob[0][1])
            index = output.data.cpu().numpy().argmax()
            if index == 1:
                c = '-CA'
            else:
                c = ''
            labeled_sub[img_name[0]].append([cl+c,actRange])
           
    for key in labeled_sub.keys():
        labeled_cleaned[key] = labeled_sub[key][0]

    # check for the unlabeled images
    not_labeled = list(set(for_inference) - set(list(labeled_cleaned.keys())))
    for img in not_labeled:
        labeled_cleaned[img] = ['other','NA']
     
    if(save_csv):
	with open(root_dir+'/MR-Class_labels.csv', 'w') as csv_file:  
		writer = csv.writer(csv_file)
		for key, value in labeled_cleaned.items():
		    writer.writerow([key, value])   
    else:
    	return labeled_cleaned
       
