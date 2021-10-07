# -*- coding: utf-8 -*-


import torch 
from mrclass_resnet.utils import load_checkpoint
from mrclass_resnet.MRClassiferDataset import MRClassifierDataset
from torch.utils.data import DataLoader
from collections import defaultdict
import glob
import os
import csv

def test(config):
    
    CP_DIR =  config['checkpoint_dir']
    
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
        print(class_names)
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
                labeled[img_name[0]].append([cl,actRange,img_name[-1].split('/')[-2].split('_')[0]])

#check double classification and compare the activation value of class 0
    labeled_correct = defaultdict(list)
    labeled_wrong = defaultdict(list)
    labeled_cleaned = defaultdict(list)
    for key in labeled.keys():
        r = 0
        j = 0
        for i in range(len(labeled[key])):
            if labeled[key][i][1] > r:
                r = labeled[key][i][1]
                j = i
                
        labeled_cleaned[key] = labeled[key][j]
    
    for key in labeled_cleaned.keys():            
        predicted_label = labeled_cleaned[key][0]
        true_label = labeled_cleaned[key][2]
        if predicted_label in true_label:
            labeled_correct[key] = labeled_cleaned[key]
        else:
            labeled_wrong[key]= labeled_cleaned[key]
    

    labeled_images = defaultdict(list)
    for key in labeled_correct.keys():
        labeled_images[labeled_correct[key][0]].append(key)
        
    for key in labeled_wrong.keys():
        labeled_images[labeled_wrong[key][0]].append(key)
        
        
    # subclassification        
    labeled_subcorrect = defaultdict(list)
    labeled_subwrong = defaultdict(list)
    for cl in sub_checkpoints.keys():
        model,class_names,scan = load_checkpoint(sub_checkpoints[cl])
        test_dataset = MRClassifierDataset(list_images = labeled_images[cl],
                                           class_names = class_names, scan = scan, 
                                           subclasses = True,infer = True)
        test_dataloader = DataLoader(test_dataset, batch_size = 1, shuffle=False, num_workers=8)
        for step, data in enumerate(test_dataloader):
            inputs = data['image']
            img_name = data['fn']
            actual_label = data['label']
            inputs = inputs.to(device)
            output = model(inputs)
            prob = output.data.cpu().numpy()
            actRange = abs(prob[0][0])+abs(prob[0][1])
            index = output.data.cpu().numpy().argmax()
            actual_label = img_name[0].split('/')[-2].split('_')[-1]
            if index == 1:
                c = 'KM'
            else:
                c = ''
            if class_names[index] == actual_label:

                labeled_subcorrect[img_name[0]].append([cl+c,actRange,actual_label])
            else:
                labeled_subwrong[img_name[0]].append([cl+c,actRange,actual_label])
    
    for key in labeled_subcorrect.keys():
        labeled_correct[key] = labeled_subcorrect[key][0]
    for key in labeled_subwrong.keys():
        labeled_wrong[key] = labeled_subwrong[key][0]
    

    # check for the unlabeled images
    not_labeled = list(set(for_inference) - set(list(labeled_cleaned.keys())))
    for img in not_labeled:
        cl = img.split('/')[-2].split('_')[0]
        if 'other' in cl:
            labeled_correct[img] = [cl,'NA',cl]
        else:
            labeled_wrong[img] = ['other','NA',cl]
        

    T1_acc =  sum(value[2] == 'T1' for value in labeled_correct.values()) / \
                (sum(value[2] == 'T1' for value in labeled_correct.values()) + \
                sum(value[2] == 'T1' for value in labeled_wrong.values()))
                
    T1KM_acc =  sum(value[2] == 'T1KM' for value in labeled_correct.values()) / \
                (sum(value[2] == 'T1KM' for value in labeled_correct.values()) + \
                sum(value[2] == 'T1KM' for value in labeled_wrong.values()))
                
    T2_acc =  sum(value[2] == 'T2' for value in labeled_correct.values()) / \
                (sum(value[2] == 'T2' for value in labeled_correct.values()) + \
                sum(value[2] == 'T2' for value in labeled_wrong.values()))
                
                
    FL_acc =  sum(value[2] == 'FLAIR' for value in labeled_correct.values()) / \
                (sum(value[2] == 'FLAIR' for value in labeled_correct.values()) + \
                sum(value[2] == 'FLAIR' for value in labeled_wrong.values()))   
                
    ADC_acc =  sum(value[2] == 'ADC' for value in labeled_correct.values()) / \
                (sum(value[2] == 'ADC' for value in labeled_correct.values()) + \
                sum(value[2] == 'ADC' for value in labeled_wrong.values()))  
                
#    SWI_acc =  sum(value[2] == 'SWI' for value in labeled_correct.values()) / \
#                (sum(value[2] == 'SWI' for value in labeled_correct.values()) + \
#                sum(value[2] == 'SWI' for value in labeled_wrong.values()))   
    other_acc =  sum(value[2] == 'other' for value in labeled_correct.values()) / \
                (sum(value[2] == 'other' for value in labeled_correct.values()) + \
                sum(value[2] == 'other' for value in labeled_wrong.values()))                   
    total_acc = len(labeled_correct)/(len(labeled_correct)+len(labeled_wrong))
    

    info = """\
    {'-'*40}
    # Accuracies of the different MR sequence models
    # T1: {T1_acc}
    # T1-Contrast agent: {T1KM_acc}
    # T2: {T2_acc}
    # FLAIR: {FL_acc}
    # ADC: {ADC}
    # SWI: {SWI}
    # '-'*40
    # MR-Class accuracy: {total_acc}
    {'-'*40}
    """
    print(info)

    
    with open(root_dir+'/label_wrong.csv', 'w') as csv_file:  
        writer = csv.writer(csv_file)
        for key, value in labeled_wrong.items():
            writer.writerow([key, value])   
            
    with open(root_dir+'/label_correct.csv', 'w') as csv_file:  
        writer = csv.writer(csv_file)
        for key, value in labeled_correct.items():
            writer.writerow([key, value])              