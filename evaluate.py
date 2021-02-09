import argparse
from tqdm import tqdm
import os
import PIL.Image as Image
import numpy as np

import torch
from torchvision import datasets

from model import TG_Resnet, TG_VGG, TG_Densenet, TG_Resnext, TG_Alexnet, TG_Efficientnet
from data import data_transforms_simple, data_transforms_augmented


'''

Example of commands :

!python3 "evaluate.py" --type_eval 'reliable_majority' --modelEfficientnet experiment/Efficientnet/model_1.pth 
--modelResnet experiment/Resnet/model_1.pth --modelResnext experiment/Resnext/model_1.pth 
--modelDensenet experiment/Densenet/model_1.pth --modelVGG experiment/VGG/model_1.pth   


!python3 "evaluate.py" --modeltype 'Efficientnet' --model experiment/Efficientnet/model_1.pth 

'''
parser = argparse.ArgumentParser(description='RecVis A3 evaluation script')
parser.add_argument('--data', type=str, default='bird_dataset', metavar='D',
                    help="folder where data is located. test_images/ need to be found in the folder")
parser.add_argument('--modeltype', type=str, default='bird_dataset', metavar='D',
                    help="folder where data is located. test_images/ need to be found in the folder")
parser.add_argument('--model', type=str, metavar='M',
                    help="the model file to be evaluated. Usually it is of the form model_X.pth")
parser.add_argument('--outfile', type=str, default='experiment/kaggle.csv', metavar='D',
                    help="name of the output csv file")

### For majority voting
parser.add_argument('--type_eval', type=str, default='classic', metavar='D',
                    help="reliable_majority, majority or classic")
parser.add_argument('--modelVGG', type=str, default='', metavar='D',
                    help="path to weights")
parser.add_argument('--modelResnet', type=str, default='', metavar='D',
                    help="path to weights")
parser.add_argument('--modelDensenet', type=str, default='', metavar='D',
                    help="path to weights")
parser.add_argument('--modelResnext', type=str, default='', metavar='D',
                    help="path to weights")
parser.add_argument('--modelAlexnet', type=str, default='', metavar='D',
                    help="path to weights")
parser.add_argument('--modelEfficientnet', type=str, default='', metavar='D',
                    help="path to weights")
parser.add_argument('--preferredModel', type=str, default='', metavar='D',
                    help="For normal majority voting, preferred model in case of equality.")

args = parser.parse_args()

test_dir_cropped = args.data + '/test_images/mistery_cropped'

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')
        

### Reliable Majority Voting case
if args.type_eval == "reliable_majority":
    
    entire_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.data + '/merge_images_cropped',
                             transform=data_transforms_augmented),
        batch_size=1, shuffle=True, num_workers=0)

    lengths = [int(len(entire_loader.dataset)*0.8), len(entire_loader.dataset)-int(len(entire_loader.dataset)*0.8)]
    train_dataset,val_dataset=torch.utils.data.random_split(entire_loader.dataset,lengths)
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=1, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=1, shuffle=False)
    
    def create_reliability_vector(selected_model):
        predictions_vector = [0.]*20
        correct_predictions_vector = [0.]*20
        reliability_vector = [0.]*20    
        selected_model.eval()
        for data, target in train_loader:
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            output = selected_model(data)
            # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]
            predictions_vector[pred] += 1.
            correct = pred.eq(target.data.view_as(pred)).cpu().sum()
            if correct > 0:
                correct_predictions_vector[pred] += 1
        
        for k in range(20):
            if predictions_vector[k] < 5 :
                reliability_vector[k] = 0.5
            else :
                reliability_vector[k] = correct_predictions_vector[k] / predictions_vector[k]
        
        return reliability_vector
    
    models_dict = {}
    reliability_dict = {}
    use_cuda = torch.cuda.is_available()
    
    if args.modelVGG != '':
        models_dict['VGG'] = TG_VGG()
        state_dict = torch.load(args.modelVGG)
        models_dict['VGG'].load_state_dict(state_dict)
        models_dict['VGG'].eval()
        if use_cuda:
            models_dict['VGG'].cuda()

    if args.modelResnet != '':
        models_dict['Resnet'] = TG_Resnet()
        state_dict = torch.load(args.modelResnet)
        models_dict['Resnet'].load_state_dict(state_dict)
        models_dict['Resnet'].eval()
        if use_cuda:
            models_dict['Resnet'].cuda()
        
    if args.modelDensenet != '':
        models_dict['Densenet'] = TG_Densenet()
        state_dict = torch.load(args.modelDensenet)
        models_dict['Densenet'].load_state_dict(state_dict)
        models_dict['Densenet'].eval()
        if use_cuda:
            models_dict['Densenet'].cuda()
    
    if args.modelResnext != '':
        models_dict['Resnext'] = TG_Resnext()
        state_dict = torch.load(args.modelResnext)
        models_dict['Resnext'].load_state_dict(state_dict)
        models_dict['Resnext'].eval()
        if use_cuda:
            models_dict['Resnext'].cuda()
            
    if args.modelAlexnet != '':
        models_dict['Alexnet'] = TG_Alexnet()
        state_dict = torch.load(args.modelAlexnet)
        models_dict['Alexnet'].load_state_dict(state_dict)
        models_dict['Alexnet'].eval()
        if use_cuda:
            models_dict['Alexnet'].cuda()
    
    if args.modelEfficientnet != '':
        models_dict['Efficientnet'] = TG_Efficientnet()
        state_dict = torch.load(args.modelEfficientnet)
        models_dict['Efficientnet'].load_state_dict(state_dict)
        models_dict['Efficientnet'].eval()
        if use_cuda:
            models_dict['Efficientnet'].cuda()
    
    for model_type in models_dict:
        print("Creating reliability vector for " + model_type)
        reliability_dict[model_type] = create_reliability_vector(models_dict[model_type])
        np.save("reliability_pow4_" + model_type+ ".npy", reliability_dict[model_type])
        print(reliability_dict[model_type])
    
    output_file = open("ReliableMajo_Kaggle.csv", "w")
    output_file.write("Id,Category\n")
    
    for f in tqdm(os.listdir(test_dir_cropped)):
        if 'jpg' in f:
            data = data_transforms_simple(pil_loader(test_dir_cropped + '/' + f))
            data = data.view(1, data.size(0), data.size(1), data.size(2))
            if use_cuda:
                data = data.cuda()       
            preds = [0]*20
            for model_type in models_dict:
                output = models_dict[model_type](data)
                pred = output.data.max(1, keepdim=True)[1]
                if model_type == args.preferredModel :
                    preferred_res = int(pred)
                preds[pred] += reliability_dict[model_type][pred]
                
            final_pred = int(np.argmax(preds))
            output_file.write("%s,%d\n" % (f[:-4], final_pred))
        
    output_file.close()
    
    print("Reliable Majority voting succesfully wrote " + "ReliableMajo_Kaggle.csv" + ', you can upload this file to the kaggle competition website')


### Classic Majority Voting case
elif args.type_eval == "majority":
    
    models_dict = {}
    use_cuda = torch.cuda.is_available()
    
    if args.modelVGG != '':
        models_dict['VGG'] = TG_VGG()
        state_dict = torch.load(args.modelVGG)
        models_dict['VGG'].load_state_dict(state_dict)
        models_dict['VGG'].eval()
        if use_cuda:
            models_dict['VGG'].cuda()
        
    if args.modelResnet != '':
        models_dict['Resnet'] = TG_Resnet()
        state_dict = torch.load(args.modelResnet)
        models_dict['Resnet'].load_state_dict(state_dict)
        models_dict['Resnet'].eval()
        if use_cuda:
            models_dict['Resnet'].cuda()
        
    if args.modelDensenet != '':
        models_dict['Densenet'] = TG_Densenet()
        state_dict = torch.load(args.modelDensenet)
        models_dict['Densenet'].load_state_dict(state_dict)
        models_dict['Densenet'].eval()
        if use_cuda:
            models_dict['Densenet'].cuda()
    
    if args.modelResnext != '':
        models_dict['Resnext'] = TG_Resnext()
        state_dict = torch.load(args.modelResnext)
        models_dict['Resnext'].load_state_dict(state_dict)
        models_dict['Resnext'].eval()
        if use_cuda:
            models_dict['Resnext'].cuda()
            
    if args.modelAlexnet != '':
        models_dict['Alexnet'] = TG_Alexnet()
        state_dict = torch.load(args.modelAlexnet)
        models_dict['Alexnet'].load_state_dict(state_dict)
        models_dict['Alexnet'].eval()
        if use_cuda:
            models_dict['Alexnet'].cuda()
    
    if args.modelEfficientnet != '':
        models_dict['Efficientnet'] = TG_Efficientnet()
        state_dict = torch.load(args.modelEfficientnet)
        models_dict['Efficientnet'].load_state_dict(state_dict)
        models_dict['Efficientnet'].eval()
        if use_cuda:
            models_dict['Efficientnet'].cuda()
    
    
    output_file = open(args.outfile, "w")
    output_file.write("Id,Category\n")
    
    for f in tqdm(os.listdir(test_dir_cropped)):
        if 'jpg' in f:
            data = data_transforms_simple(pil_loader(test_dir_cropped + '/' + f))
            data = data.view(1, data.size(0), data.size(1), data.size(2))
            if use_cuda:
                data = data.cuda()       
            preds = [0]*20
            for model_type in models_dict:
                output = models_dict[model_type](data)
                pred = output.data.max(1, keepdim=True)[1]
                if model_type == args.preferredModel :
                    preferred_res = int(pred)
                preds[pred] += 1
                
            if max(preds) < 2:
                final_pred = preferred_res
                print("By default, we choose : " + args.preferredModel)
            else :
                final_pred = int(np.argmax(preds))
            output_file.write("%s,%d\n" % (f[:-4], final_pred))
    
    output_file.close()
    
    print("Majority voting succesfully wrote " + args.outfile + ', you can upload this file to the kaggle competition website')


### Single model evaluation case
else :

    use_cuda = torch.cuda.is_available()
    
    state_dict = torch.load(args.model)
    
    if args.modeltype == "VGG":
        model = TG_VGG()
    elif args.modeltype == "Resnet":
        model = TG_Resnet()
    elif args.modeltype == "Densenet":
        model = TG_Densenet()
    elif args.modeltype == "Resnext":
        model = TG_Resnext()
    elif args.modeltype == "Alexnet":
        model = TG_Alexnet() 
    elif args.modeltype == "Efficientnet":
        model = TG_Efficientnet() 
        
    model.load_state_dict(state_dict)
    model.eval()
    
    if use_cuda:
        print('Using GPU')
        model.cuda()
    else:
        print('Using CPU')
    
    output_file = open("Kaggle_"+args.modeltype+".csv", "w")
    output_file.write("Id,Category\n")
    for f in tqdm(os.listdir(test_dir_cropped)):
        if 'jpg' in f:
            data = data_transforms_simple(pil_loader(test_dir_cropped + '/' + f))
            data = data.view(1, data.size(0), data.size(1), data.size(2))
            if use_cuda:
                data = data.cuda()
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]
            output_file.write("%s,%d\n" % (f[:-4], pred))
    
    output_file.close()
    
    print("Succesfully wrote " + "Kaggle_"+args.modeltype+".csv" + ', you can upload this file to the kaggle competition website')
        


