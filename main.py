import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import datasets
from torch.autograd import Variable
from tqdm import tqdm
import PIL.Image as Image
import shutil
import numpy as np
import cv2


import torchvision.transforms as transforms

'''
Example of commands :

!python3 "main.py" --batch-size 16 --epochs 15 --lr 0.01 --model "VGG"

'''

# Training settings
parser = argparse.ArgumentParser(description='RecVis Theophane Gregoir submission')

parser.add_argument('--data', type=str, default='bird_dataset', metavar='D',
                    help="folder where data is located. train_images/ and val_images/ need to be found in the folder")
parser.add_argument('--batch-size', type=int, default=16, metavar='B',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--experiment', type=str, default='experiment', metavar='E',
                    help='folder where experiment outputs are located.')
parser.add_argument('--needMerge', type=str, default='No', metavar='D',
                    help='Yes if you need to create merged dataset, No if already existing')
parser.add_argument('--crop', type=str, default='None', metavar='D',
                    help='None if your dataset is already cropped, merge if you need to crop the merged dataset, test if you need to crop the test dataset, all for both')
parser.add_argument('--model', type=str, default='Resnext', metavar='D',
                    help='Either all to train all models and observe RMV ensemble model or the name of the one you want to train (Resnet, Resnext...)')

args = parser.parse_args()
use_cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)

# Create experiment folder
if not os.path.isdir(args.experiment):
    os.makedirs(args.experiment)
    
from data import data_transforms_simple, data_transforms_augmented

from model import TG_Resnet, TG_VGG, TG_Densenet, TG_Resnext, TG_Alexnet, TG_Efficientnet


#%% MERGING
#Merging validation and train to recreate a balanced split !

val_dir = args.data + '/val_images'
train_dir = args.data + '/train_images'
test_dir = args.data + '/test_images/mistery_category'
merge_dir = args.data + '/merge_images'

if not os.path.isdir(merge_dir):
    os.mkdir(merge_dir)
                  
def create_merge(train_dir, val_dir, merge_dir):
    
    ### Copy training data in merge
    train_folders = os.listdir(train_dir)
    for train_folder in train_folders :
        if not train_folder.startswith('.'):
            merge_folder = merge_dir + '/' + train_folder
            if not os.path.isdir(merge_folder):
                os.mkdir(merge_folder)
            src_files = os.listdir(train_dir+'/'+train_folder)
            for file_name in src_files:
                full_file_name = os.path.join(train_dir + '/' + train_folder, file_name)
                if os.path.isfile(full_file_name):
                    shutil.copy(full_file_name, merge_folder)
    
    ### Copy validation data in merge
    val_folders = os.listdir(val_dir)
    for val_folder in val_folders :
        if not val_folder.startswith('.'):
            merge_folder = merge_dir + '/' + val_folder
            if not os.path.isdir(merge_folder):
                os.mkdir(merge_folder)
            src_files = os.listdir(val_dir + '/' + val_folder)
            for file_name in src_files:
                full_file_name = os.path.join(val_dir + '/' + val_folder, file_name)
                if os.path.isfile(full_file_name):
                    shutil.copy(full_file_name, merge_folder)

if args.needMerge == "Yes":
    create_merge(train_dir, val_dir, merge_dir)
    
#%% BOX DETECTION AND CROPPING
# Cropping each image in dataset to center on the bird part

if args.crop != "None":

    box_detection_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    box_detection_model.eval()
    
    COCO_INSTANCE_CATEGORY_NAMES = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
        'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    
    def get_prediction_box(img_path, threshold):
      img = Image.open(img_path) # Load the image
      transform = transforms.Compose([transforms.ToTensor()]) # Defing PyTorch Transform
      img = transform(img)
      c, h, w = img.shape # Apply the transform to the image
      pred = box_detection_model([img]) # Pass the image to the model
      pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())] # Get the Prediction Score
      pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())] # Bounding boxes
      pred_score = list(pred[0]['scores'].detach().numpy())
      predictions = [pred_score.index(x) for x in pred_score if x > threshold]
      if len(predictions) > 0:
          pred_t = predictions[-1] # Get list of index with score greater than threshold.
          pred_boxes = pred_boxes[:pred_t+1]
          pred_class = pred_class[:pred_t+1]
      else :
          pred_boxes = []
          pred_class = []
      return pred_boxes, pred_class, list(pred[0]['labels'].numpy())
    
    
    if not os.path.isdir(merge_dir):
        os.mkdir(merge_dir)
                      
    merge_dir_cropped = args.data + '/merge_images_cropped'
    
    #Creating train and valid file for cropped image
    if not os.path.isdir(merge_dir_cropped):
        os.mkdir(merge_dir_cropped)
        
    merge_dir_boxed = args.data + '/merge_images_boxed'
    #Creating train and valid file for boxed image
    if not os.path.isdir(merge_dir_boxed):
        os.mkdir(merge_dir_boxed)
    
    test_dir_cropped = args.data + '/test_images/mistery_cropped'
    
    #Creating test file for cropped image
    if not os.path.isdir(test_dir_cropped):
        os.mkdir(test_dir_cropped)
    
    def crop_test(test_dir, test_dir_cropped):
        for f in tqdm(os.listdir(test_dir)):
            pred_boxes, pred_class, _ = get_prediction_box(test_dir + '/' +f, threshold=0.60)
            if 'bird' in pred_class :
                print("Bird found")
                idx = pred_class.index('bird')
                bird_box = pred_boxes[idx]
                img = Image.open(test_dir + '/' +f)
                img_cropped = img.crop((int(bird_box[0][0]), int(bird_box[0][1]), int(bird_box[1][0]), int(bird_box[1][1])))
                img_cropped.save(test_dir_cropped+'/'+f, "JPEG", quality=95, optimize=True, progressive=True)
            else :
                print("No bird found")
                img = Image.open(test_dir + '/' +f)
                img.save(test_dir_cropped+'/'+f, "JPEG", quality=95, optimize=True, progressive=True)
                
    def crop_merge(merge_dir, merge_dir_cropped):
        merge_folders = os.listdir(merge_dir)
        for merge_folder in merge_folders:
            merge_folder_cropped = merge_dir_cropped + '/' + merge_folder
            if not os.path.isdir(merge_folder_cropped):
                os.mkdir(merge_folder_cropped)
            for f in tqdm(os.listdir(merge_dir+'/'+merge_folder)):
                pred_boxes, pred_class, _ = get_prediction_box(merge_dir + '/' + merge_folder + '/' +f, threshold=0.60)
                if 'bird' in pred_class :
                    print("Bird found")
                    idx = pred_class.index('bird')
                    bird_box = pred_boxes[idx]
                    img = Image.open(merge_dir + '/' + merge_folder + '/' +f)
                    img_cropped = img.crop((int(bird_box[0][0]), int(bird_box[0][1]), int(bird_box[1][0]), int(bird_box[1][1])))
                    img_cropped.save(merge_folder_cropped+'/'+f, "JPEG", quality=95, optimize=True, progressive=True)
                else :
                    print("No bird found")
                    img = Image.open(merge_dir + '/' + merge_folder + '/' +f)
                    img.save(merge_folder_cropped+'/'+f, "JPEG", quality=95, optimize=True, progressive=True)
    
    def draw_boxes(boxes, classes, labels, image):
        # read the image with OpenCV
        COLORS = np.random.uniform(0, 255, size=(len(COCO_INSTANCE_CATEGORY_NAMES), 3))
        image = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB)
        for i, box in enumerate(boxes):
            color = COLORS[labels[i]]
            cv2.rectangle(
                image,
                (int(box[0][0]), int(box[0][1])),
                (int(box[1][0]), int(box[1][1])),
                color, 2
            )
            cv2.putText(image, classes[i], (int(box[0][0]), int(box[0][1]-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, 
                        lineType=cv2.LINE_AA)
        return image
    
    def box_merge(merge_dir, merge_dir_boxed):
        merge_folders = os.listdir(merge_dir)
        for merge_folder in merge_folders:
            merge_folder_cropped = merge_dir_cropped + '/' + merge_folder
            if not os.path.isdir(merge_folder_cropped):
                os.mkdir(merge_folder_cropped)
            for f in tqdm(os.listdir(merge_dir+'/'+merge_folder)):
                pred_boxes, pred_class, pred_label = get_prediction_box(merge_dir + '/' + merge_folder + '/' +f, threshold=0.60)
                img = Image.open(merge_dir + '/' + merge_folder + '/' +f)
                img_boxed = draw_boxes(pred_boxes, pred_class, pred_label, img)
                cv2.imwrite(merge_dir_boxed+'/'+f,img_boxed)
    
    if args.crop == "merge" or args.crop == "all" :
        crop_merge(merge_dir, merge_dir_cropped)
    if args.crop == "test" or args.crop == "all" :
        crop_test(test_dir, test_dir_cropped)


#%% Training and validation of classification model

entire_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/merge_images_cropped',
                         transform=data_transforms_augmented),
    batch_size=args.batch_size, shuffle=True, num_workers=0)

lengths = [int(len(entire_loader.dataset)*0.8), len(entire_loader.dataset)-int(len(entire_loader.dataset)*0.8)]
train_dataset,val_dataset=torch.utils.data.random_split(entire_loader.dataset,lengths)

train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=args.batch_size, shuffle=False)


def train(epoch, classification_model, optimizer):
    classification_model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = classification_model(data)
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))

def validation(classification_model):
    classification_model.eval()
    validation_loss = 0
    correct = 0
    for data, target in val_loader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        output = classification_model(data)
        # sum up batch loss
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        validation_loss += criterion(output, target).data.item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    validation_loss /= len(val_loader.dataset)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        validation_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))
    return 100. * correct / len(val_loader.dataset)


### Training of the 5 chosen models and Reliable Voting Majority evaluated

if args.model == 'all':
    
    models_dict = {}
    use_cuda = torch.cuda.is_available()
    
    models_dict['VGG'] = TG_VGG()
    if use_cuda:
        models_dict['VGG'].cuda()

    models_dict['Resnet'] = TG_Resnet()
    if use_cuda:
        models_dict['Resnet'].cuda()
    
    models_dict['Densenet'] = TG_Densenet()
    if use_cuda:
        models_dict['Densenet'].cuda()

    models_dict['Resnext'] = TG_Resnext()
    if use_cuda:
        models_dict['Resnext'].cuda()
        
    models_dict['Efficientnet'] = TG_Efficientnet()
    if use_cuda:
        models_dict['Efficientnet'].cuda()
    
    
    optimizers = {}
    for m in models_dict: 
        optimizers[m] = optim.SGD(models_dict[m].parameters(), lr=args.lr, momentum=args.momentum)
        
    for m in models_dict:         
        exp_model = args.experiment + '/' + m
        # Create experiment folder for this model
        if not os.path.isdir(exp_model):
            os.makedirs(exp_model)
    
    weights_to_keep = {}
    
    for m in models_dict:   
        validation_accuracy = []
        max_val = 0
        epoch_to_keep = 0
        for epoch in range(1, args.epochs + 1):
            train(epoch, models_dict[m], optimizers[m])
            val_accuracy_epoch = validation(models_dict[m])
            validation_accuracy.append(val_accuracy_epoch)
            if val_accuracy_epoch > max_val :
                max_val = val_accuracy_epoch
                epoch_to_keep = epoch
            model_file = args.experiment + '/' + m + '/model_' + str(epoch) + '.pth'
            torch.save(models_dict[m].state_dict(), model_file)
            print('Saved model to ' + model_file + '. You can run `python evaluate.py --model ' + model_file + '` to generate the Kaggle formatted csv file')
            np.save(m + "_validation_accuracy.npy", np.array(validation_accuracy))
            print("History of validation accuracies saved !")
            
        weights_to_keep[m] = args.experiment + '/' + m + '/model_' + str(epoch_to_keep) + '.pth'

    models_eval = {}
    use_cuda = torch.cuda.is_available()
    
    models_eval['VGG'] = TG_VGG()
    state_dict = torch.load(weights_to_keep['VGG'])
    models_eval['VGG'].load_state_dict(state_dict)
    models_eval['VGG'].eval()
    if use_cuda:
        models_eval['VGG'].cuda()

    models_eval['Resnet'] = TG_Resnet()
    state_dict = torch.load(weights_to_keep['Resnet'])
    models_eval['Resnet'].load_state_dict(state_dict)
    models_eval['Resnet'].eval()
    if use_cuda:
        models_eval['Resnet'].cuda()
    
    models_eval['Densenet'] = TG_Densenet()
    state_dict = torch.load(weights_to_keep['Densenet'])
    models_eval['Densenet'].load_state_dict(state_dict)
    models_eval['Densenet'].eval()
    if use_cuda:
        models_eval['Densenet'].cuda()

    models_eval['Resnext'] = TG_Resnext()
    state_dict = torch.load(weights_to_keep['Resnext'])
    models_eval['Resnext'].load_state_dict(state_dict)
    models_eval['Resnext'].eval()
    if use_cuda:
        models_eval['Resnext'].cuda()
        
    models_eval['Efficientnet'] = TG_Efficientnet()
    state_dict = torch.load(weights_to_keep['Efficientnet'])
    models_eval['Efficientnet'].load_state_dict(state_dict)
    models_eval['Efficientnet'].eval()
    if use_cuda:
        models_eval['Efficientnet'].cuda()
    
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=1, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=1, shuffle=False)
    
    ### Reliable majority Voting validation accuracy
    
    def create_reliability_vector(selected_model, train_loader):
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
    
    reliability_dict = {}
    for model_type in models_eval:
        print("Creating reliability vector for " + model_type)
        reliability_dict[model_type] = create_reliability_vector(models_eval[model_type],train_loader)
        np.save("reliability_" + model_type+ ".npy", reliability_dict[model_type])
        print(reliability_dict[model_type])
    
    total = 0
    pred_True = 0
    for data, target in val_loader:
        total+=1
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        preds = [0]*20
        for model_type in models_dict:
            output = models_dict[model_type](data)
            pred = output.data.max(1, keepdim=True)[1]
            preds[pred] += reliability_dict[model_type][pred]
            
        final_pred = int(np.argmax(preds))
        correct = target.data.view_as(pred)
        if int(correct) == final_pred:
            pred_True+=1
    print("Validation accuracy RMV : ")
    print(float(pred_True)/total)
    
    total = 0
    pred_True = 0
    for data, target in val_loader:
        total+=1
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        preds = [0]*20
        for model_type in models_dict:
            output = models_dict[model_type](data)
            pred = output.data.max(1, keepdim=True)[1]
            preds[pred] += 1 
            
        final_pred = int(np.argmax(preds))
        correct = target.data.view_as(pred)
        if int(correct) == final_pred:
            pred_True+=1
    print("Validation accuracy Classic MV : ")
    print(float(pred_True)/total)


### training of a single model 
else :
    if args.model == "Resnet":
        classification_model = TG_Resnet()
    if args.model == "VGG":
        classification_model = TG_VGG()
    if args.model == "Densenet":
        classification_model = TG_Densenet()
    if args.model == "Alexnet":
        classification_model = TG_Alexnet()
    if args.model == "Resnext":
        classification_model = TG_Resnext()
    if args.model == "Efficientnet":
        classification_model = TG_Efficientnet()
    
    if use_cuda:
        print('Using GPU')
        classification_model.cuda()
    else:
        print('Using CPU')
    
    exp_model = args.experiment + '/' + args.model
        # Create experiment folder for this model
    if not os.path.isdir(exp_model):
        os.makedirs(exp_model)
    
    
    optimizer = optim.SGD(classification_model.parameters(), lr=args.lr, momentum=args.momentum)
    validation_accuracy = []
    for epoch in range(1, args.epochs + 1):
        train(epoch, classification_model,optimizer)
        validation_accuracy.append(validation(classification_model))
        model_file = args.experiment + '/' + args.model + '/model_' + str(epoch) + '.pth'
        torch.save(classification_model.state_dict(), model_file)
        print('Saved model to ' + model_file + '. You can run `python evaluate.py --model ' + model_file + '` to generate the Kaggle formatted csv file\n')
        np.save(args.model + "_validation_accuracy.npy", np.array(validation_accuracy))
        print("History of validation accuracies saved !")
    


