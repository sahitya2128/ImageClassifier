# Imports here
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from PIL import Image
import json
from collections import OrderedDict


# Set up model
def setup_model(structure = 'vgg16', dropout = 0.5, lr=0.001, power = 'GPU', hidden_layer = 512):
    
    if structure == 'vgg16':
        model = models.vgg16(pretrained=True)
    if structure == 'alexnet':
        model = models.alexnet(pretrained=True)
    if structure == 'densenet121':
        model = models.densenet121(pretrained=True)
        
    # Freeze
    for param in model.parameters():
        param.requires_grad = False

    num_features = model.classifier[0].in_features
    
    ##Build Classifier
    classifier = nn.Sequential(OrderedDict([
                              ('dropout',nn.Dropout(p=0.5)),
                              ('fc1', nn.Linear(num_features, 500)),
                              ('relu', nn.ReLU()),
                              ('fc2', nn.Linear(500, 100)),
                              ('relu2', nn.ReLU()),
                              ('fc3', nn.Linear(100,102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    model.classifier = classifier
    
    # Select Criterion and Optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    device = torch.device("cuda" if power and torch.cuda.is_available() else "cpu")
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    return model, criterion, optimizer

# Train model using your setting
def train_model(model, criterion, optimizer, dataloaders, validloaders, power = 'GPU', epochs=1):
    # Training
    for e in range(epochs):
        running_loss = 0
        model.train()
        device = torch.device("cuda" if power and torch.cuda.is_available() else "cpu")
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for images, labels in dataloaders:
            images,labels = images.to(device), labels.to(device)
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        else:
            model.eval()
            accuracy = 0
            vaild_loss = 0
            with torch.no_grad():
                for images, labels in validloaders:
                    images,labels = images.to(device), labels.to(device)
                    log_ps = model(images)
                    vaild_loss += criterion(log_ps, labels)
                    ps = torch.exp(log_ps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor))

            running_loss = running_loss / len(dataloaders)
            valid_loss = vaild_loss / len(validloaders)
            accuracy = accuracy / len(validloaders)
            print("epoch {0}/10 Training loss: {1} ".format(e+1,running_loss),
                 "Vaildation loss: {}".format(valid_loss),
                 "Accurancy:{}".format(accuracy))
        
    print('Finished!')
        
# Test your model
def test_model(model, testloaders, criterion, power = 'GPU'):
    # TODO: Do validation on the test set
    size = 0
    correct = 0
    model.eval()
    with torch.no_grad():
            for images, labels in testloaders:
                device = torch.device("cuda" if power and torch.cuda.is_available() else "cpu")
                #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                images,labels = images.to(device), labels.to(device)
                log_ps = model(images)
                vaild_loss += criterion(log_ps, labels)
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                size += labels.size(0)
                correct += equals.sum().item()
    test_accuracy = correct / size
    print("Accurancy:{:.4f}".format(test_accuracy))
    
# Save model
def save_model(class_to_idx, path, model, structure, optimizer):
    model.class_to_idx = class_to_idx
    checkpoint = {'classifier': model.classifier,
        'optimizer': optimizer,
        'arch': structure,
        'state_dict': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'class_to_idx': model.class_to_idx
    }
    torch.save(checkpoint,path)
    print('Trained model has been saved!')
    
# Loading the checkpoint
def load_checkpoin(path = 'checkpoint.pth'):
    checkpoint = torch.load(path)
    classifier = checkpoint['classifier']
    structure = checkpoint['arch']
    if structure == 'vgg16':
        model = models.vgg16(pretrained=True)
    if structure == 'alexnet':
        model = models.alexnet(pretrained=True)
    if structure == 'densenet121':
        model = models.densenet121(pretrained=True)
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])
    return model    
  
def predict(image_path, model, topk=5, power = 'GPU', category_names = 'cat_to_name.json'):
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
    # TODO: Implement the code to predict the class from an image file
    image = process_image(image_path)
    device = torch.device("cuda" if power and torch.cuda.is_available() else "cpu")
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    image = image.to(device)
    image = image.unsqueeze_(0)
    image = image.float()
    with torch.no_grad():
        output = model.forward(image)
    probability = torch.exp(output)
    prob, label = probability.topk(topk)
    folder_index = []
    for i in np.array(label[0]):
        for folder, num in model.class_to_idx.items():
            if num == i:
                folder_index += [folder]
            
    
    return prob, folder_index
    
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    transform = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
    img_tensor = transform(Image.open(image))
    return img_tensor


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax
   
def check_sanity(path):

    img = process_image(path)
    fig, ax1 = plt.subplots(figsize=(3.5,3.5), ncols=1)
    image = imshow(img)
    prob, label = predict(path,model)
    flower_name = [cat_to_name[i] for i in label]
    prob_1 = prob.data.cpu().numpy().squeeze()
    ax1.barh(flower_name, prob_1)
    print('The flower is', cat_to_name['1'])
    
