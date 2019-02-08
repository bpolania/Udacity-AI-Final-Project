import argparse
import sys
import torch
import functions

from torch import nn
from torch import optim
from torchvision import datasets, transforms
from collections import OrderedDict

parser = argparse.ArgumentParser(
    description='Parser for Image Classifier Udacity Project',
)

parser.add_argument('data_dir', action='store',
                    help='Data Directory')
parser.add_argument('--arch', action='store',
                    dest='architecture',
                    default="vgg16",
                    help='Model Architecture with options: alexnet, vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn, resnet18, resnet34, resnet50, resnet101, resnet152, squeezenet1_0, squeezenet1_1, densenet121, densenet161, densenet169, densenet201, inception_v3')
parser.add_argument('--hidden_units', action='store',
                    dest='hidden_units',
                    type=int,
                    default=25088,
                    help='Number of inputs for classifier input layer')
parser.add_argument('--learning_rate', action='store',
                    dest='learning_rate',
                    default=0.001,
                    help='Learning rate')
parser.add_argument('--epochs', action='store',
                    dest='epochs',
                    type=int,
                    default=5,
                    help='Number of epochs for training the set')
parser.add_argument('--gpu', action='store_true',
                    dest='gpu',
                    help='Use gpu for processing')
parser.add_argument('--save_dir', action='store',
                    dest='save_dir',
                    help='the directory to save the checkpoint')

# Get arguments
args = parser.parse_args()

# Select model architecture
f = functions.Functions()
model = f.select_architecture(args.architecture)

# Select Device
device = "cuda"
if  not args.gpu:
    device = "cpu"
print("Device set to " + device)

## Load the data
# Set directories path
if len(sys.argv) > 0:
    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    # Define transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])


    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)

    ## Building and training the classifier
    # Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
    for param in model.parameters():
        param.requires_grad = False
    
    f = functions.Functions()
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(args.hidden_units, 500)),
                              ('relu', nn.ReLU()),
                              ('fc2', nn.Linear(500, f.number_of_categories(train_dir))),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    model.classifier = classifier

    # Build and train the network
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    
    
    f.do_deep_learning(model, trainloader, args.epochs, 40, criterion, optimizer, device)

    ## Testing the Network
    # Do validation on the test set
    f.check_accuracy_on_test(testloader, model, device)

    ## Save the checkpoint
    if args.save_dir:
        # Set Checkpoint
        checkpoint = {'input_size': args.hidden_units,
                      'model': model}
        # Save model
        f.save_model(checkpoint, args.save_dir)
else:
    print("You must set a data directory: train.py 'path/to/your/dataset'")
    

    