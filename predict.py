import argparse
import functions
import json

import numpy as np

from torchvision import datasets, transforms

parser = argparse.ArgumentParser(
    description='Parser for Image Classifier Udacity Project',
)

parser.add_argument('path_to_image', action='store',
                    help='Path to image')
parser.add_argument('checkpoint', action='store',
                    help='Path to checkpoint')
parser.add_argument('--top_k', action='store',
                    dest='top_k',
                    type=int,
                    default=3,
                    help='Top K most likely classes')
parser.add_argument('--category_names', action='store',
                    dest='category_names',
                    help='Mapping of categories to real names')
parser.add_argument('--gpu', action='store_true',
                    dest='gpu',
                    help='Use gpu for processing')

# Get arguments
args = parser.parse_args()

# Select Device
device = "cuda"
if  not args.gpu:
    device = "cpu"
print("Device set to " + device)

## Loading the checkpoint
# Load model
f = functions.Functions()
model = f.load_checkpoint(args.checkpoint)

## Class Prediction function
valid_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                           std=[0.229, 0.224, 0.225])])
valid_data = datasets.ImageFolder(args.path_to_image.split('/')[0] + "/" + args.path_to_image.split('/')[1], transform=valid_transforms)

# Run the prediction for all the images in a random folder
class_names = valid_data.classes
probs, classes = f.predict(args.path_to_image, model.to(device), valid_transforms, device, args.top_k)

if args.category_names:
    # Label Mapping
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    flower_names = [cat_to_name[class_names[e]] for e in classes]
    print(flower_names)
print(probs)

    