import os
import torch
import torchvision
import numpy as np
import torch.nn.functional as F

from torch.autograd import Variable
from torchvision import models
from PIL import Image


class Functions:
    
    # Number of categories
    def number_of_categories(self, path_to_data):
        number_of_folders = 0
        for _, dirnames, filenames in os.walk(path_to_data):
            number_of_folders += len(dirnames)
        return number_of_folders
    
    # Save model function
    def save_model(self, checkpoint, save_dir):
        torch.save(checkpoint, save_dir + "/checkpoint.pth")
        
    # Loads a checkpoint and rebuilds the model
    def load_checkpoint(self, filepath):
        checkpoint = torch.load(filepath + "/checkpoint.pth")
        model = getattr(torchvision.models, checkpoint["arch"])(pretrained=True)
        model.classifier = checkpoint['classifier']
        model.load_state_dict(checkpoint['state_dict'])
        return model

    # Training function
    def do_deep_learning(self, model, trainloader, epochs, print_every, criterion, optimizer, device):
        epochs = epochs
        print_every = print_every
        steps = 0

        # change to device
        model.to(device)

        for e in range(epochs):
            running_loss = 0
            for ii, (inputs, labels) in enumerate(trainloader):
                steps += 1

                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                # Forward and backward passes
                outputs = model.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if steps % print_every == 0:
                    print("Epoch: {}/{}... ".format(e+1, epochs), running_loss,
                          "Loss: {:.4f}".format(running_loss/print_every))

                    running_loss = 0

    # Check accuracy function
    def check_accuracy_on_test(self, testloader, model, device):    
        correct = 0
        total = 0
        model.to(device)
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

    # Select architecture function
    def select_architecture(self, architecture):
        model = models.vgg16(pretrained=True)
        if architecture == "alexnet":
            model = models.alexnet(pretrained=True)
        elif architecture == "vgg11":
            model = models.vgg11(pretrained=True)
        elif architecture == "vgg11":
            model = models.vgg11(pretrained=True)
        elif architecture == "vgg11_bn":
            model = models.vgg11_bn(pretrained=True)
        elif architecture == "vgg13":
            model = models.vgg13(pretrained=True)
        elif architecture == "vgg13_bn":
            model = models.vgg13_bn(pretrained=True)
        elif architecture == "vgg16":
            model = models.vgg16(pretrained=True)
        elif architecture == "vgg16_bn":
            model = models.vgg16_bn(pretrained=True)
        elif architecture == "vgg19":
            model = models.vgg19(pretrained=True)
        elif architecture == "vgg19_bn":
            model = models.vgg19_bn(pretrained=True)
        elif architecture == "resnet18":
            model = models.resnet18(pretrained=True)
        elif architecture == "resnet34":
            model = models.resnet34(pretrained=True)
        elif architecture == "resnet50":
            model = models.resnet50(pretrained=True)
        elif architecture == "resnet101":
            model = models.resnet101(pretrained=True)
        elif architecture == "resnet152":
            model = models.resnet152(pretrained=True)
        elif architecture == "squeezenet1_0":
            model = models.squeezenet1_0(pretrained=True)
        elif architecture == "squeezenet1_1":
            model = models.squeezenet1_1(pretrained=True)
        elif architecture == "densenet121":
            model = models.densenet121(pretrained=True)
        elif architecture == "densenet161":
            model = models.densenet161(pretrained=True)
        elif architecture == "densenet169":
            model = models.densenet169(pretrained=True)
        elif architecture == "densenet201":
            model = models.densenet201(pretrained=True)
        elif architecture == "inception_v3":
            model = models.inception_v3(pretrained=True)
        return model
    
    # Process a PIL image for use in a PyTorch model
    def process_image(self, image, transforms):
        ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
            returns an Numpy array
        '''
        img = Image.open(image)
        image = transforms(img)
        return image.numpy()
    
    # Prediction function
    def predict(self, image_path, model, transforms, device, topk=5):
        ''' Predict the class (or classes) of an image using a trained deep learning model.
        '''
        image = self.process_image(image_path, transforms)
        # Convert 2D image to 1D vector
        image = np.expand_dims(image, 0)
        # Convert to a tensor
        image = torch.from_numpy(image)
        # Set to eval
        model.eval()
        #Set to GPU
        inputs = Variable(image).to(device)
        # prediction
        logits = model.forward(inputs)
        ps = F.softmax(logits,dim=1)
        topk = ps.cpu().topk(topk)
        return (e.data.numpy().squeeze().tolist() for e in topk)