import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim 
from torch.utils import data as pytorch_data
from torchvision import datasets, transforms

from tqdm import tqdm

import numpy as np

from tensorboardX import SummaryWriter

from data_generator import *

import argparse

from models import MNIST_Net, Cifar_Net, ResNet18

import os 

from attacks import *
from random_mnist import MNISTRandomLabels

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


criterion = nn.KLDivLoss()

# ------------------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='flags to train the model.')
parser.add_argument('--mnist_batch_size', type=int, default=0, help="clean batch size")
parser.add_argument('--convex_batch_size', type=int, default=0, help="convex batch size")
parser.add_argument('--negative_batch_size', type=int, default=0, help='negative batch size')
parser.add_argument('--gaussian_batch_size', type=int, default=0, help='gaussian batch size')
parser.add_argument('--cifar_batch_size', type=int, default=0, help='size of CIFAR data')
parser.add_argument('--fashion_batch_size', type=int, default=0, help='size of fashion MNIST data')
parser.add_argument('--background_batch_size', type=int, default=0, help='size of background data')
parser.add_argument('--restart', type=bool, default=False, help='whether we are loading in a pretrained model')
parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--optimizer', type=str, default='SGD', help='which optimizer to use')
parser.add_argument('--momentum', type=float, default=0.5, help='momentum')
parser.add_argument('--test_batch_size', type=int, default=1000, help='test batch size')
parser.add_argument('--adversarial_train', type=bool, default=False)
parser.add_argument('--num_epochs_to_wait', type=int, default=5)
parser.add_argument('--adversarial_batch_size', type=int, default=0)
parser.add_argument('--letter_batch_size', type=int, default=0)
parser.add_argument('--output_directory',type=str,default=os.getcwd())
parser.add_argument('--closed_loop', type=bool, default=False)
parser.add_argument('--attack', type=str, default='pgd_attack', help='which attack to run')
parser.add_argument('--fashion_directory', type=str, default='fashion_data', help='which directory the fashion data is in')
parser.add_argument('--cifar_directory', type=str, default='cifar_data', help='which directory the CIFAR data is in')
parser.add_argument('--mnist_directory', type=str, default ='mnist_data', help='which directory the MNIST data is in')
parser.add_argument('--use_cifar',type=bool,default=False,help='whether to train on MNIST or CIFAR')
parser.add_argument('--model_name', type=str, default = None, help = 'name to save or load model, if none is defined it will be made based on the component batch sizes')
parser.add_argument('--random_labels', type=bool, default=False, help='whether to use randomize labels on ground truth data, useful for testing rademacher complexity of model + data')
parser.add_argument('--num_rand_points', type=int, default=500, help='how many datapoints to train on when we use random labels')

args = parser.parse_args()
# ------------------------------------------------------------------------------------------------------------------
mnist_batch_size = args.mnist_batch_size
convex_batch_size = args.convex_batch_size
fashion_batch_size = args.fashion_batch_size
gaussian_batch_size = args.gaussian_batch_size
cifar_batch_size = args.cifar_batch_size
test_batch_size = args.test_batch_size
letter_batch_size = args.letter_batch_size
background_batch_size = args.background_batch_size
# ------------------------------------------------------------------------------------------------------------------
num_epochs = args.num_epochs
lr = args.lr
momentum = args.momentum
# ------------------------------------------------------------------------------------------------------------------
restart = args.restart 

adversarial_train = args.adversarial_train
adversarial_batch_size = args.adversarial_batch_size
num_epochs_to_wait = args.num_epochs_to_wait

use_cifar = args.use_cifar
output_directory = args.output_directory
fashion_root = args.fashion_directory
cifar_root = args.cifar_directory
mnist_root = args.mnist_directory
# ------------------------------------------------------------------------------------------------------------------
model_name = args.model_name
# ------------------------------------------------------------------------------------------------------------------
random_labels = args.random_labels
num_rand_points = args.num_rand_points
# ------------------------------------------------------------------------------------------------------------------
if letter_batch_size > 0:
    letter_dataset_x = np.load('/alphabet_data/letter_dataset_x.npy')
    letter_dataset_y = np.load('/alphabet_data/letter_dataset_y.npy')
    letter_dataset_x = torch.from_numpy(letter_dataset_x)
    letter_dataset_y = torch.from_numpy(letter_dataset_y)
# ------------------------------------------------------------------------------------------------------------------
if args.attack not in ['fgsm_targeted', 'pgd_attack']:
    raise RuntimeError('Attack method is not implemented')

attack = eval(args.attack)
# ------------------------------------------------------------------------------------------------------------------
closed_loop = args.closed_loop

if not adversarial_train:
    adversarial_batch_size = 0
# ------------------------------------------------------------------------------------------------------------------
if use_cifar:
    model_to_train = ResNet18().to(device)
    if device == 'cuda':
        model_to_train = torch.nn.DataParallel(model_to_train)

else:
    model_to_train = MNIST_Net().to(device)
# ------------------------------------------------------------------------------------------------------------------
cifar_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
cifar_train_set = datasets.CIFAR10(root=cifar_root, train=True, download=True)
mnist_train_set = datasets.MNIST(root=mnist_root, train=True, download=True)
if random_labels:
    mnist_train_set = MNISTRandomLabels(root=mnist_root, train=True, download=True)



fashion_train_set = datasets.FashionMNIST(root=fashion_root, train=True, download=True)
# ------------------------------------------------------------------------------------------------------------------
positive_set = mnist_train_set
if args.use_cifar:
    positive_set = cifar_train_set
    mnist = False
    train_set = cifar_train_set
    clean_batch_size = cifar_batch_size
    clean_batch_size = 128
    test_set = datasets.CIFAR10(root=cifar_root, train=False,
                                       download=True, transform=cifar_transform)
    negative_dataset = mnist_train_set
else:
    mnist = True
    train_set = mnist_train_set
    clean_batch_size = mnist_batch_size
    test_set = datasets.MNIST(root = mnist_root, train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
    if random_labels:
        test_set = datasets.MNIST(root = mnist_root, train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
    negative_dataset = cifar_train_set
# ------------------------------------------------------------------------------------------------------------------

# This function gets us a map from a corresponding class to all indices of said class. 
# It is used when generating convex combinations to make sure we dont draw two images from the same class

if not random_labels:
    mapping = get_mapping(positive_set, mnist = (not use_cifar))


fashion_dataset = generate_negative_dataset(fashion_train_set, len(fashion_train_set), mnist = (not use_cifar))


# ------------------------------------------------------------------------------------------------------------------
def train_model(new_model, optimizer, criterion, train_data_loader, test_data_loader, num_epochs, Writer, mnist_batch_size, convex_batch_size, 
                fashion_batch_size, gaussian_batch_size, cifar_batch_size, letter_batch_size, background_batch_size, adversarial_train = False,
                adversarial_batch_size = None, num_epochs_to_wait = 5, log_interval=10, name='model_robust_convex_positive_gaussian'):
    """
    Params:
    :param new_model: The (possibly pretrained) model we are training
    :param optimizer: The optimizer we are using
    :param criterion: Which loss function we are using (we set this to KL loss by default)
    :param train_data_loader: A dataloader consisting of the training data 
    :param test_data_loader: A dataloader consisting of the test data 
    :param num_epochs: How many epochs we are training for 
    :param Writer: A utility which writes data so it can be displayed with Tensorboard 
    :param mnist_batch_size: How many MNIST samples we should sample (per batch)
    :param convex_batch_size: How many convex combinations we should make (per batch)
    :param fashion_batch_size: How many Fashion MNIST samples we should sample (per batch)
    :param gaussian_batch_size: How many gaussian noise samples we should generate (per batch)
    :param cifar_batch_size: How many cifar samples we should sample (per batch)
    :param letter_batch_size: How many EMNIST samples we should sample (per batch)
    :param background_batch_size: How many background samples we should sample (per batch)
    :param adversarial_train: Whether we are using adversarial training 
    :param adversarial_batch_size: How many adversarial samples we should generate (per batch)
    :param num_epochs_to_wait: How many epochs to wait before introducing adversarial examples (only matters if adversarial_train is True)
    :param log_interval: How many batches to wait before displaying training loss
    :param name: File we should save the model to after completing training
    """

    data_size = [1,28,28]
    if use_cifar:
        data_size = [3,32,32]
    batch_view_size = [64] + data_size

    if use_cifar:
        negative_batch_size = mnist_batch_size
    else:
        negative_batch_size = cifar_batch_size

    
    n_iter = 0
    for epoch in range(num_epochs):
        torch.save(new_model.state_dict(), output_directory + name)
        new_model.train()
        for batch_idx, (data, target) in enumerate(train_data_loader):
            original_data, original_target = data.clone(), target.clone()

            if closed_loop:
                example_indices = np.random.choice(range_examples, 64, replace = False)
                examples_x = adversarial_x[example_indices].view(*batch_view_size)
                examples_y = adversarial_y[example_indices]
                data = torch.cat((data, examples_x))
                examples_y = examples_y.detach()
                examples_y.requires_grad = False
                target = torch.cat((target, examples_y))


            if adversarial_train and (epoch >= num_epochs_to_wait) and (batch_idx % 100 == 0) :
                adversarial_data, adversarial_labels = generate_adversarial(mnist_set, adversarial_batch_size, new_model, criterion, attack, device=device)
                adversarial_data_size = [adversarial_batch_size] + data_size
                adversarial_data = adversarial_data.view(*adversarial_data_size)
                data = torch.cat((data, adversarial_data))
                target = torch.cat((target, adversarial_labels))


            if gaussian_batch_size > 0:
                gaussian_size = gaussian_batch_size
                gaussian_dataset = generate_gaussian(gaussian_batch_size, mnist = (not use_cifar))
                gaussian_data, gaussian_labels = gaussian_dataset[0], gaussian_dataset[1]

            if background_batch_size > 0:
                background_dataset = generate_background(background_batch_size)
                background_data, background_labels = background_dataset



            if ((mnist_batch_size > 0) and use_cifar) or ((cifar_batch_size > 0) and (not use_cifar)):
                indices = np.random.choice(range(len(negative_dataset[0])), negative_batch_size, replace= False)
                negative_data = negative_dataset[0][indices]
                negative_labels = negative_dataset[1][indices]
                data = torch.cat((data, negative_data))
                target = torch.cat((target, negative_labels))

            if fashion_batch_size > 0:
                indices = np.random.choice(range(len(fashion_dataset[0])), fashion_batch_size, replace = False)
                fashion_data = fashion_dataset[0][indices]
                fashion_labels = fashion_dataset[1][indices]
                data = torch.cat((data, fashion_data))
                target = torch.cat((target, fashion_labels))



            if convex_batch_size > 0:
                convex_arguments = [train_set]
                if gaussian_batch_size > 0:
                    convex_arguments.append(gaussian_dataset)

                if negative_batch_size > 0:
                    convex_arguments.append(negative_dataset)
                if fashion_batch_size > 0:
                    convex_arguments.append(fashion_dataset)



                convex_data, convex_labels = generate_convex_dataset(convex_arguments, mapping,  convex_batch_size, mnist = (not use_cifar))
                convex_data_size = [convex_batch_size] + data_size
                convex_data = convex_data.view(*convex_data_size)
                data = torch.cat((data, convex_data))
                target = torch.cat((target, convex_labels))


            if background_batch_size > 0:
                background_data_size = [background_batch_size] + data_size
                background_data = background_data.view(*background_data_size)
                data = torch.cat((data, background_data))
                target = torch.cat((target, background_labels))


            if gaussian_batch_size > 0:
                gaussian_data_size = [gaussian_size] + data_size
                gaussian_data = gaussian_data.view(*gaussian_data_size)
                data = torch.cat((data, gaussian_data))
                target = torch.cat((target, gaussian_labels))

            if letter_batch_size > 0:
                new_letter_x, new_letter_y = sample_dataset(letter_dataset_x, letter_dataset_y, letter_batch_size)
                data = torch.cat((data, new_letter_x))
                y_shape = new_letter_y.size()
                target = torch.cat((target, new_letter_y.float()))


            if len(data) > clean_batch_size:
                non_mnist_data = data.clone()
                non_mnist_target = target.clone()
                non_mnist_data = non_mnist_data[clean_batch_size:]
                non_mnist_target = non_mnist_target[clean_batch_size:]

            r = torch.randperm(len(data))
            data = data[r]
            target = target[r]

            data, target = data.to(device), target.to(device)

            
            optimizer.zero_grad()
            output = new_model(data)
            new_model.eval()
            only_mnist_output = new_model(original_data.to(device))
            only_mnist_loss = nn.KLDivLoss()(only_mnist_output, original_target.to(device)) * 10


            if len(data) > clean_batch_size:
                non_mnist_output = new_model(non_mnist_data.to(device))
                non_mnist_loss = nn.KLDivLoss()(non_mnist_output, non_mnist_target.to(device))
            new_model.train()

            loss = nn.KLDivLoss()(output, target) * 10
            Writer.add_scalar('data/loss', loss.item(), n_iter)
            n_iter += 1
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print('{{"metric": "mnist only loss", "value": {}, "step": {}}}'.format(only_mnist_loss.item(), n_iter))
                print('{{"metric": "loss", "value": {}}}'.format(loss.item()))
                #if len(data) > clean_batch_size:
                #    print('{{"metric": "non mnist loss", "value": {}}}'.format(non_mnist_loss.item()))
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_data_loader.dataset),100. * batch_idx / len(train_data_loader), loss.item()))

            
            
            
        new_model.eval()
        test_loss = 0
        correct = 0

        next_loader = test_data_loader

        if random_labels:
            next_loader = test_data_loader
        with torch.no_grad():
            for data, target in next_loader:
                data, target = data.to(device), target.to(device)
                output = new_model(data)
                original_target = target
                target = target.view(target.size()[0], 1)
                test_onehot = torch.FloatTensor(len(target), 10).to(device)
                test_onehot.zero_()
                target = test_onehot.scatter_(1, target, 1).to(device)
                test_loss += criterion(output, target)
                pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
                correct += pred.eq(original_target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_data_loader.dataset),100. * correct / len(test_data_loader.dataset)))
    torch.save(new_model.state_dict(), output_directory +  name)
    Writer.close()
    return new_model
# ------------------------------------------------------------------------------------------------------------------

model_name = args.model_name

if model_name == None:
    model_name = 'clean{}_convex{}_negative{}_gaussian{}_cifar{}_letter{}adversarial{}'.format(mnist_batch_size, convex_batch_size, fashion_batch_size,  
                                                                                            gaussian_batch_size, cifar_batch_size, letter_batch_size, adversarial_batch_size)
# ------------------------------------------------------------------------------------------------------------------

if closed_loop:
    adversarial_x_file_name = '/tmp_data/' +  model_name + '_adversarial_examples.pt'
    adversarial_y_file_name = '/tmp_data/' + model_name + '_adversarial_labels.pt'

    adversarial_x = torch.load(adversarial_x_file_name)
    adversarial_y = torch.load(adversarial_y_file_name)

    range_examples = range(len(adversarial_x))
# ------------------------------------------------------------------------------------------------------------------
if restart:
    model_to_train.load_state_dict(torch.load(model_name))

# ------------------------------------------------------------------------------------------------------------------
train_set = PositiveDataSet(train_set, mnist = (not use_cifar))
test_test = PositiveDataSet(test_set, mnist = (not use_cifar))

train_loader = torch.utils.data.DataLoader(train_set, batch_size=clean_batch_size, shuffle=True)

# ------------------------------------------------------------------------------------------------------------------

if args.optimizer == 'Adam':
    optimizer = optim.SGD(model_to_train.parameters(), lr=lr,weight_decay=5e-4)
else:
    optimizer = optim.SGD(model_to_train.parameters(), lr=lr, momentum=momentum,weight_decay=5e-4)

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# ------------------------------------------------------------------------------------------------------------------
test_loader = torch.utils.data.DataLoader(test_set,batch_size=test_batch_size, shuffle=True, **kwargs)
# ------------------------------------------------------------------------------------------------------------------ 
if args.use_cifar:
    negative_dataset = generate_negative_dataset(mnist_train_set, num_datapoints = len(mnist_train_set), mnist = (not use_cifar))
    negative_batch_size = mnist_batch_size
else:
    negative_dataset = generate_cifar_dataset(cifar_train_set, num_datapoints = len(cifar_train_set))
    negative_batch_size = cifar_batch_size
# ------------------------------------------------------------------------------------------------------------------

model_to_train = train_model(model_to_train, optimizer, criterion, train_loader, test_loader, num_epochs, SummaryWriter(),
                mnist_batch_size, convex_batch_size, fashion_batch_size, gaussian_batch_size, cifar_batch_size, letter_batch_size,
                 background_batch_size, adversarial_train = adversarial_train, adversarial_batch_size = adversarial_batch_size ,
                 num_epochs_to_wait = num_epochs_to_wait, name = model_name)