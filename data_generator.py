import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from torch.utils import data as pytorch_data
import copy
#import matplotlib.pyplot as plt


"""
Utility functions for generating and loading data
"""
# ------------------------------------------------------------------------------------------------------------------
def sample_dataset(dataset_x, dataset_y, num_datapoints):
	"""
	:param dataset_x: the training data 
	:param dataset_y: the training labels
	:param num_datapoints: how many datapoints to draw
	"""
	indices = np.random.choice(range(len(dataset_x)), num_datapoints, replace = False)
	new_dataset_x = dataset_x[indices]
	new_dataset_y = dataset_y[indices]
	return new_dataset_x, new_dataset_y
# ------------------------------------------------------------------------------------------------------------------
def generate_adversarial(train_set, num_datapoints, model_to_test, loss_function, attack, epsilon = 0.2, device="cpu", mnist=True):
	"""
	Generates adversarial examples given a train set as well as the model we are testing 
	:param training_set: which set to draw examples from 
	:param num_datapoints: how many samples to draw 
	:param model: which model to test 
	:param loss_function: which loss function to use
	:param attack: which attack to use 
	:param epsilon: epsilon we should use when perturbing the example 
	:param device: whether we are running on CPU or GPU
	:param mnist: whehther we are using CIFAR or MNIST 
    """

	data_size = [1,28,28]
	if not mnist:
		data_size = [3,32,32]
	tmp_size = [num_datapoints] + data_size
	new_data_tensor = torch.randn(*tmp_size)
	new_label_tensor = torch.zeros(num_datapoints, 10)
	indices = np.random.choice(range(len(train_set)), num_datapoints, replace = False)
	num_examples_generated = 0
	model_cp = copy.deepcopy(model_to_test)
	for p in model_cp.parameters():
		p.requires_grad = False
	model_cp.eval()
	for idx in tqdm(indices):
		ipt = train_set[idx]
		view_size = [1] + data_size
		ipt_tensor = torch.Tensor(np.array(ipt[0])).view(*data_size).to(device)
		ipt_label = torch.exp(model_cp(ipt_tensor)).to(device)
		perturbed_example = attack(model_cp, loss_function, ipt_tensor, epsilon = epsilon, steps = 2000, device=device)
		new_data_tensor[num_examples_generated] = perturbed_example[0].view(*data_size).to(device)
		new_label_tensor[num_examples_generated] = ipt_label
		num_examples_generated += 1
	return new_data_tensor, new_label_tensor

		
# ------------------------------------------------------------------------------------------------------------------
def generate_letter_dataset(dataset):
	"""
	Function to generate the EMNIST dataset 
	:param dataset: EMNIST dataset
	"""
	transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
	new_dataset_x = []
	new_dataset_y = []
	for index in tqdm(range(len(dataset))):
		new_dataset_x.append(transform(dataset[index][0]))
		new_dataset_y.append(dataset[index][1])
	return new_dataset_x, new_dataset_y
# ------------------------------------------------------------------------------------------------------------------

def get_mapping(dataset, num_classes = 10, mnist=True):
	"""
	Generates a mapping from training classes to indices which have those classes 
	Used to make sure no two of the same image are drawn when making convex combinations 
	:param dataset: which dataset we are using 
	:param num_classes: how many classes the dataset has 
	:param mnist: whether we are working with MNIST or CIFAR
	"""
	mapping = {}
	for i in range(num_classes):
		mapping[i] = []
	for index, elem in enumerate(dataset):
		if mnist:
			label = elem[1].item()
		else:
			label = elem[1]
		mapping[label].append(index)
	return mapping

# ------------------------------------------------------------------------------------------------------------------
def generate_negative_dataset(negative_dataset, num_datapoints, mnist = True):
	"""
	Generates a pytorch dataset from negative data of MNIST dimension (for example fashion MNIST)
	:param negative_dataset: the dataset to convert 
	:param num_datapoints: how many samples to draw 
	:param mnist: whether we are working with MNIST or CIFAR
	"""
	new_labels = torch.ones(num_datapoints, 10) * 0.1
	if mnist:
		new_data = torch.ones(num_datapoints, 1, 28, 28)
		transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
	else:
		new_data = torch.ones(num_datapoints,3,32,32)
		transform = transforms.Compose([transforms.Pad(2), transforms.ToTensor()])
	indices = np.random.choice(range(len(negative_dataset)), num_datapoints, replace=False)
	for index in indices:
		if mnist:
			new_data[index] = transform(negative_dataset[index][0])
		else:
			new_image = transform(negative_dataset[index][0])
			new_image = torch.cat((new_image, new_image, new_image))
			new_data[index] = new_image
	return new_data, new_labels


# ------------------------------------------------------------------------------------------------------------------
def generate_cifar_dataset(cifar_dataset, num_datapoints = 50000):
	"""
	Generates a negative CIFAR dataset when training on MNIST
	:param cifar_dataset: CIFAR dataset 
	:param num_datapoints: how many samples to draw

	"""


	transform = transforms.Compose([transforms.Grayscale(), 
									transforms.CenterCrop(28), 
									transforms.ToTensor(), 
									transforms.Normalize((0.1307,), (0.3081,))])
	new_labels = torch.ones(num_datapoints, 10) * 0.1
	new_data = torch.ones(num_datapoints, 1, 28, 28)
	for index in tqdm(range(num_datapoints)):
		new_data[index] = transform(cifar_dataset[index][0])
	return new_data, new_labels

# ------------------------------------------------------------------------------------------------------------------
def extract_training_images(train_set, train_map, num_images):
	"""
	Extract training images to make a convex combination. The map is used 
	to insure we never draw two images from the same class 
	:param train_set: which training set we are using 
	:param train_map: mapping between classes and which indices have those classes 
	:param num_images: how many images to extract
	"""
	classes = np.random.choice(range(10), num_images, replace=False)
	images = []
	labels = []
	for idx, sub_class in enumerate(classes):
		new_index = np.random.choice(train_map[sub_class])
		new_image = train_set[new_index][0].numpy()
		images.append(new_image)
		labels.append(train_set[new_index][1].numpy())
	return images, labels
 
# ------------------------------------------------------------------------------------------------------------------
def generate_new_combination(datasets, train_map, min_number_images = 2, max_number_images= 5, mnist=True):
	"""
	Function to generate a convex combination from different datasets
	:param datasets: which datasets to make convex combinations of 
	:param train_map: mapping between classes and which indices have those classes 
	:param min_number_images: minimum number of images for convex combination 
	:param max_number_images: maximum number of images for convex combination 
	:param mnist: whether we are using MNIST or CIFAR
	"""
	num_images = np.random.choice(range(min_number_images, max_number_images), 1)[0]
	dataset_distribution = np.round(np.random.dirichlet(range(1, len(datasets) + 1))* num_images).astype(int)
	dataset_distribution = dataset_distribution[::-1]
	weights = np.random.dirichlet(range(1, np.sum(dataset_distribution) + 2))
	images, labels =  [], []
	for idx, elem in enumerate(dataset_distribution):
		# Note: In the code we make it so that the first class is always our training class because we need to handle it separately
		if elem != 0:
				if idx == 0:
					images, labels = extract_training_images(datasets[idx], train_map, elem)
				else:
					dataset_indices = np.random.choice(range(len(datasets[idx][0])), elem, replace=False)
					new_images = [datasets[idx][0][elem].numpy() for elem in dataset_indices]
					new_labels = [datasets[idx][1][elem].numpy() for elem in dataset_indices]
					images += new_images
					labels += new_labels
		
	new_combination_image = [np.dot(images[i], weights[i]) for i in range(len(labels))]
	new_combination_image = np.sum(np.array(new_combination_image), axis=0)
	if mnist:
		new_combination_image = torch.from_numpy(new_combination_image).view(1,28,28)
	else:
		new_combination_image = torch.from_numpy(new_combination_image).view(3,32,32)
	if mnist:
		transformation = transforms.Normalize((0.1307,), (0.3081,))
	else:
		transformation = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	new_combination_image = transformation(new_combination_image)
	new_combination_label = [np.dot(labels[i], weights[i]) for i in range(len(labels))]
	new_combination_label = torch.from_numpy(np.sum(np.array(new_combination_label), axis=0))
	return new_combination_image, new_combination_label
		
# ------------------------------------------------------------------------------------------------------------------
def generate_convex_dataset(datasets, train_map, num_datapoints,mnist=True):
	"""
	Generates a PyTorch convex combination dataset
	"""
	if mnist:
		data_tensor = torch.zeros(num_datapoints, 1, 28, 28)
	else:
		data_tensor = torch.zeros(num_datapoints, 3, 32, 32)
	label_tensor = torch.zeros(num_datapoints, 10)
	for idx in range(num_datapoints):
		label_tensor = torch.zeros(num_datapoints, 10)
	for idx in range(num_datapoints):
		data_tensor[idx], label_tensor[idx] = generate_new_combination(datasets, train_map, mnist = mnist)
	return data_tensor, label_tensor


# ------------------------------------------------------------------------------------------------------------------
def generate_gaussian(num_datapoints, mnist = True):
	"""
	Generates a gaussian noise dataset with size dependent on whether we are training on CIFAR or MNIST
	:param num_datapoints: how many datapoints to use 
	:param mnist: whether we are using MNIST or CIFAR 
	"""
	if mnist:
		new_data_tensor = torch.randn(num_datapoints, 28, 28)
	else:
		new_data_tensor = torch.randn(num_datapoints, 3, 32, 32)
	new_label_tensor = torch.ones(num_datapoints, 10) * 0.1
	return new_data_tensor, new_label_tensor 



"""
Note: The dataset classes below are more useful if you want to make a fixed dataset 
with this kind of data not if you want to generate it every batch like we do, but could
still be useful
"""

# ------------------------------------------------------------------------------------------------------------------
class ConvexDataSet(pytorch_data.Dataset):
		def __init__(self, datasets, train_map, num_datapoints, set_tensors = False, x = None, y = None):
			"""
			:param datasets: which datsets to use for convex combinations 
			:param train_map: a map from training classes to indices having those classes
			:param num_datapoints: how many datapoints to use 
			:param set_tensors: whether we are preloading an already generated dataset 
			:param x: pregenerated data (ignored if set_tensors is False)
			:param y: pregenerated labels (ignored if set_tensors is False)
			"""
			if set_tensors:
				self.new_data_tensor = x 
				self.new_label_tensor = y
			else:
				self.new_data_tensor, self.new_label_tensor = generate_convex_dataset(datasets, train_map, num_datapoints)
		def __len__(self):
			return self.new_data_tensor.size()[0]
		def __getitem__(self, idx):
			return (self.new_data_tensor[idx].view(1, 28, 28), self.new_label_tensor[idx])

# ------------------------------------------------------------------------------------------------------------------
class NegativeDataSet(pytorch_data.Dataset):
		def __init__(self, negative_dataset, num_datapoints=1000):
			"""
			:param negative_dataset: the dataset we are converting 
			:param num_datapoints: how many datapoints to use
			"""
			self.new_data_tensor, self.new_label_tensor = generate_negative_dataset(negative_dataset, num_datapoints)

		def __len__(self):
			return self.new_data_tensor.size()[0]
		def __getitem__(self, idx):
			return self.new_data_tensor[idx].view(1, 28, 28), self.new_label_tensor[idx]

# ------------------------------------------------------------------------------------------------------------------
class GaussianDataSet(pytorch_data.Dataset):
	def __init__(self, num_datapoints):
		"""
		:param num_datapoints: how many datapoints to use
		"""
		self.new_data_tensor, self.new_label_tensor = generate_gaussian(num_datapoints)
	def __len__(self):
		return self.new_data_tensor.size()[0]
	def __getitem__(self, idx):
		return (self.new_data_tensor[idx].view(1, 28, 28), self.new_label_tensor[idx])

# ------------------------------------------------------------------------------------------------------------------
class PositiveDataSet(pytorch_data.Dataset):
	def __init__(self, positive_dataset, mnist=True):
		"""
		:param positive_dataset: which dataset we are converting
		"""
		self.positive_dataset = positive_dataset
		self.mnist = mnist
		self.length = len(positive_dataset)
		self.transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
		if not mnist:
			self.transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
   
	def __len__(self):
		return self.length 
	def __getitem__(self, idx):
		data_size = [1,28,28]
		if not self.mnist:
			data_size = [3,32,32]
		data_tensor = self.positive_dataset[idx][0]
		label_tensor = self.positive_dataset[idx][1]
		label_tensor = torch.Tensor([label_tensor]).long().view(1, 1)
		onehot = torch.zeros(1, 10)
		onehot = onehot.scatter_(1, label_tensor, 1)
		return self.transform(data_tensor).view(*data_size), onehot.float().view(10,)
