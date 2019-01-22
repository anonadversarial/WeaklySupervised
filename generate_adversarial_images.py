import torch
from attacks import * 
from models import MNIST_Net, ResNet18
import argparse
from tqdm import tqdm
import os 
from data_generator import PositiveDataSet
from torchvision import datasets
from torch.distributions.categorical import Categorical
from torch.distributions.kl import kl_divergence
import scipy.misc
from PIL import Image


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# ------------------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='flags for generating images.')
parser.add_argument('--model_name', type=str, help = 'which model to test')
parser.add_argument('--epsilon', type=float, default=0.00001, help='epsilon to use for adversarial attacks')
parser.add_argument('--gpu_enabled',type=str,default=False,help='whether to remap everything to CPU before reloading')
parser.add_argument('--input_directory',type=str, default='trained_models/',help='input directory')
parser.add_argument('--mnist_directory', type=str,default='mnist_data', help='mnist directory')
parser.add_argument('--target_confidence', type=float, default=0.9, help='target confidence')
parser.add_argument('--selection_confidence', type=float, default=0.6, help='selection-confidence')
parser.add_argument('--attack', type=str, default='pgd_attack', help='which attack to run')
parser.add_argument('--save_examples', type=bool, default=False, help='whether we are saving examples or not')
parser.add_argument('--num_examples', type=int, default=1000, help='number of adversarial examples to generate')
parser.add_argument('--output_directory', type=str, default=None, help='where to save the model')
parser.add_argument('--plot_confidences', type=bool, default=False, help='whether to plot confidences')
parser.add_argument('--plot_gradients', type=bool, default=False, help='whether to plot confidences')
parser.add_argument('--num_images', type=int, default=5, help='how many images to generate')
parser.add_argument('--test_noise', type=bool, default=False,help='whether to test noise')
parser.add_argument('--test_mnist',type=bool, help='whether we are testing MNIST or CIFAR')
parser.add_argument('--cifar_directory',type=str, default='cifar_data',help='cifar directory')
parser.add_argument('--step_threshold', type=int, default=1000, help='how many steps to use when calculating effort')

args = parser.parse_args()
# ------------------------------------------------------------------------------------------------------------------
output_directory = args.output_directory
if output_directory == None:
	output_directory = os.getcwd() + '/'
# ------------------------------------------------------------------------------------------------------------------
model_name = args.model_name
gpu_enabled = args.gpu_enabled
epsilon = args.epsilon
input_directory = args.input_directory
target_confidence = args.target_confidence
selection_confidence = args.selection_confidence
num_images = args.num_images

step_threshold = args.step_threshold

save_examples = args.save_examples
num_examples = args.num_examples
test_noise = args.test_noise
# ------------------------------------------------------------------------------------------------------------------
if args.attack not in ['fgsm_targeted', 'pgd_attack']:
	raise RuntimeError('Attack method is not implemented')
attack = eval(args.attack)
# ------------------------------------------------------------------------------------------------------------------
plot_confidences = args.plot_confidences
plot_gradients = args.plot_gradients

mnist_root = args.mnist_directory
# ------------------------------------------------------------------------------------------------------------------
if args.test_mnist:
	mnist_train_set = datasets.MNIST(root=mnist_root, train=True, download=False)
	test_set = PositiveDataSet(mnist_train_set, mnist = args.test_mnist)
	data_size = [1,28,28]
else:
	cifar_train_set = datasets.CIFAR10(root=args.cifar_directory,train=True,download=False)
	test_set = PositiveDataSet(cifar_train_set, mnist = args.test_mnist)
	data_size = [3,32,32]
# ------------------------------------------------------------------------------------------------------------------


loss_function = nn.KLDivLoss()


adversarial_path = model_name + "_noise_"

if not save_examples:
	if not os.path.exists(adversarial_path):
		os.mkdir(adversarial_path)


if args.test_mnist:
	model_to_test = MNIST_Net().to(device)

else:
	model_to_test = ResNet18().to(device)


extension = '_effort_values_' + '.txt'


if gpu_enabled:
	model_to_test.load_state_dict(torch.load(input_directory + model_name))
else:
	model_to_test.load_state_dict(torch.load(input_directory + model_name, map_location={'cuda:0': 'cpu'}))

# ------------------------------------------------------------------------------------------------------------------
if save_examples:
	# Note: When we are saving the examples and labels we are not saving the images like below
	new_lst = [num_examples] + data_size
	new_data_tensor = torch.randn(*new_lst)
 	new_label_tensor = torch.zeros(num_examples, 10)
 	efforts = []
 	num_confident_examples = 0
 	num_target = 0
 	num_over_half = 0
 	for iteration in tqdm(range(num_examples)):
 		result = test_model_effort(model_to_test, attack, mnist_set, target_confidence, selection_confidence, loss_function, step_threshold = step_threshold, epsilon = epsilon, device=device)
 		new_effort = result[1]
 		efforts.append(new_effort)
 		adversarial_example = result[3]
 		ground_truth_example = result[2]
 		new_data_tensor[iteration] = adversarial_example.view(*data_size).to(device)
 		ipt_label = torch.exp(model_to_test(adversarial_example)).to(device)
 		sorted_label = torch.sort(ipt_label)[0]
 		#import pdb; pdb.set_trace()
 		largest, second_largest = sorted_label[0][-1].item(), sorted_label[0][-2].item()
 		if largest > 0.5:
 			num_over_half += 1
 		if largest >= target_confidence:
 			num_target += 1
 		if largest >= 3 * second_largest:
 			num_confident_examples += 1
 		new_label_tensor[iteration] = ipt_label
 	base_tensor_name = output_directory + model_name + '_adversarial'

# ------------------------------------------------------------------------------------------------------------------
elif test_noise:
	for iteration in range(num_images):
		result = perturb_noise(model_to_test, attack, loss_function, 0.6, step_threshold=step_threshold, mnist = args.test_mnist)
		original_probabilities, num_steps,true_example, example,  fake_label, confidences, SNR, gradients = result
		print "******"
		print 'SNR: ', SNR 
		print 'original_probabilities: ', original_probabilities
		print 'new probabilities: ', torch.exp(model_to_test(example))
		print 'fake label: ', fake_label
		print 'num steps: ', num_steps
		if plot_confidences:
			import matplotlib.pyplot as plt 
			plt.xlabel('Number of iterations')
			fig1 = plt.figure()
			ax1 = fig1.add_subplot(111)
			ax1.plot(range(len(confidences)), confidences, label = "noise" + " to " + str(fake_label))
			plt.ylabel('Example confidence')
			plt.show()

		data_size = [28,28]
		if not args.test_mnist:
			data_size = [32,32,3]
		new_noise_file_name = 'new_' + model_name + "_" +  str(fake_label) + "_" +  str(iteration) 
		original_noise_file_name = 'original_' + model_name + "_" +  str(iteration)
		scipy.misc.imsave(model_name + "_noise_" + '/' + original_noise_file_name + '.png', true_example.data.view(*data_size).numpy())
		scipy.misc.imsave(model_name + "_noise_" + '/' + new_noise_file_name + '.png', example.data.view(*data_size).numpy())
# ------------------------------------------------------------------------------------------------------------------
else:
	for iteration in range(num_images):
		result = test_model_effort(model_to_test, attack, test_set, target_confidence, selection_confidence, loss_function, step_threshold = step_threshold, epsilon = epsilon, device=device, mnist = args.test_mnist)
		#print result[0]
		#print result[1]
		original_image, adversarial_image = result[2], result[3]
		adversarial_image = adversarial_image.to(device)
		original_probs = torch.exp(model_to_test(original_image))
		ground_truth_prediction = torch.argmax(model_to_test(original_image)).item()
		prediction = torch.argmax(model_to_test(adversarial_image)).item()
		#print original_probs
		#print torch.exp(model_to_test(adversarial_image))
		gradients = result[-1]
		confidences = result[-2]
		if plot_confidences or plot_gradients:
			import matplotlib.pyplot as plt 
			plt.xlabel('Number of iterations')
		if plot_gradients:
			fig1 = plt.figure()
			ax1 = fig1.add_subplot(111)
			ax1.plot(range(len(gradients)), gradients, label = str(ground_truth_prediction) + " to " + str(prediction))
			plt.ylabel('Example gradient')
			plt.show()
		if plot_confidences:
			fig1 = plt.figure()
			ax1 = fig1.add_subplot(111)
			ax1.plot(range(len(confidences)), confidences, label = str(ground_truth_prediction) + " to " + str(prediction))
			plt.ylabel('Example confidence')
			plt.show()

		data_size = [28,28]
		if not args.test_mnist:
			data_size = [32,32,3]
		adversarial_file_name = 'adversarial_' + model_name + "_" +  str(prediction) + "_" +  str(iteration) 
		ground_truth_file_name = 'original_' + model_name + "_" +  str(iteration)
		scipy.misc.imsave(adversarial_path + '/' + adversarial_file_name + '.png', adversarial_image.data.view(*data_size).numpy())
		scipy.misc.imsave(adversarial_path + '/' + ground_truth_file_name + '.png', original_image.data.view(*data_size).numpy())