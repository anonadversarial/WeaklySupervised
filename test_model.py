import torch
import torch.nn as nn 
from torch.distributions.categorical import Categorical
from torch.distributions.kl import kl_divergence
from attacks import * 
from models import MNIST_Net, ResNet18
import argparse
from tqdm import tqdm
from data_generator import PositiveDataSet
from torchvision import datasets
import os


"""
Code to generate the effort score for models and depending on the flags plot the results
"""
# ------------------------------------------------------------------------------------------------------------------
def test_model_epsilon_range(model_name, attack, epsilon_range, num_iters, test_effort = True, ideal_SNR = 0.6, device="cpu"):
	if args.test_mnist:
		model_to_test = MNIST_Net()
	else:
		model_to_test = ResNet18()
	if gpu_enabled:
		model_to_test.load_state_dict(torch.load(input_directory + model_name))
		
	else:
		model_to_test.load_state_dict(torch.load(input_directory + model_name, map_location={'cuda:0': 'cpu'}))	
	model_to_test.to(device)
	lst_efforts = []
	for epsilon in epsilon_range:
		lst_efforts.append(test_model(model_to_test, attack, epsilon, num_iters, ideal_SNR, test_effort = test_effort, device=device))
		print 'finished testing model: {} with epsilon: {}'.format(model_name, str(epsilon))
	print lst_efforts
	np.savetxt(output_directory + model_name + extension, np.array(lst_efforts), fmt = '%f')
	return lst_efforts
# ------------------------------------------------------------------------------------------------------------------
def test_model(model_to_test, attack, epsilon, num_iters, ideal_SNR = 0.6, test_effort= True, device='cpu'):

	sum_ = 0
	results = []
	for _ in tqdm(range(num_iters)):
		if test_effort:
			partial = test_model_effort(model_to_test, attack, mnist_set, target_confidence, selection_confidence, nn.KLDivLoss(), step_threshold= step_threshold, epsilon = epsilon, device=device, mnist = test_mnist)[1]
		else:
			partial = perturb_noise(model_to_test, attack, nn.KLDivLoss(), ideal_SNR, step_threshold = step_threshold, epsilon = epsilon)
			partial = partial[1] * partial[6]
		results.append(partial)
		sum_ += partial
	return sum_
# ------------------------------------------------------------------------------------------------------------------

# This is the epsilon range we use to test the attack
epsilon_range = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.8, 1, 1.5, 2]



parser = argparse.ArgumentParser(description='flags for testing adversarial robustness.')
parser.add_argument('--attack', type=str, default='pgd_attack', help='which attack to run')
parser.add_argument('--model_name', type=str, default='model_convex', help='name of model to load')
parser.add_argument('--target_confidence', type=float, default=0.9, help='target confidence')
parser.add_argument('--selection_confidence', type=float, default=0.6, help='selection-confidence')
parser.add_argument('--num_iters', type=int, default=100, help='number of iterations to run for')
parser.add_argument('--epsilon', type=float, default=0.1, help='epsilon to use for adversarial attacks')
parser.add_argument('--model_names', nargs='+', help = 'which models to test')
parser.add_argument('--plot', type=bool, default=False, help='whether to plot the model names given')
parser.add_argument('--test_noise', type=bool, default=False)
parser.add_argument('--input_directory',type=str, default='trained_models/',help='input directory')
parser.add_argument('--output_directory',type=str, default='plot_effort_values/', help='output directory')
parser.add_argument('--mnist_directory', type=str,default='mnist_data', help='mnist directory')
parser.add_argument('--cifar_directory',type=str,default='cifar_data',help='cifar directory')
parser.add_argument('--test_mnist',type=bool,default=True,help='whether we are testing MNIST')
parser.add_argument('--gpu_enabled',type=str,default=False,help='whether to remap everything to CPU before reloading')
parser.add_argument('--ideal_SNR', type=float, default=0.6, help='minimum SNR to use for noise pertubation test')
parser.add_argument('--test_effort',type=bool,default=True,help='whether we are testing effort')
parser.add_argument('--step_threshold', type=int, default=1000, help='how many steps to use when calculating effort')
args = parser.parse_args()
# ------------------------------------------------------------------------------------------------------------------
if args.attack not in ['fgsm_targeted', 'pgd_attack']:
	raise RuntimeError('Attack method is not implemented')
attack = eval(args.attack)
# ------------------------------------------------------------------------------------------------------------------
gpu_enabled = args.gpu_enabled

input_directory = args.input_directory
output_directory = args.output_directory
# ------------------------------------------------------------------------------------------------------------------
model_name = args.model_name
target_confidence = args.target_confidence
selection_confidence = args.selection_confidence
num_iters = args.num_iters
epsilon = args.epsilon
model_names = args.model_names
plot_results = args.plot
step_threshold = args.step_threshold
test_effort = args.test_effort
test_noise = args.test_noise
test_mnist = args.test_mnist
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
ideal_SNR = args.ideal_SNR
# ------------------------------------------------------------------------------------------------------------------
mnist_root = args.mnist_directory
if not os.path.exists(mnist_root):
	os.mkdir(mnist_root)
dataset = datasets.MNIST(root=mnist_root, train=True, download=False)
mnist_set = PositiveDataSet(dataset)
# ------------------------------------------------------------------------------------------------------------------
extension = '_effort_values' + '.txt'

if test_effort:
	for model_name in model_names:
		new_lst_efforts = test_model_epsilon_range(model_name, attack, epsilon_range, num_iters, device=device)
		print 'finished generating efforts for model: ' + model_name
# ------------------------------------------------------------------------------------------------------------------
if test_noise:
	extension = '_noise_values_' + '.txt'
	for model_name in model_names:
		new_lst_efforts = test_model_epsilon_range(model_name, attack, epsilon_range, num_iters, test_effort = False, ideal_SNR = ideal_SNR, device=device)
		print 'finished generating noise pertubations for model: ' + model_name
# ------------------------------------------------------------------------------------------------------------------
if plot_results:
	import matplotlib.pyplot as plt
	params = {'legend.fontsize': 5,
		  	'legend.handlelength': 1}
	plt.rcParams.update(params)
	number_of_plots = len(model_names)
	fig1 = plt.figure()
	ax1 = fig1.add_subplot(111)
	for model_scores in model_names:
		scores = np.loadtxt(output_directory + model_scores + extension, dtype=float)
		ax1.plot(epsilon_range, scores, label = model_scores)

	#Shrink current axis by 20%
	box = ax1.get_position()
	ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])

	# Put a legend to the right of the current axis
	ax1.legend(loc='center left',  bbox_to_anchor=(1, 0.5), prop={'size': 8})
	colormap = plt.cm.nipy_spectral
	ax1.set_color_cycle([colormap(i) for i in np.linspace(0, 1,number_of_plots + 1)])
	plt.xlabel('Epsilon')
	plt.ylabel('Effort')
	plt.show()