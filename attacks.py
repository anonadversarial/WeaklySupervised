import torch 
import torch.nn as nn
import numpy as np 
import copy
from torch.autograd.gradcheck import zero_gradients
from torch.distributions.categorical import Categorical
from torch.distributions.kl import kl_divergence

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
# ------------------------------------------------------------------------------------------------------------------
def perturb_noise(model, attack, loss_function, target_similarity, return_true_example = True, steps = 10000, mnist=True, use_confidence=True, epsilon = 0.0001, device="cpu"):
    """
    This function is used to test how easily we can take gaussian noise and perturb it so that our model thinks its a digit with high certainty
    :param model: which model to test 
    :param loss_function: which loss function we are using
    :param attack: which attakck to run
    :param target_similarity: lower bound for SNR
    :param return_true_example: whether to return_true_example for comparison purpose 
    :param target: which target class we are using, if none we are running an untargeted attack
    :param steps: how many steps to run the attack for
    :param mnist: whehther we are using CIFAR or MNIST 
    :param use_confidence: whether to use model confidence or SNR in perturbed example
    :param epsilon: epsilon we should use when perturbing the example 
    :param device: whether we are running on CPU or GPU
    """
    example = torch.randn(1,1,28,28)
    if not mnist:
     example = torch.randn(1,3,32,32)
    true_example = torch.from_numpy(np.copy(example)).to(device)
    original_probabilities = torch.exp(model(true_example))
    fake_label = np.random.choice(range(10),1)[0]
    num_steps = 0
    least_likely_tensor = torch.Tensor([fake_label]).long().view(1,1)
    onehot = torch.zeros(1,10).to(device)
    onehot = onehot.scatter_(1, least_likely_tensor, 1).to(device)
    perturbed_enough = False 
    confidences = []
    gradients = []
    model.eval()
    while (not perturbed_enough) and (num_steps <= steps):
        example, prediction, new_gradients = attack(model, loss_function, example, target = fake_label, steps=1, epsilon = epsilon)
        probabilities = torch.exp(model(example)).view(10,)
        confidence = probabilities[fake_label].item()
        confidences.append(confidence)
        gradients.extend(new_gradients)
        condition = (confidence >= 0.9)
        size = 28 * 28
        if not mnist:
         size = 3 * 32 * 32
        SNR = torch.dot(true_example.view(size), example.view(size)) / (torch.norm(true_example) * torch.norm(example))
        if not use_confidence:
            condition = (SNR <= target_similarity)    
        if condition:
            perturbed_enough = True
        num_steps += 1
    if return_true_example:
        return original_probabilities, num_steps,true_example, example,  fake_label, confidences, SNR, gradients
    return num_steps * SNR
# ------------------------------------------------------------------------------------------------------------------
def pgd_attack(model_to_test, loss_function, example, target = None, steps=2000,  step_size = 0.1, epsilon=0.4, device="cpu", use_confidence = True, target_confidence = 0.9):
    """
    Significant lifting from https://github.com/wanglouis49/pytorch-adversarial_box/blob/master/adversarialbox/attacks.py

    :param model_to_test: which model to test 
    :param loss_function: which loss function we are using
    :param example: which example we are using
    :param target: which target class we are using, if none we are running an untargeted attack
    :param steps: how many steps to run the attack for
    :param step_size: how much to scale the gradient by when computing the adversarial ttack
    :param epsilon: epsilon we should use when perturbing the example 
    :param mnist: whehther we are using CIFAR or MNIST 
    :param device: whether we are running on CPU or GPU
    """
    onehot = torch.zeros(1,10).to(device)
    model = copy.deepcopy(model_to_test).to(device)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    if use_cuda:
        raw_example = torch.from_numpy(np.copy(example.data.cpu().numpy())).to(device)
    else:
        raw_example = torch.from_numpy(np.copy(example.data.numpy())).to(device)

    ground_truth_output = model(example).to(device)
    ground_truth = torch.argmax(ground_truth_output).item()
    ground_truth_tensor = torch.Tensor([ground_truth]).long().view(1,1).to(device)
    onehot = onehot.scatter_(1, ground_truth_tensor, 1).to(device)

    misclassified = False 

    num_steps = 0
    gradients = []

    if target != None:
        least_likely_tensor = torch.Tensor([target]).long().view(1,1).to(device)
        onehot = torch.zeros(1,10).to(device)
        onehot = onehot.scatter_(1, least_likely_tensor, 1).to(device)

    while (not misclassified) and (num_steps < steps):
        example = example.to(device)
        zero_gradients(example)
        example.requires_grad = True 
        output = model(example).to(device)
        prediction = torch.argmax(output).item()
        probabilities = torch.exp(output).view(10,)
        incorrect_confidence = probabilities[target]
        condition = (prediction != target)

        if use_confidence:
            condition = (incorrect_confidence < target_confidence)

        if condition:
            loss = loss_function(output, onehot)
            loss.backward()
            grad = example.grad.data
            gradients.append(torch.norm(grad).item())
            if use_cuda:
                example = torch.from_numpy(np.copy(example.cpu().detach()))
            else:
                example = torch.from_numpy(np.copy(example.detach()))
            if target != None:
                example -= step_size * np.sign(grad)
            else:
                example += step_size * np.sign(grad)
            example = np.clip(example, raw_example - epsilon, raw_example + epsilon)
            example = np.clip(example, 0, 1)
            num_steps += 1
        else:
            misclassified = True 
    return [example.to(device), prediction, gradients]
# ------------------------------------------------------------------------------------------------------------------
def fgsm_targeted(model, loss_function, example, target = None, mnist=True, epsilon=0.1, step_size=1, steps=2000):
    """
    This function calculates the effort for a given model and example 
    :param model: which model to test 
    :param loss_function: which loss function we should use 
    :param example: which example we are using 
    :param target: which target class we are using, if none we are running an untargeted attack
    :param step_size: how much to scale the gradient by when computing the adversarial attack
    :param steps: how many steps of the attack we should limit ourselves to 
    :param epsilon: epsilon we should use when perturbing the example 
    :param mnist: whehther we are using CIFAR or MNIST 
    """
    onehot = torch.zeros(1, 10)
    ground_truth_output = model(example)
    ground_truth = torch.argmax(ground_truth_output).item()
    least_likely = torch.argmin(ground_truth_output).item()
    model.eval()
    if target != None:
        least_likely = target
    least_likely_tensor = torch.Tensor([least_likely]).long().view(1, 1)
    onehot = onehot.scatter_(1, least_likely_tensor, 1)
    misclassified = False
    num_steps = 0

    raw_pertubation =torch.zeros(1,1,28,28)
    if not mnist:
     raw_pertubation = torch.zeros(1,3,32,32)
    while not misclassified and num_steps < steps:
        zero_gradients(example)
        example.requires_grad = True
        output = model(example)
        prediction = torch.argmax(output).item()
        if prediction != least_likely:
            loss = loss_function(output, onehot)
            loss.backward()
            example_gradient = step_size * torch.sign(example.grad.data)
            pertubation = torch.clamp(example.data - epsilon * example_gradient, 0, 1)
            raw_pertubation += epsilon * example_gradient
            example = pertubation
            
            num_steps += 1
        else:
            misclassified = True
    return example, prediction
# ------------------------------------------------------------------------------------------------------------------
def calculate_effort(model, example, true_probabilities, criterion, target_confidence, attack, step_threshold = 20000, epsilon = 0.01, device="cpu"):
    """
    This function calculates the effort for a given model and example 
    :param model: which model to test 
    :param attack: which attack to use 
    :param training_set: which set to draw examples from 
    :param target_confidence: what confidence we should get our model to classify the adversarial example with 
    :param loss_function: which loss function we should use 
    :param step_size: how many steps of the attack we should limit ourselves to 
    :param epsilon: epsilon we should use when perturbing the example 
    :param mnist: whehther we are using CIFAR or MNIST 
    :param device: whether we are running on CPU or GPU
    """
    true_example = example.clone()
    true_distribution = Categorical(probs = true_probabilities)
    true_label = torch.argmax(model(example)).item()
    bad_labels = range(10)
    bad_labels.remove(true_label)
    fake_label = np.random.choice(bad_labels, 1)[0]
    misclassified = False 
    num_steps = 0
    confidences = []
    gradients = []

    while (not misclassified) and (num_steps < step_threshold):
        result = attack(model, criterion, example, target = fake_label, steps = 1, epsilon=epsilon, device = device)
        example, prediction = result[0], result[1]
        new_gradients = result[2]
        gradients.extend(new_gradients)
        probabilities = torch.exp(model(example)).view(10,)
        confidence = probabilities[fake_label].item()

        if confidence >= target_confidence:
            misclassified = True 
        else:
            num_steps += 1
        confidences.append(confidence)

    probabilities = torch.exp(model(example)).view(10,)
    adversarial_distribution = Categorical(probs = probabilities)
    JS = 0.5 *(kl_divergence(true_distribution, adversarial_distribution) + kl_divergence(adversarial_distribution, true_distribution))
    JS = torch.sqrt(JS)
    effort = num_steps / (JS + 0.0001)
    #print 'fake label: ', fake_label
    return [num_steps, torch.log(effort).item(), true_example, example, confidences, gradients]
# ------------------------------------------------------------------------------------------------------------------
def test_model_effort(model, attack, training_set, target_confidence, selection_confidence, loss_function, steps = 100, epsilon=0.1,mnist=True, device='cpu'):
    """
    This function helps us select an example to calculate the effort for and then calls the function to calculate the effort
    :param model: which model to test 
    :param attack: which attack to use 
    :param training_set: which set to draw examples from 
    :param target_confidence: what confidence we should get our model to classify the adversarial example with 
    :param selection_confidence: threshold for original confidence 
    :param loss_function: which loss function we should use 
    :param steps: how many steps of the attack we should limit ourselves to 
    :param epsilon: epsilon we should use when perturbing the example 
    :param mnist: whehther we are using CIFAR or MNIST 
    :param device: whether we are running on CPU or GPU
    """
    data_size = (1,1,28,28)
    if not mnist:
     data_size = (1,3,32,32)
    selected_example = False 
    effort_val = float('-inf')
    model.eval()
    while not selected_example and (effort_val == float('-inf')):
        idx = np.random.choice(range(len(training_set)), 1)[0]

        example = training_set[idx][0].view(*data_size).to(device)
        probabilities = torch.exp(model(example))
        if torch.max(probabilities) >= selection_confidence:
            probabilities = probabilities.view(10,)
            selected_example = True
            effort = calculate_effort(model,example, probabilities, loss_function, target_confidence, attack, steps, epsilon, device=device)
            effort_val = effort[1]
    return effort
