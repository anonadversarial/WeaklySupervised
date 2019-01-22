import numpy as np
import torch
import torchvision.datasets as datasets
from torchvision import transforms

class MNISTRandomLabels(datasets.MNIST):
  """
  Class which generates MNIST with random labels
  """
  def __init__(self, num_classes=10, corrupt_prob=0.9, **kwargs):
    """

  :param num_classes: How many classes our dataset has (default is 10)
  :param corrupt_prob: The probability that a label is corrupted
  Code adapted from https://github.com/pluskid/fitting-random-labels/blob/master/cifar10_data.py
    """
    super(MNISTRandomLabels, self).__init__(**kwargs)

    self.n_classes = num_classes
    self.corrupt_prob = corrupt_prob
    self.corrupt_labels()
    

  def corrupt_labels(self):
    if self.train:
      num_examples = 300
      self.train_data = self.train_data[:num_examples]
      self.train_labels = self.train_labels[:num_examples]
    
    labels = np.array(self.train_labels)
    label_length = len(self.train_labels) if self.train else len(self.test_labels)
    mask = np.random.rand(len(labels)) <= self.corrupt_prob
    new_labels = []
    for idx, elem in enumerate(labels):
      if mask[idx]:
        new_labels.append(np.random.choice(self.n_classes,1)[0])
      else:
        new_labels.append(int(labels[idx]))
    if self.train:
      self.train_labels = labels
    else:
      self.test_labels = labels