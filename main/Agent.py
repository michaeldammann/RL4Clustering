# Adapted from https://github.com/openai/spinningup/blob/master/spinup/examples/pytorch/pg_math/1_simple_pg.py

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import torchvision
import torchvision.datasets as datasets
import config
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from MNIST_CNN import MINST_CNN

import numpy as np
import gym
from gym.spaces import Discrete, Box
from sklearn.metrics import silhouette_samples

class Agent:
    def __init__(self):
        if config.DATASET == "MNIST":
            mnist_trainset = datasets.MNIST(root='./data/mnist', train=True, download=True, transform = ToTensor())
            mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=ToTensor())
            self.dataset = torch.utils.data.ConcatDataset([mnist_trainset, mnist_testset])
            self.neural_net = MINST_CNN()

    # make function to compute action distribution
    def get_policy(self, obs):
        batch_logits = self.neural_net(obs)
        batch_policies = Categorical(logits=batch_logits)
        return batch_policies

    # make action selection function (outputs int actions, sampled from policy)
    def get_action(self,obs):
        return self.get_policy(obs).sample()

    def compute_loss(self, batch_obs, batch_actions, batch_weights):
        logp = self.get_policy(batch_obs).log_prob(batch_actions)
        return -(logp * batch_weights).mean()



    def train(self):
        dataloader = DataLoader(dataset=self.dataset, batch_size=config.BATCH_SIZE, shuffle=True)
        dataloader_iter = iter(dataloader)

        for i in range(len(dataloader)):
            obs, _ = next(dataloader_iter)

            batch_policies=self.get_policy(obs)
            #batch_action are the sampled cluster assignments
            batch_actions = self.get_action()
            #todo: batch actions to numpy

            # Get the representations from the second to last layer (dimension = config.FEATURE_DIM)
            representations = nn.Sequential(*list(self.neural_net.children())[:-1])
            #todo: representations to numpy
            rewards = silhouette_samples(representations, batch_actions)

            optimizer = Adam(self.neural_net.parameters(), lr=config.LR)
            optimizer.zero_grad()
            batch_loss = self.compute_loss(obs, batch_actions, rewards)
            batch_loss.backward()
            optimizer.step()

            break

        #get outputs
        #get/sample actions
        #get rewards
        #gradient step


agent = Agent()
agent.train()



