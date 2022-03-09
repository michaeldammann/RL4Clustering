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
from nn_utilities import mnist_cnn
from pathlib import Path
import json

import numpy as np
import gym
from gym.spaces import Discrete, Box
from sklearn.metrics import silhouette_samples, silhouette_score


class Agent:
    def __init__(self):
        if config.DATASET == "MNIST":
            mnist_trainset = datasets.MNIST(root='./data/mnist', train=True, download=True, transform=ToTensor())
            mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=ToTensor())
            self.dataset = torch.utils.data.ConcatDataset([mnist_trainset, mnist_testset])
            self.neural_net = mnist_cnn()
            print(self.neural_net)

    # make function to compute action distribution
    def get_policy(self, obs):
        batch_logits = self.neural_net(obs)
        batch_policies = Categorical(logits=batch_logits)
        return batch_policies

    # make action selection function (outputs int actions, sampled from policy)
    def get_action(self, obs):
        return self.get_policy(obs).sample()

    def compute_loss(self, batch_obs, batch_actions, batch_weights):
        logp = self.get_policy(batch_obs).log_prob(batch_actions)
        return -(logp * batch_weights).mean()

    def rewards_maxclasses(self, representations_np, batch_actions_np, n_classes, r_classes):
        rewards_np = silhouette_samples(representations_np, batch_actions_np)
        n_clusters = len(np.unique(batch_actions_np))
        print(n_clusters)
        return rewards_np - r_classes * (n_classes - n_clusters)

    def train(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(device)
        self.neural_net.to(device)

        Path('..', config.SAVE_DIR).mkdir(parents=True, exist_ok=True)
        score_history = []
        for epoch in range(config.EPOCHS):
            dataloader = DataLoader(dataset=self.dataset, batch_size=config.BATCH_SIZE, shuffle=True)
            dataloader_iter = iter(dataloader)
            for i in range(len(dataloader) - 1):  # -1 to only use "full" batches
                obs, _ = next(dataloader_iter)
                obs = obs.to(device)

                # batch_action are the sampled cluster assignments
                batch_actions = self.get_action(obs)
                batch_actions_np = batch_actions.cpu().detach().numpy()

                # Get the representations from the second to last layer (dimension = config.FEATURE_DIM)
                representations_model = nn.Sequential(*list(self.neural_net.children())[:-1])

                representations = representations_model(obs)
                representations_np = representations.cpu().detach().numpy()

                # rewards_np = silhouette_samples(representations_np, batch_actions_np)
                rewards_np = self.rewards_maxclasses(representations_np, batch_actions_np, n_classes=10,
                                                     r_classes=config.REWARD_CLASSES)
                rewards = torch.from_numpy(rewards_np).to(device)

                score = silhouette_score(representations_np, batch_actions_np)
                print(score)
                score_history.append(str(score))

                optimizer = Adam(self.neural_net.parameters(), lr=config.LR)
                optimizer.zero_grad()
                batch_loss = self.compute_loss(obs, batch_actions, rewards)
                batch_loss.backward()
                optimizer.step()

            if epoch % config.SAVE_INTERVALL == 0:
                torch.save(self.neural_net.state_dict(),
                           Path('..', config.SAVE_DIR, 'model_epoch_{}'.format(str(epoch))))

        with open(Path('..', config.SAVE_DIR, 'score_history.json'), 'w') as f:
            json.dump(score_history, f)


agent = Agent()
agent.train()
