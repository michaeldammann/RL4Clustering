from pathlib import Path
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import umap
import torch.nn as nn
from nn_utilities import mnist_cnn
import torch
import torchvision.datasets as datasets
import config
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import numpy as np

def score_over_epochs(modelfolder):
    score_history_path = Path('..', modelfolder, 'data.json')
    with open(score_history_path) as d:
        score_history = json.load(d)

    score_history = [float(score) for score in score_history]
    epochs_history = [i for i in range(len(score_history))]

    d = {'epochs_history':epochs_history, 'score_history': score_history}
    pdnumsqr = pd.DataFrame(d)

    sns.set(style='darkgrid')
    sns.lineplot(x='epochs_history', y='score_history', data=pdnumsqr)
    plt.show()

def init_dataloader_mnist():
    mnist_trainset = datasets.MNIST(root='./data/mnist', train=True, download=True, transform=ToTensor())
    mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=ToTensor())
    dataset = torch.utils.data.ConcatDataset([mnist_trainset, mnist_testset])
    dataloader = DataLoader(dataset=dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    return dataloader

def load_representation_model(modelfolder, epoch):
    model_path = Path('..', modelfolder, 'model_epoch_{}'.format(str(epoch)))
    model = mnist_cnn()
    model.load_state_dict(torch.load(model_path))
    representations_model = nn.Sequential(*list(model.children())[:-1])

    return representations_model

def prepare_representations(modelfolder, epoch, dataloader, percentage_representations=0.1):
    all_representations = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataloader_iter = iter(dataloader)

    rep_model = load_representation_model(modelfolder, epoch)
    rep_model.to(device)

    sample_interval = int(100/percentage_representations)

    for i in range(len(dataloader)):
        obs, classes = next(dataloader_iter)
        obs = obs.to(device)

        representations = rep_model(obs)
        representations_np = representations.cpu().detach().numpy()

        for idx, rep in enumerate(representations_np):
            if idx%sample_interval==0:
                all_representations.append(rep)

    return np.array(all_representations)

def umap_representations(modelfolder, epoch, dataloader, percentage_representations=0.1):
    #prepare data
    rep_data = prepare_representations(modelfolder, epoch, dataloader, percentage_representations)
    reducer = umap.UMAP(random_state=42)
    print('Starting UMAP Fitting')
    embedding = reducer.fit_transform(rep_data)

modelfolder = 'result_batch_size_1024_feature_dim_16_reward_classes_1_learningrate_0.001'
mnist_dataloader = init_dataloader_mnist()
umap_representations(modelfolder, 15, mnist_dataloader)


#score over time
#umap viz of representations every n epochs
