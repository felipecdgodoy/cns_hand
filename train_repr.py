import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
import pickle
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from tqdm import tqdm

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='ADNI')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 100)')
parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 16)')
parser.add_argument('--lamb', type=float, default=0.5, help='disentanglement factor (default: 0.5)')
parser.add_argument('--wd', type=float, default=0.0, help='weight decay (default: 0.0)')
parser.add_argument('--seed', type=int, default=0, help='seed')
args = parser.parse_args()

# set cuda device (GPU / CPU)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# set deterministic behavior based on seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# debug
# torch.autograd.set_detect_anomaly(True)

class Net(nn.Module):

    def __init__(self, image_dim=64, tau_dim=512):
        """ image_dim (int): value 'D' such that all images are a 3D-volume DxDxD"""
        """ tau_dim (int): number of dimensions to encode the tau-direction in"""
        super(Net, self).__init__()
        self.device = device
        self.dim = image_dim
        self.tau = (1 / tau_dim * torch.ones(tau_dim).double()).to(self.device)
        # Encoder from 3D-volume input to latent-space vector
        self.encode = nn.Sequential(
            nn.Sequential(nn.Conv3d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
                          nn.ReLU(),
                          nn.MaxPool3d(kernel_size=2)),
            nn.Sequential(nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
                          nn.ReLU(),
                          nn.MaxPool3d(kernel_size=2)),
            nn.Sequential(nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
                          nn.ReLU(),
                          nn.MaxPool3d(kernel_size=2)),
            nn.Sequential(nn.Conv3d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
                          nn.ReLU(),
                          nn.MaxPool3d(kernel_size=2)),
            nn.Flatten(start_dim=1)
        )
        # Perform Binary Soft-Classification
        self.linear = nn.Sequential(
            nn.Linear(in_features=tau_dim, out_features=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        ## Re-format
        x = x.view((-1, 1, self.dim, self.dim, self.dim))
        ## Encode to latent-space
        x = self.encode(x)
        ## Classif
        x = self.linear(x)
        return x

    def pair_loss(self, emb_t1, emb_t2, delta_sigma, pred_1, pred_2, labels_1, labels_2, lamb, tau):
        """
        emb_t1 (batch of (tau-dim)-1dtensor): embedding for first image in pair
        emb_t2 (batch of (tau-dim)-1dtensor): embedding for second image in pair
        delta_sigma ((tau-dim)-1dtensor): ground truth difference in value for factor to disentangle
        pred_1, pred_2 (batch of tuple[float, float]): model's probability prediction for each of the labels, i.e. (0.12, 0.96)
        label_1, label_2 (batch of tuple[int, int]): ground truth class membership for the two independent label, i.e. (0, 0), (0, 1), (1, 0), (1, 1)
        lamb: weighting factor between BCE loss and disentangle loss
        tau: current value of disentanglement vector
        """
        bce = nn.BCELoss()
        labels_1, labels_2 = torch.tensor(labels_1), torch.tensor(labels_2)
        try:
            bce_loss_1 = bce(pred_1, labels_1)
        except:
            bce_loss_1 = bce(0.5 * torch.ones(2), 0.5 * torch.ones(2))
        try:
            bce_loss_2 = bce(pred_2, labels_2)
        except:
            bce_loss_2 = bce(0.5 * torch.ones(2), 0.5 * torch.ones(2))
        bce_loss = bce_loss_1 + bce_loss_2
        if lamb == 0:
            return bce_loss
        proj_e1_len = torch.tensor([torch.norm((torch.dot(e, tau) / torch.dot(tau, tau)) * tau) for e in emb_t1])
        proj_e2_len = torch.tensor([torch.norm((torch.dot(e, tau) / torch.dot(tau, tau)) * tau) for e in emb_t2])
        emb_len_diff = torch.abs(proj_e2_len - proj_e1_len)
        proj_e1_len = torch.sum(emb_t1 * tau.repeat(emb_t1.shape[0], 1), dim=1) # dot-product
        proj_e2_len = torch.sum(emb_t2 * tau.repeat(emb_t2.shape[0], 1), dim=1) # dot-product
        emb_len_diff = torch.abs(proj_e2_len - proj_e1_len)
        disentangle_loss = torch.sum(torch.abs(emb_len_diff - delta_sigma)) 
        reg = nn.MSELoss()         
        disentangle_loss = reg(emb_len_diff, delta_sigma) 
        return bce_loss + lamb*disentangle_loss

    def bce_loss(self, emb_t1, emb_t2, delta_sigma, pred_1, pred_2, labels_1, labels_2, lamb, tau):
        """
        emb_t1 (batch of (tau-dim)-1dtensor): embedding for first image in pair
        emb_t2 (batch of (tau-dim)-1dtensor): embedding for second image in pair
        delta_sigma ((tau-dim)-1dtensor): ground truth difference in value for factor to disentangle
        pred_1, pred_2 (batch of tuple[float, float]): model's probability prediction for each of the labels, i.e. (0.12, 0.96)
        label_1, label_2 (batch of tuple[int, int]): ground truth class membership for the two independent label, i.e. (0, 0), (0, 1), (1, 0), (1, 1)
        lamb: weighting factor between BCE loss and disentangle loss
        tau: current value of disentanglement vector
        """
        bce = nn.BCELoss()
        labels_1, labels_2 = torch.tensor(labels_1), torch.tensor(labels_2)
        bce_loss_1 = bce(pred_1, labels_1)
        bce_loss_2 = bce(pred_2, labels_2)
        bce_loss = bce_loss_1 + bce_loss_2
        return bce_loss        

    def projection_loss(self, emb_t1, emb_t2, delta_sigma, pred_1, pred_2, labels_1, labels_2, lamb, tau):
        """
        See pair_loss docs
        """
        proj_e1_len = torch.tensor([torch.norm((torch.dot(e, tau) / torch.dot(tau, tau)) * tau) for e in emb_t1])
        proj_e2_len = torch.tensor([torch.norm((torch.dot(e, tau) / torch.dot(tau, tau)) * tau) for e in emb_t2])
        emb_len_diff = torch.abs(proj_e2_len - proj_e1_len)
        proj_e1_len = torch.sum(emb_t1 * tau.repeat(emb_t1.shape[0], 1), dim=1) # dot-product
        proj_e2_len = torch.sum(emb_t2 * tau.repeat(emb_t2.shape[0], 1), dim=1) # dot-product
        emb_len_diff = torch.abs(proj_e2_len - proj_e1_len)
        disentangle_loss = torch.sum(torch.abs(emb_len_diff - delta_sigma)) 
        reg = nn.MSELoss()         
        disentangle_loss = reg(emb_len_diff, delta_sigma)
        return disentangle_loss

    def update_tau(self):
        tau = self.encode(torch.ones((1, 1, self.dim, self.dim, self.dim), dtype=torch.float64))
        unit_tau = tau / float(torch.norm(tau))
        unit_tau = torch.reshape(unit_tau, (-1,))
        return unit_tau.to(self.device)

    def count_correct_batch(self, batch_preds, batch_labels):
        corr = 0
        for i in range(len(batch_preds)):
            try:
                corr += np.any(np.round(batch_preds[i].detach().numpy()) == batch_labels[i].detach().numpy())
            except:
                pass
        return corr

    def train(self, train_pairs, test_pairs, train_del_sigmas, test_del_sigmas, train_labels, test_labels,
            train_sigmas, test_sigmas, alpha, epochs, batch_size, lamb, wd):
        train_labels = np.array(train_labels)
        test_labels = np.array(test_labels)
        train_loss_vals, train_acc, test_loss_vals, test_acc = [0], [0], [0], [0]
        train_pred_error, test_pred_error = [0], [0]
        opt = optim.Adam(self.parameters(), lr=alpha, weight_decay=wd)
        train_dl = DataLoader(train_pairs, batch_size=batch_size, shuffle=False)
        test_dl = DataLoader(test_pairs, batch_size=batch_size, shuffle=False)
        imgs_seen = 0
        pred1, pred2, l1, l2 = None, None, None, None
        for e in (range(epochs)):
            total_loss = 0
            total_pred_error = 0
            correct = 0
            for i, batch in enumerate(tqdm(train_dl, desc=f'Epoch {e+1} -- TRAINING')):
                if i >= 2:
                    break
                if len(batch[0]) < batch_size:
                    break
                imgs_seen += 2 * len(batch[0])
                img1 = batch[0].view(batch_size, 1, self.dim, self.dim, self.dim)
                img2 = batch[1].view(batch_size, 1, self.dim, self.dim, self.dim)
                pred1, pred2 = self.forward(img1), self.forward(img2)
                emb1, emb2 = self.encode(img1).to(self.device), self.encode(img2).to(self.device)
                ds = torch.tensor(train_del_sigmas[i*batch_size : (i+1)*batch_size]).to(self.device)
                l1 = torch.tensor([tup[0] for tup in train_labels[i*batch_size : (i+1)*batch_size]])
                l2 = torch.tensor([tup[1] for tup in train_labels[i*batch_size : (i+1)*batch_size]])
                total_pred_error += float(torch.sum(torch.abs(l1 - pred1)).data)
                total_pred_error += float(torch.sum(torch.abs(l2 - pred2)).data)                
                correct += self.count_correct_batch(pred1, l1)
                correct += self.count_correct_batch(pred2, l2)
                opt.zero_grad()
                loss1 = self.pair_loss(emb_t1=emb1, emb_t2=emb2, delta_sigma=ds, pred_1=pred1, pred_2=pred2,
                                      labels_1=l1, labels_2=l2, lamb=lamb, tau=self.tau)
                loss1.retain_grad()
                loss1.backward(retain_graph=True)
                opt.step()
                if lamb > 0:
                    self.tau = self.update_tau() # update tau-direction (jointly learned)
                total_loss = total_loss + float(loss1.data)
            p = nn.utils.parameters_to_vector(self.parameters())
            # print(f'Grad: {list(self.parameters())[0].grad}')
            # print(f'Mean: {torch.mean(p)} | Var = {torch.var(p)}')
            train_loss_vals.append(total_loss / imgs_seen)
            train_acc.append(correct / imgs_seen)
            train_pred_error.append(total_pred_error / imgs_seen)
            total_loss = 0
            correct = 0
            total_pred_error = 0
            ## TESTING PORTION
            with torch.no_grad():
                imgs_seen = 0
                for i, batch in enumerate(tqdm(test_dl, desc=f'Epoch {e+1} --  TESTING')):
                    if i >= 2:
                        break
                    if len(batch[0]) < batch_size:
                        break
                    imgs_seen += 2 * len(batch[0])
                    img1 = batch[0].view(batch_size, 1, self.dim, self.dim, self.dim)
                    img2 = batch[1].view(batch_size, 1, self.dim, self.dim, self.dim)
                    pred1, pred2 = self.forward(img1), self.forward(img2)
                    emb1, emb2 = self.encode(img1).to(self.device), self.encode(img2).to(self.device)
                    ds = torch.tensor(test_del_sigmas[i*batch_size : (i+1)*batch_size]).to(self.device)
                    l1 = torch.tensor([tup[0] for tup in test_labels[i*batch_size : (i+1)*batch_size]])
                    l2 = torch.tensor([tup[1] for tup in test_labels[i*batch_size : (i+1)*batch_size]])
                    total_pred_error += float(torch.sum(torch.abs(l1 - pred1)).data)
                    total_pred_error += float(torch.sum(torch.abs(l2 - pred2)).data)
                    correct += self.count_correct_batch(pred1, l1)
                    correct += self.count_correct_batch(pred2, l2)
                    loss = self.pair_loss(emb_t1=emb1, emb_t2=emb2, delta_sigma=ds, pred_1=pred1, pred_2=pred2,
                                          labels_1=l1, labels_2=l2, lamb=lamb, tau=self.tau)
                    total_loss += float(loss.data)
                test_loss_vals.append(total_loss / imgs_seen)
                test_acc.append(correct / imgs_seen)
                test_pred_error.append(total_pred_error / imgs_seen)                
            # torch.save(self.state_dict(), os.path.join('/scratch/users/fgodoy/saved_model', 'full_curr_model.pth'))
            print(f'    Epoch {e+1} -> avg train loss = {round(train_loss_vals[-1], 4)} | avg test loss = {round(test_loss_vals[-1], 4)} | train acc = {round(train_acc[-1], 4)} | test acc = {round(test_acc[-1], 4)}\n')


    def load_data(self, path_to_pickles='', sigma_feat='age', num_folds=5, seed=2023, show_label_counts=False):
        from sklearn.utils import shuffle
        # Read in the data
        train_fold = list()
        test_fold = list()
        label_train_fold = list()
        label_test_fold = list()
        sigma_train_fold = list()
        sigma_test_fold = list()
        for fold in range(num_folds):
            with open(f'{path_to_pickles}original_train_data_{fold}.pickle', 'rb') as handle:
                train_fold.append(pickle.load(handle))
            with open(f'{path_to_pickles}original_test_data_{fold}.pickle', 'rb') as handle:
                test_fold.append(pickle.load(handle))
            with open(f'{path_to_pickles}original_train_label_{fold}.pickle', 'rb') as handle:
                label_train_fold.append(pickle.load(handle))
            with open(f'{path_to_pickles}original_test_label_{fold}.pickle', 'rb') as handle:
                label_test_fold.append(pickle.load(handle))
            with open(f'{path_to_pickles}train_{sigma_feat}_{fold}.pickle', 'rb') as handle:
                sigma_train_fold.append(pickle.load(handle))
            with open(f'{path_to_pickles}test_{sigma_feat}_{fold}.pickle', 'rb') as handle:
                sigma_test_fold.append(pickle.load(handle))
        train_data = np.concatenate(tuple(train_fold))
        test_data = np.concatenate(tuple(test_fold))
        label_train = np.concatenate(tuple(label_train_fold))
        label_test = np.concatenate(tuple(label_test_fold))
        sigma_train = np.concatenate(tuple(sigma_train_fold))
        sigma_test = np.concatenate(tuple(sigma_test_fold))
        # print(len(train_data), len(label_train), len(sigma_train))
        # print(len(test_data), len(label_test), len(sigma_test))
        # Apply deterministic shuffle in the list (meaning all lists are shuffled in a same random order)
        train_data = shuffle(train_data, random_state=seed)        
        test_data = shuffle(test_data, random_state=seed)
        label_train = shuffle(label_train, random_state=seed)
        label_test = shuffle(label_test, random_state=seed)
        sigma_train = shuffle(sigma_train, random_state=seed)
        sigma_test = shuffle(sigma_test, random_state=seed)
        # Make into pairs
        train_size = min(train_data.shape[0] // 2, label_train.shape[0] // 2, sigma_train.shape[0] // 2)
        test_size = min(test_data.shape[0] // 2, label_test.shape[0] // 2, sigma_test.shape[0] // 2)
        train_pairs = [(train_data[2*i, 0, :, :, :], train_data[2*i + 1, 0, :, :, :]) for i in range(train_size)]
        test_pairs = [(test_data[2*i, 0, :, :, :], test_data[2*i + 1, 0, :, :, :]) for i in range(test_size)]
        train_labels = [(label_train[2*i, :], label_train[2*i + 1, :]) for i in range(train_size)]
        test_labels = [(label_test[2*i, :], label_train[2*i + 1, :]) for i in range(test_size)]
        train_sigma = [(sigma_train[2*i], sigma_train[2*i] + 1) for i in range(train_size)]
        test_sigma = [(sigma_test[2*i], sigma_test[2*i] + 1) for i in range(test_size)]
        train_del_sigma = [sigma_train[2*i] - sigma_train[2*i + 1] for i in range(train_size)]
        test_del_sigma = [sigma_test[2*i] - sigma_test[2*i + 1] for i in range(test_size)]
        if show_label_counts:
            tr_counts = dict()
            for l in train_labels:
                l1 = str(list(l[0])).strip()
                l2 = str(list(l[1])).strip()
                if l1 not in tr_counts.keys():
                    tr_counts[l1] = 0
                if l2 not in tr_counts.keys():
                    tr_counts[l2] = 0
                tr_counts[l1] += 1
                tr_counts[l2] += 1
            te_counts = dict()
            for l in test_labels:
                l1 = str(list(l[0])).strip()
                l2 = str(list(l[1])).strip()
                if l1 not in te_counts.keys():
                    te_counts[l1] = 0
                if l2 not in te_counts.keys():
                    te_counts[l2] = 0
                te_counts[l1] += 1
                te_counts[l2] += 1
            print(f'\tTraining Label Counts: {[item for item in list(sorted(tr_counts.items()))]}')
            print(f'\t Testing Label Counts: {[item for item in list(sorted(te_counts.items()))]}')
        return train_pairs, test_pairs, train_labels, test_labels, train_del_sigma, test_del_sigma, train_sigma, test_sigma
        

if __name__ == '__main__':
    """
    Typping:
    train_pairs, valid_pairs: (list[tuple(3d-tensor, 3d-tensor)])
    train_del_sigmas, valid_del_sigmas: (list[float]), i.e. [3.4, 5.3, -2.1, 0.3]
    train_labels, valid_labels: (list[tuple([float, float], [float, float])]), i.e. [([1., 0.], [1., 1.]), ([0., 1.], [0., 0.])]
    alpha: float
    epochs: int
    lamb: float
    """
    alpha = args.lr
    epochs = args.epochs
    lamb = args.lamb
    batch_size = args.batch_size
    wd = args.wd
    print(f'Starting Run: epochs = {epochs}; alpha = {alpha}; batch_size = {batch_size}; lambda = {lamb}')
    model = Net(image_dim=64, tau_dim=2048).double()
    print(f'Model Initialized: image_dim = {model.dim}x{model.dim}x{model.dim}; tau_dim = {model.tau.shape[0]}')
    train_pairs, test_pairs, train_labels, test_labels, train_del_sigmas, test_del_sigmas, train_sigmas, test_sigmas = model.load_data()
    print(f'Data Loaded')
    print(f'Starting Training:')
    model.train(train_pairs, test_pairs, train_del_sigmas, test_del_sigmas, train_labels, test_labels, train_sigmas, test_sigmas,
        alpha, epochs, batch_size, lamb, wd)
