import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from tqdm import tqdm

parser = argparse.ArgumentParser(description='3dconv_repr_learn')
parser.add_argument('--data_path', type=str, default='', help='absolute path to root directory with patient data')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate (default: 0.0002)')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 100)')
parser.add_argument('--batch_size', type=int, default=16, help='input batch size for training (default: 16)')
parser.add_argument('--wd', type=float, default=0.0, help='weight decay (default: 0.0)')
parser.add_argument('--lamb', type=float, default=0.5, help='disentanglement factor (default: 0.5)')
parser.add_argument('--tau_dim', type=int, default=256, help='dimension R^d for representation direction vector (default: 256)')
parser.add_argument('--img_dim', type=int, default=256, help='dimension (R^d)x(R^d)x(R^d) for 3D volumes (default: 64)')
parser.add_argument('--seed', type=int, default=0, help='seed')
args = parser.parse_args()

# set cuda device (GPU / CPU)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# set deterministic behavior based on seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# DEBUG FLAG:
# torch.autograd.set_detect_anomaly(True)

class Net(nn.Module):

    def __init__(self, image_dim=64, tau_dim=256):
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
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=2048, out_features=tau_dim)
        )
        # Perform Binary Soft-Classification
        self.linear = nn.Sequential(
            nn.Linear(in_features=tau_dim, out_features=128),
            nn.Linear(in_features=128, out_features=16),
            nn.Linear(in_features=16, out_features=2),
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

    def pair_loss(self, emb_t1, emb_t2, delta_sigma, pred_1, pred_2, labels_1, labels_2, lamb):
        """
        emb_t1 (batch of (tau-dim)-1dtensor): embedding for first image in pair
        emb_t2 (batch of (tau-dim)-1dtensor): embedding for second image in pair
        delta_sigma ((tau-dim)-1dtensor): ground truth difference in value for factor to disentangle
        pred_1, pred_2 (batch of tuple[float, float]): model's probability prediction for each of the labels, i.e. (0.12, 0.96)
        label_1, label_2 (batch of tuple[int, int]): ground truth class membership for the two independent label, i.e. (0, 0), (0, 1), (1, 0), (1, 1)
        lamb: weighting factor between BCE loss and disentangle loss
        tau: current value of disentanglement vector
        """
        # classification component loss
        bce = nn.BCELoss()
        labels_1, labels_2 = torch.tensor(labels_1), torch.tensor(labels_2)
        bce_loss_1 = bce(pred_1, labels_1)
        bce_loss_2 = bce(pred_2, labels_2)
        bce_loss = bce_loss_1 + bce_loss_2
        if lamb == 0:
            return bce_loss
        # efficient dot-product calculation
        proj_e1_len = torch.sum(emb_t1 * self.tau.repeat(emb_t1.shape[0], 1), dim=1) 
        proj_e2_len = torch.sum(emb_t2 * self.tau.repeat(emb_t2.shape[0], 1), dim=1) # dot-product
        # representation component loss through projection penalty
        emb_len_diff = torch.abs(proj_e2_len - proj_e1_len)
        disentangle_loss = torch.sum(torch.abs(emb_len_diff - delta_sigma))   
        return bce_loss + lamb*disentangle_loss

    def update_tau(self):
        """ update the feature representation direction, jointly learned in training """
        tau = self.encode(torch.ones((1, 1, self.dim, self.dim, self.dim), dtype=torch.float64))
        unit_tau = tau / float(torch.norm(tau))
        unit_tau = torch.reshape(tau, (-1,))
        self.tau = unit_tau.to(self.device)

    def count_correct_batch(self, batch_preds, batch_labels):
        corr = 0
        for i in range(len(batch_preds)):
            pred = batch_preds[i]
            lab = batch_labels[i]
            corr += [round(float(pred[0])), round(float(pred[1]))] == [round(float(x)) for x in lab]
        return corr

    def train(self, data_config, alpha, epochs, batch_size, lamb, wd):
        # restore data from config
        train_pairs = data_config['train_pairs']
        test_pairs = data_config['test_pairs']
        train_del_sigmas = data_config['train_del_sigmas']
        test_del_sigmas = data_config['test_del_sigmas']
        train_labels = data_config['train_labels']
        test_labels = data_config['test_labels']
        train_sigmas = data_config['train_sigmas']
        test_sigmas = data_config['test_sigmas']
        # prepare dataloader
        train_dl = DataLoader(train_pairs, batch_size=batch_size, shuffle=False)
        test_dl = DataLoader(test_pairs, batch_size=batch_size, shuffle=False)
        # stats loggers
        train_loss_vals, train_acc, test_loss_vals, test_acc = list(), list(), list(), list()
        # optimizer
        opt = optim.Adam(self.parameters(), lr=alpha, weight_decay=wd)
        imgs_seen = 0
        for e in (range(epochs)):
            total_loss = 0
            correct = 0
            for i, batch in enumerate(tqdm(train_dl, desc=f'Epoch {e+1} -- TRAINING')):
                imgs_seen += 2 * len(batch[0])
                # forward portion (prediction)
                img1 = batch[0].view(batch_size, 1, self.dim, self.dim, self.dim)
                img2 = batch[1].view(batch_size, 1, self.dim, self.dim, self.dim)
                pred1, pred2 = self.forward(img1), self.forward(img2)
                # latent-space representation for disentanglement
                emb1, emb2 = self.encode(img1).to(self.device), self.encode(img2).to(self.device)
                ds = torch.tensor(train_del_sigmas[i*batch_size : (i+1)*batch_size]).to(self.device)
                l1 = torch.tensor([tup[0] for tup in train_labels[i*batch_size : (i+1)*batch_size]])
                l2 = torch.tensor([tup[1] for tup in train_labels[i*batch_size : (i+1)*batch_size]])
                sigma1 = torch.tensor([tup[0] for tup in train_sigmas[i*batch_size : (i+1)*batch_size]])
                sigma2 = torch.tensor([tup[1] for tup in train_sigmas[i*batch_size : (i+1)*batch_size]])
                # accuracy calculation
                correct += self.count_correct_batch(pred1, l1)
                correct += self.count_correct_batch(pred2, l2)
                # gradient optimization
                opt.zero_grad()
                loss = self.pair_loss(emb_t1=emb1, emb_t2=emb2, delta_sigma=ds, pred_1=pred1, pred_2=pred2,
                                      labels_1=l1, labels_2=l2, lamb=lamb)
                loss.retain_grad() # NECESSARY SINCE THE COMPUTATIONAL GRAPH NEEDS INTERMEDIATE GRADIENTS!
                loss.backward(retain_graph=True)
                opt.step() # update
                # representation direction update if needed
                if lamb > 0:
                    self.update_tau() # update tau-direction (jointly learned)
                total_loss = total_loss + float(loss.data)
            train_loss_vals.append(total_loss)
            train_acc.append(correct / imgs_seen)
            total_loss = 0
            correct = 0
            ## TESTING PORTION (mimics main training loop!)
            with torch.no_grad():
                imgs_seen = 0
                for i, batch in enumerate(tqdm(test_dl, desc=f'Epoch {e+1} --  TESTING')):
                    imgs_seen += 2 * len(batch[0])
                    img1 = batch[0].view(batch_size, 1, self.dim, self.dim, self.dim)
                    img2 = batch[1].view(batch_size, 1, self.dim, self.dim, self.dim)
                    pred1, pred2 = self.forward(img1), self.forward(img2)
                    emb1, emb2 = self.encode(img1).to(self.device), self.encode(img2).to(self.device)
                    ds = torch.tensor(test_del_sigmas[i*batch_size : (i+1)*batch_size]).to(self.device)
                    l1 = torch.tensor([tup[0] for tup in test_labels[i*batch_size : (i+1)*batch_size]])
                    l2 = torch.tensor([tup[1] for tup in test_labels[i*batch_size : (i+1)*batch_size]])
                    sigma1 = torch.tensor([tup[0] for tup in test_sigmas[i*batch_size : (i+1)*batch_size]])
                    sigma2 = torch.tensor([tup[1] for tup in test_sigmas[i*batch_size : (i+1)*batch_size]])
                    correct += self.count_correct_batch(pred1, l1)
                    correct += self.count_correct_batch(pred2, l2)
                    loss = self.pair_loss(emb_t1=emb1, emb_t2=emb2, delta_sigma=ds, pred_1=pred1, pred_2=pred2,
                                          labels_1=l1, labels_2=l2, lamb=lamb)
                    total_loss += float(loss.data)
                test_loss_vals.append(total_loss)
                test_acc.append(correct / imgs_seen)
            torch.save(self.state_dict(), os.path.join('/', 'curr_model.pth'))
            print(f'    Epoch {e+1} -> train loss = {round(train_loss_vals[-1], 6)} | test loss = {round(test_loss_vals[-1], 6)} | train acc = {round(train_acc[-1], 4)} | test acc = {round(test_acc[-1], 4)}\n')
        # Train/Test Loss Plot 
        plt.plot(range(1, len(train_loss_vals)+1), train_loss_vals, label='combined train loss')
        plt.plot(range(1, len(test_loss_vals)+1), test_loss_vals, label='combined test loss')
        plt.legend(loc='upper right')
        plt.xlabel('epochs')
        plt.ylabel('avg {cross-entropy + disentanglement} loss')
        plt.title(f'Model Train/Test Loss - lr={alpha}, epochs={epochs}, lambda={lamb}')
        plt.savefig(f'loss__lr{alpha}_lamb{lamb}_e{epochs}_bs{batch_size}_wd{wd}.png')
        # Train/Test Accuracy Plot
        plt.plot(range(1, len(train_acc)+1), train_acc, label='train accuracy')
        plt.plot(range(1, len(test_acc)+1), test_acc, label='test accuracy')
        plt.legend(loc='lower right')
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.title(f'Model Binary Classif Acc - lr={alpha}, epochs={epochs}, lambda={lamb}')
        plt.savefig(f'acc___lr{alpha}_lamb{lamb}_e{epochs}_bs{batch_size}_wd{wd}.png')


    def load_data(self, path_to_pickles='cns_data', num_folds=5, seed=2022):
        from sklearn.utils import shuffle
        # Read in the data
        train_fold = list()
        test_fold = list()
        label_train_fold = list()
        label_test_fold = list()
        sigma_train_fold = list()
        sigma_test_fold = list()
        for fold in range(num_folds):
            with open(f'{path_to_pickles}/original_train_data_{fold}.pickle', 'rb') as handle:
                train_fold.append(pickle.load(handle))
            with open(f'{path_to_pickles}/original_test_data_{fold}.pickle', 'rb') as handle:
                test_fold.append(pickle.load(handle))
            with open(f'{path_to_pickles}/original_train_label_{fold}.pickle', 'rb') as handle:
                label_train_fold.append(pickle.load(handle))
            with open(f'{path_to_pickles}/original_test_label_{fold}.pickle', 'rb') as handle:
                label_test_fold.append(pickle.load(handle))
            with open(f'{path_to_pickles}/train_age_{fold}.pickle', 'rb') as handle:
                sigma_train_fold.append(pickle.load(handle))
            with open(f'{path_to_pickles}/test_age_{fold}.pickle', 'rb') as handle:
                sigma_test_fold.append(pickle.load(handle))
        train_data = np.concatenate(tuple(train_fold))
        test_data = np.concatenate(tuple(test_fold))
        label_train = np.concatenate(tuple(label_train_fold))
        label_test = np.concatenate(tuple(label_test_fold))
        sigma_train = np.concatenate(tuple(sigma_train_fold))
        sigma_test = np.concatenate(tuple(sigma_test_fold))
        # Apply deterministic shuffle in the list (meaning all lists are shuffled in a same random order)
        train_data = shuffle(train_data, random_state=seed)        
        test_data = shuffle(test_data, random_state=seed)
        label_train = shuffle(label_train, random_state=seed)
        label_test = shuffle(label_test, random_state=seed)
        sigma_train = shuffle(sigma_train, random_state=seed)
        sigma_test = shuffle(sigma_test, random_state=seed)
        # Make into pairs for pairwise disentanglement representation training
        train_size = train_data.shape[0] // 2
        test_size = test_data.shape[0] // 2
        train_pairs = [(train_data[2*i, 0, :, :, :], train_data[2*i + 1, 0, :, :, :]) for i in range(train_size)]
        test_pairs = [(test_data[2*i, 0, :, :, :], test_data[2*i + 1, 0, :, :, :]) for i in range(test_size)]
        train_labels = [(label_train[2*i, :], label_train[2*i + 1, :]) for i in range(train_size)]
        test_labels = [(label_test[2*i, :], label_train[2*i + 1, :]) for i in range(test_size)]
        train_sigma = [(sigma_train[2*i + 1], sigma_train[2*i]) for i in range(train_size)]
        test_sigma = [(sigma_test[2*i + 1], sigma_test[2*i]) for i in range(test_size)]
        train_del_sigma = [sigma_train[2*i + 1] - sigma_train[2*i] for i in range(train_size)]
        test_del_sigma = [sigma_test[2*i + 1] - sigma_test[2*i] for i in range(test_size)]
            print(f'\tTraining Label Counts: {[item for item in list(sorted(tr_counts.items()))]}')
            print(f'\t Testing Label Counts: {[item for item in list(sorted(te_counts.items()))]}')
        data_config = {'train_pairs':train_pairs, 'test_pairs':test_pairs, 'train_labels':train_labels, 'test_labels':test_labels,
            'train_del_sigmas':train_del_sigma, 'test_del_sigmas':test_del_sigma, 'train_sigmas':train_sigma, 'test_sigmas':test_sigma}
        return data_config
        

if __name__ == '__main__':
    """
    Typping:
    train_pairs, valid_pairs: (list[tuple(3d-tensor, 3d-tensor)])
    train_del_sigmas, valid_del_sigmas: (list[float]), i.e. [3.4, 5.3, -2.1, 0.3]
    train_labels, valid_labels: (list[tuple([float, float], [float, float])]), i.e. [([1., 0.], [1., 1.]), ([0., 1.], [0., 0.])]
    alpha: float
    epochs: int
    lamb: float
    bs (batch_size): int
    """
    print(f'Starting Run: epochs = {args.epochs}; alpha = {args.alpha}; batch_size = {args.batch_size}; lambda = {args.lamb}; wd = {args.wd}')
    model = Net(image_dim=args.img_dim, tau_dim=args.tau_dim).double()
    print(f'Model Initialized: img_dim = {model.dim}x{model.dim}x{model.dim}; tau_dim = {model.tau.shape[0]}')
    data_config = model.load_data(path_to_pickles=args.data_path)
    print(f'Data Loaded')
    print(f'Starting Training:')
    model.train(data_config=data_config, alpha=args.alpha, epochs=args.epochs,
        batch_size=args.batch_size, lamb=args.lamb, wd=args.wd)
