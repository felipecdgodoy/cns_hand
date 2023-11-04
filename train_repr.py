import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
import pickle as pk
import nibabel as nib

from torch.utils.data import DataLoader

from tqdm import tqdm

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='ADNI')
parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 100)')
parser.add_argument('--batch_size', type=int, default=16, help='input batch size for training (default: 16)')
parser.add_argument('--lamb', type=float, default=1e-3, help='disentanglement factor (default: 0.5)')
parser.add_argument('--wd', type=float, default=0.0, help='weight decay (default: 0.0)')
parser.add_argument('--seed', type=int, default=2023, help='seed')
parser.add_argument('--num_folds', type=int, default=5, help='number of folds to split data for CV')
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

def load_matched_data(path_to_pickles='', sigma_feat='npz', seed=2023):
        sigma_feat = sigma_feat.lower()
        assert sigma_feat in ['npz', 'age', 'gender'], f'Unsurported feature. must be age, npz, or gender'
        from sklearn.utils import shuffle
        # Read in the data
        with open(f'{path_to_pickles}matched_patients.pickle', 'rb') as handle:
            ucsf_df = pk.load(handle)
        # Apply deterministic shuffle in the list (meaning all lists are shuffled in a same random order)
        ids = shuffle(np.array(ucsf_df['id']), random_state=seed)
        images = shuffle(np.stack(ucsf_df['image']), random_state=seed)
        labels = shuffle(np.array(ucsf_df['label']), random_state=seed)
        sigmas = shuffle(np.array(ucsf_df[sigma_feat]), random_state=seed)
        # Make into pairs
        num_pairs = len(labels) // 2
        image_pairs = np.array([(images[2*i], images[2*i + 1]) for i in range(num_pairs)])
        label_pairs = np.array([(labels[2*i], labels[2*i + 1]) for i in range(num_pairs)])
        sigma_pairs = np.array([(sigmas[2*i], sigmas[2*i + 1]) for i in range(num_pairs)])
        delta_sigma_pairs = np.array([sigmas[2*i] - sigmas[2*i + 1] for i in range(num_pairs)])
        id_pairs = np.array([(ids[2*i], ids[2*i + 1]) for i in range(num_pairs)])
        return image_pairs, label_pairs, sigma_pairs, delta_sigma_pairs, id_pairs

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
        x = x.view((-1, 1, self.dim, self.dim, self.dim)) ## Re-format
        x = self.encode(x) ## Encode to latent-space
        x = self.linear(x) ## Classification
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
        pred_1 = pred_1.nan_to_num(0.5).double()
        pred_2 = pred_2.nan_to_num(0.5).double()
        bce_loss_1 = bce(pred_1, labels_1.double())
        bce_loss_2 = bce(pred_2, labels_2.double())
        bce_loss = bce_loss_1 + bce_loss_2
        if lamb == 0:
            return bce_loss
        proj_e1_len = torch.tensor([torch.norm((torch.dot(e, tau) / torch.dot(tau, tau)) * tau) for e in emb_t1])
        proj_e2_len = torch.tensor([torch.norm((torch.dot(e, tau) / torch.dot(tau, tau)) * tau) for e in emb_t2])
        emb_len_diff = torch.abs(proj_e2_len - proj_e1_len)
        proj_e1_len = torch.sum(emb_t1 * tau.repeat(emb_t1.shape[0], 1), dim=1) # dot-product
        proj_e2_len = torch.sum(emb_t2 * tau.repeat(emb_t2.shape[0], 1), dim=1) # dot-product
        emb_len_diff = torch.abs(proj_e2_len - proj_e1_len)
        sigma_feature_present = ~ torch.isnan(delta_sigma)
        disentangle_loss = (emb_len_diff - delta_sigma)[sigma_feature_present].square().sum()
        return bce_loss + lamb*disentangle_loss

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

    def train_fold(self, image_pairs, label_pairs, sigma_pairs, delta_sigma_pairs, model_args, fold):
        alpha, epochs, lamb, batch_size, wd, num_folds = model_args
        idx = np.arange(len(image_pairs))
        train_mask = (idx % num_folds != fold)
        test_mask = (idx % num_folds == fold)
        train_labels, test_labels = label_pairs[train_mask], label_pairs[test_mask]
        train_del_sigmas, test_del_sigmas = delta_sigma_pairs[train_mask], delta_sigma_pairs[test_mask]
        train_dl = DataLoader(image_pairs[train_mask], batch_size=batch_size, shuffle=False)
        test_dl = DataLoader(image_pairs[test_mask], batch_size=batch_size, shuffle=False)
        train_loss_vals, train_acc, test_loss_vals, test_acc = [], [], [], []
        opt = optim.Adam(self.parameters(), lr=alpha, weight_decay=wd)
        for e in tqdm(range(epochs), desc=f'Fold {fold + 1}'):
            total_loss = 0
            imgs_seen = 0
            correct = 0
            for i, batch in enumerate(train_dl):
                eff_batch_size = min(batch_size, batch.shape[0])
                imgs_seen += 2 * eff_batch_size
                try:
                    img1 = batch[:, 0, :, :, :].unsqueeze(1)
                    img2 = batch[:, 1, :, :, :].unsqueeze(1)
                except:
                    break
                pred1, pred2 = self.forward(img1), self.forward(img2)
                emb1, emb2 = self.encode(img1).to(self.device), self.encode(img2).to(self.device)
                ds = torch.tensor(train_del_sigmas[i*eff_batch_size : (i+1)*eff_batch_size]).to(self.device)
                l1 = torch.tensor([tup[0] for tup in train_labels[i*eff_batch_size : (i+1)*eff_batch_size]])
                l2 = torch.tensor([tup[1] for tup in train_labels[i*eff_batch_size : (i+1)*eff_batch_size]])             
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
            train_loss_vals.append(total_loss / imgs_seen)
            train_acc.append(correct / imgs_seen)
            ## TESTING PORTION
            fold_preds = []
            with torch.no_grad():
                total_loss = 0
                imgs_seen = 0
                correct = 0
                for i, batch in enumerate(test_dl):                   
                    eff_batch_size = min(batch_size, batch.shape[0])
                    imgs_seen += 2 * eff_batch_size
                    try:
                        img1 = batch[:, 0, :, :, :].unsqueeze(1)
                        img2 = batch[:, 1, :, :, :].unsqueeze(1)
                    except:
                        break
                    pred1, pred2 = self.forward(img1), self.forward(img2)
                    fold_preds.append(pred1)
                    fold_preds.append(pred2)
                    emb1, emb2 = self.encode(img1).to(self.device), self.encode(img2).to(self.device)
                    ds = torch.tensor(test_del_sigmas[i*eff_batch_size : (i+1)*eff_batch_size]).to(self.device)           
                    l1 = torch.tensor([tup[0] for tup in test_labels[i*eff_batch_size : (i+1)*eff_batch_size]])
                    l2 = torch.tensor([tup[1] for tup in test_labels[i*eff_batch_size : (i+1)*eff_batch_size]])
                    correct += self.count_correct_batch(pred1, l1)
                    correct += self.count_correct_batch(pred2, l2)
                    loss = self.pair_loss(emb_t1=emb1, emb_t2=emb2, delta_sigma=ds, pred_1=pred1, pred_2=pred2,
                                          labels_1=l1, labels_2=l2, lamb=lamb, tau=self.tau)
                    total_loss += float(loss.data)
                test_loss_vals.append(total_loss / imgs_seen)
                test_acc.append(correct / imgs_seen)
        print(f'    Final Epoch Results -> avg train loss = {round(train_loss_vals[-1], 4)} | avg test loss = {round(test_loss_vals[-1], 4)} | train acc = {round(train_acc[-1], 4)} | test acc = {round(test_acc[-1], 4)}\n')
        return fold_preds, test_acc, test_loss_vals

if __name__ == '__main__':
    alpha = args.lr
    epochs = args.epochs
    lamb = args.lamb
    batch_size = args.batch_size
    wd = args.wd
    num_folds = args.num_folds
    model_args = (alpha, epochs, lamb, batch_size, wd, num_folds)
    print(f'Run Initialized: epochs = {epochs}; alpha = {alpha}; batch_size = {batch_size}; lambda = {lamb}; FOLDS = {num_folds}')
    image_pairs, label_pairs, sigma_pairs, delta_sigma_pairs, id_pairs = load_matched_data(sigma_feat='npz')
    print(f'Data Loaded - {2 * len(image_pairs)} matched patients')
    model = Net(image_dim=64, tau_dim=2048).double()
    print(f'Model Initialized: image_dim = {model.dim}x{model.dim}x{model.dim}; tau_dim = {model.tau.shape[0]}')
    print(f'--> Starting Training')
    for k in range(num_folds):
        model = Net(image_dim=64, tau_dim=2048).double()
        fold_preds, fold_accs, fold_losses = model.train_fold(image_pairs, label_pairs, sigma_pairs,
                                                              delta_sigma_pairs, model_args, fold=k)
        with open(f'final_preds_fold_{k}.pickle', 'wb') as handle:
            pk.dump(fold_preds, handle)
        with open(f'final_acc_fold_{k}.pickle', 'wb') as handle:
            pk.dump(fold_accs, handle)
        with open(f'final_loss_fold_{k}.pickle', 'wb') as handle:
            pk.dump(fold_losses, handle)
        with open(f'ids_pidn_fold_{k}.pickle', 'wb') as handle:
            idx = np.arange(len(image_pairs))
            fold_mask = (idx % num_folds == k)
            pk.dump(id_pairs[fold_mask], handle)
        torch.save(model.state_dict(), os.path.join('', f'trained_model_fold_{k}.pth'))
