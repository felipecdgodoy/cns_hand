import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import warnings
import pickle as pk

from torch.utils.data import DataLoader

from tqdm import tqdm

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
parser.add_argument('--lamb', type=float, default=0, help='disentanglement factor')
parser.add_argument('--batch_size', type=int, default=16, help='input batch size for training')
parser.add_argument('--wd', type=float, default=0.0, help='weight decay')
parser.add_argument('--seed', type=int, default=0, help='seed')
parser.add_argument('--W', type=float, default=1.05, help='additional penalizing weighting for 1s over 0s, on top of ratio')
parser.add_argument('--num_folds', type=int, default=5, help='number of folds to split data for CV')
parser.add_argument('--verbose', type=int, default=0, help='defines how much info is logged to console during training')
args = parser.parse_args()

# set cuda device (GPU / CPU)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if args.verbose >= 1:
    print(f'Device: {device}')

# set deterministic behavior based on seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False # DO NOT REMOVE!

# DEBUG
# torch.autograd.set_detect_anomaly(True)

class Net(nn.Module):

    def __init__(self, image_dim=64, tau_dim=2048):
        """ image_dim (int): value 'D' such that all images are a 3D-volume DxDxD"""
        """ tau_dim (int): number of dimensions to encode the tau-direction in"""
        super(Net, self).__init__()
        self.device = device
        self.dim = image_dim
        self.split = 427
        self.thresh = 0.5
        self.W = args.W
        self.ratio = None # ratio between label 0s and label 1s for the BCE loss weighting 
        # Encoder from 3D-volume input to latent-space vector
        self.first_encode = nn.Sequential(
            nn.Sequential(nn.Conv3d(in_channels=1, out_channels=4, kernel_size=3, padding=1),
                          nn.ReLU(),
                          nn.MaxPool3d(kernel_size=2)),
            nn.BatchNorm3d(4),
            nn.Sequential(nn.Conv3d(in_channels=4, out_channels=8, kernel_size=3, padding=1),
                          nn.ReLU(),
                          nn.MaxPool3d(kernel_size=2)),
            nn.BatchNorm3d(8),
            nn.Sequential(nn.Conv3d(in_channels=8, out_channels=4, kernel_size=3, padding=1),
                          nn.ReLU(),
                          nn.MaxPool3d(kernel_size=2)),
            nn.BatchNorm3d(4),
            nn.Sequential(nn.Conv3d(in_channels=4, out_channels=2, kernel_size=3, padding=1),
                          nn.ReLU(),
                          nn.MaxPool3d(kernel_size=2)),
            nn.Flatten(start_dim=1),
        )

        self.second_encode = nn.Sequential(
            nn.Sequential(nn.Conv3d(in_channels=1, out_channels=4, kernel_size=3, padding=1),
                          nn.ReLU(),
                          nn.MaxPool3d(kernel_size=2)),
            nn.BatchNorm3d(4),
            nn.Sequential(nn.Conv3d(in_channels=4, out_channels=8, kernel_size=3, padding=1),
                          nn.ReLU(),
                          nn.MaxPool3d(kernel_size=2)),
            nn.BatchNorm3d(8),
            nn.Sequential(nn.Conv3d(in_channels=8, out_channels=4, kernel_size=3, padding=1),
                          nn.ReLU(),
                          nn.MaxPool3d(kernel_size=2)),
            nn.BatchNorm3d(4),
            nn.Sequential(nn.Conv3d(in_channels=4, out_channels=2, kernel_size=3, padding=1),
                          nn.ReLU(),
                          nn.MaxPool3d(kernel_size=2)),
            nn.Flatten(start_dim=1),
        )

        self.first_linear = nn.Sequential(
            nn.Linear(in_features=128, out_features=32),
            nn.LeakyReLU(),
            # nn.Linear(in_features=256, out_features=128),
            # nn.LeakyReLU(),
            nn.Linear(in_features=32, out_features=1),
            # nn.LeakyReLU(),
            # nn.Linear(16, 1)
        ).to(self.device)

        self.second_linear = nn.Sequential(
            nn.Linear(in_features=128, out_features=32),
            nn.LeakyReLU(),
            # nn.Linear(in_features=256, out_features=128),
            # nn.LeakyReLU(),
            nn.Linear(in_features=32, out_features=1),
            # nn.LeakyReLU(),
            # nn.Linear(16, 1)
        ).to(self.device)

    def forward(self, x):
        e1 = self.first_encode(x) ## Encode to latent-space
        e2 = self.second_encode(x)
        e = 0.5 * (e1 + e2)
        y1 = nn.Sigmoid()(self.first_linear(e1)) # first label classif
        y2 = nn.Sigmoid()(self.second_linear(e2)) # second label classif
        y = torch.stack((y1, y2), dim=1).squeeze()
        return e, y
    
    def bce_loss(self, emb_t1, emb_t2, delta_sigma, pred_1, pred_2, labels_1, labels_2, lamb):
        # bce = nn.BCELoss()
        # pred_1 = pred_1.nan_to_num(0.5).double()
        # pred_2 = pred_2.nan_to_num(0.5).double()
        # return bce(pred_1, labels_1) + bce(pred_2, labels_2)
        pred_1 = pred_1.nan_to_num(0.5).double()
        pred_2 = pred_2.nan_to_num(0.5).double()
        # first element of pairs
        w = torch.ones(labels_1.shape[0], device=self.device)
        w[labels_1[:, 0] == 1] = self.ratio
        bce = nn.BCELoss(weight=w)
        bce_loss = bce(pred_1[:, 0], labels_1[:, 0])
        w = torch.ones(labels_1.shape[0], device=self.device)
        w[labels_1[:, 1] == 1] = self.ratio
        bce = nn.BCELoss(weight=w)
        bce_loss += bce(pred_1[:, 1], labels_1[:, 1])
        # second element of pairs
        w = torch.ones(labels_2.shape[0], device=self.device)
        w[labels_2[:, 0] == 1] = self.ratio
        bce = nn.BCELoss(weight=w)
        bce_loss += bce(pred_2[:, 0], labels_2[:, 0])
        w = torch.ones(labels_2.shape[0], device=self.device)
        w[labels_2[:, 1] == 1] = self.ratio
        bce = nn.BCELoss(weight=w)
        bce_loss += bce(pred_2[:, 1], labels_2[:, 1])
        return bce_loss

    def repr_loss(self, emb_t1, emb_t2, delta_sigma, pred_1, pred_2, labels_1, labels_2, lamb):
        bs = emb_t1.shape[0]
        ones = torch.ones((1, 1, self.dim, self.dim, self.dim), device=self.device).double()
        # tau = self.encode(ones).flatten()
        tau = 0.5 * (self.first_encode(ones).flatten() + self.second_encode(ones).flatten())
        proj_e1_len = (emb_t1 @ tau) / tau.norm(p=1)
        proj_e2_len = (emb_t2 @ tau) / tau.norm(p=1)
        proj_len_diff = proj_e2_len - proj_e1_len
        disentangle_loss = (proj_len_diff - delta_sigma).nan_to_num(0).square().sum()
        return lamb * disentangle_loss
    
    def pair_loss(self, emb_t1, emb_t2, delta_sigma, pred_1, pred_2, labels_1, labels_2, lamb):
        bce = self.bce_loss(emb_t1, emb_t2, delta_sigma, pred_1, pred_2, labels_1, labels_2, lamb)
        if lamb == 0:
            return bce
        repr = self.repr_loss(emb_t1, emb_t2, delta_sigma, pred_1, pred_2, labels_1, labels_2, lamb)
        return bce + repr

    def update_tau(self, eps=1e-4):
        ones = torch.ones((1, 1, self.dim, self.dim, self.dim), device=self.device).double()
        # tau = self.encode(ones).flatten()
        tau = 0.5 * (self.first_encode(ones).flatten() + self.second_encode(ones).flatten())
        tau[tau > 0] += eps
        tau[tau < 0] -= eps
        return tau

    def count_correct_batch(self, batch_preds, batch_labels):
        corr = ((batch_preds >= self.thresh) == batch_labels).all(dim=1).sum()
        return int(corr.data)
    
    def load_matched_data(self, path_to_pickles='', sigma_feat='npz', seed=2023):
        sigma_feat = sigma_feat.lower()
        assert sigma_feat in ['npz', 'age', 'gender'], f'Unsurported feature. must be age, npz, or gender'
        from sklearn.utils import shuffle
        # Read in the data
        with open(f'{path_to_pickles}matched_patients.pickle', 'rb') as handle:
            ucsf_df = pk.load(handle)
        ids = np.array(ucsf_df['id'])
        images = np.stack(ucsf_df['image'])
        labels = np.array(ucsf_df['label'])
        sigmas = np.array(ucsf_df[sigma_feat])
        # set the weighting ratio for the BCE Loss
        lab = labels[:self.split]
        num_ones = lab.sum().sum()
        num_zeros = 2 * int(lab.shape[0]) - num_ones
        self.ratio = self.W * (num_zeros / num_ones)
        # Apply deterministic shuffle in the list (meaning all lists are shuffled in a same random order)
        #   pair samples with sigma feat (NPZ) present together!
        has_sigma = ~np.isnan(sigmas)[:self.split]
        ordering = np.hstack((
            shuffle(np.arange(self.split)[has_sigma], random_state=seed),
            shuffle(np.arange(self.split)[~has_sigma], random_state=seed),
            shuffle(np.arange(self.split, len(images)), random_state=seed)
        ))
        ids = ids[ordering]
        images = images[ordering]
        labels = labels[ordering]
        sigmas = sigmas[ordering]
        # Make into pairs
        num_pairs = len(labels) // 2
        image_pairs = torch.tensor([(images[2*i], images[2*i + 1]) for i in range(num_pairs)])
        label_pairs = torch.tensor([(labels[2*i], labels[2*i + 1]) for i in range(num_pairs)]).double()
        sigma_pairs = torch.tensor([(sigmas[2*i], sigmas[2*i + 1]) for i in range(num_pairs)])
        delta_sigma_pairs = torch.tensor([sigmas[2*i] - sigmas[2*i + 1] for i in range(num_pairs)])
        id_pairs = torch.tensor([(ids[2*i], ids[2*i + 1]) for i in range(num_pairs)])
        return image_pairs, label_pairs, sigma_pairs, delta_sigma_pairs, id_pairs

    def train_fold(self, image_pairs, label_pairs, sigma_pairs, delta_sigma_pairs, model_args, fold, verbose):
        alpha, epochs, lamb, batch_size, wd, num_folds, _ = model_args
        print(self.W, lamb)
        # train_mask = np.arange(len(image_pairs)) % num_folds != fold
        # valid_mask = np.arange(len(image_pairs)) % num_folds == (0 if fold != 0 else 1)
        train_mask = np.arange(len(image_pairs)) % num_folds == 0 # all
        valid_mask = np.arange(len(image_pairs)) % num_folds != 0 # none
        train_mask[(self.split - 1) // 2:] = False
        valid_mask[(self.split - 1) // 2:] = False
        test_mask = np.arange((self.split + 1) // 2, len(image_pairs))
        train_labels, valid_labels, test_labels = label_pairs[train_mask], label_pairs[valid_mask], label_pairs[test_mask]
        train_del_sigmas, valid_del_sigmas, test_del_sigmas = delta_sigma_pairs[train_mask], delta_sigma_pairs[valid_mask], delta_sigma_pairs[test_mask]
        train_dl = DataLoader(image_pairs[train_mask], batch_size=batch_size, shuffle=False)
        valid_dl = DataLoader(image_pairs[valid_mask], batch_size=batch_size, shuffle=False)
        test_dl = DataLoader(image_pairs[test_mask], batch_size=batch_size, shuffle=False)
        train_loss_vals, train_acc, valid_loss_vals, valid_acc, test_loss_vals, test_acc = [], [], [], [], [], []
        opt = optim.Adam(self.parameters(), lr=alpha, weight_decay=wd)
        fold_preds = dict()
        for e in tqdm(range(epochs), desc=f'Fold {fold}'):
            total_loss, imgs_seen, correct = 0, 0, 0
            for i, batch in enumerate(train_dl):
                eff_batch_size = batch.shape[0]
                imgs_seen += 2 * eff_batch_size
                # SHAPE: (batch_size, channel_in, image_dim, image_dim, image_dim)
                img1 = batch[:, 0, :, :, :].unsqueeze(1).to(self.device)
                img2 = batch[:, 1, :, :, :].unsqueeze(1).to(self.device)
                img_input = torch.cat((img1, img2))
                embeddings, class_preds = self.forward(img_input)
                pred1, pred2 = class_preds[:eff_batch_size], class_preds[eff_batch_size:]
                emb1, emb2 = embeddings[:eff_batch_size], embeddings[eff_batch_size:]
                ds = train_del_sigmas[i*eff_batch_size : (i+1)*eff_batch_size].to(self.device)
                l1 = train_labels[i*eff_batch_size : (i+1)*eff_batch_size, 0, :].to(self.device)
                l2 = train_labels[i*eff_batch_size : (i+1)*eff_batch_size, 1, :].to(self.device)
                # print('TRAIN\n', torch.hstack((pred1, l1))[:3].detach().cpu().numpy().round(decimals=1))
                correct += self.count_correct_batch(pred1, l1)
                correct += self.count_correct_batch(pred2, l2)
                opt.zero_grad()
                loss1 = self.bce_loss(emb_t1=emb1, emb_t2=emb2, delta_sigma=ds, pred_1=pred1, pred_2=pred2,
                                        labels_1=l1, labels_2=l2, lamb=lamb)
                if lamb > 0:
                    loss2 = self.repr_loss(emb_t1=emb1, emb_t2=emb2, delta_sigma=ds, pred_1=pred1, pred_2=pred2,
                                            labels_1=l1, labels_2=l2, lamb=lamb)
                    loss1.backward(retain_graph=True)                
                    loss2.backward()
                    total_loss = total_loss + float(loss1.data) + float(loss2.data)
                else:
                    loss1.backward()
                    total_loss = total_loss + float(loss1.data)
                opt.step()                
            p = nn.utils.parameters_to_vector(self.parameters())
            train_loss_vals.append(total_loss / 1)
            train_acc.append(correct / imgs_seen)
            ## VALIDATION PORTION
            if num_folds >= 2:
                with torch.no_grad():
                    total_loss, imgs_seen, correct = 0, 0, 0
                    for i, batch in enumerate(valid_dl):
                        eff_batch_size = batch.shape[0]
                        imgs_seen += 2 * eff_batch_size
                        img1 = batch[:, 0, :, :, :].unsqueeze(1).to(self.device)
                        img2 = batch[:, 1, :, :, :].unsqueeze(1).to(self.device)
                        img_input = torch.cat((img1, img2))
                        embeddings, class_preds = self.forward(img_input)
                        pred1, pred2 = class_preds[:eff_batch_size], class_preds[eff_batch_size:]
                        emb1, emb2 = embeddings[:eff_batch_size], embeddings[eff_batch_size:]
                        ds = valid_del_sigmas[i*eff_batch_size : (i+1)*eff_batch_size].to(self.device)
                        l1 = valid_labels[i*eff_batch_size : (i+1)*eff_batch_size, 0, :].to(self.device)
                        l2 = valid_labels[i*eff_batch_size : (i+1)*eff_batch_size, 1, :].to(self.device)
                        correct += self.count_correct_batch(pred1, l1)
                        correct += self.count_correct_batch(pred2, l2)
                        loss = self.pair_loss(emb_t1=emb1, emb_t2=emb2, delta_sigma=ds, pred_1=pred1, pred_2=pred2,
                                            labels_1=l1, labels_2=l2, lamb=lamb)
                        total_loss += float(loss.data)
                    valid_loss_vals.append(total_loss / 1)
                    try:
                        valid_acc.append(correct / imgs_seen)
                    except:
                        valid_acc.append(correct / 1)
            ## TESTING PORTION
            fold_preds[e] = []
            fold_labels = []
            with torch.no_grad():
                total_loss, imgs_seen, correct = 0, 0, 0
                for i, batch in enumerate(test_dl):
                    eff_batch_size = batch.shape[0]
                    imgs_seen += 2 * eff_batch_size
                    img1 = batch[:, 0, :, :, :].unsqueeze(1).to(self.device)
                    img2 = batch[:, 1, :, :, :].unsqueeze(1).to(self.device)
                    img_input = torch.cat((img1, img2))
                    embeddings, class_preds = self.forward(img_input)
                    pred1, pred2 = class_preds[:eff_batch_size], class_preds[eff_batch_size:]
                    emb1, emb2 = embeddings[:eff_batch_size], embeddings[eff_batch_size:]
                    ds = test_del_sigmas[i*eff_batch_size : (i+1)*eff_batch_size].to(self.device)
                    l1 = test_labels[i*eff_batch_size : (i+1)*eff_batch_size, 0, :].to(self.device)
                    l2 = test_labels[i*eff_batch_size : (i+1)*eff_batch_size, 1, :].to(self.device)
                    correct += self.count_correct_batch(pred1, l1)
                    correct += self.count_correct_batch(pred2, l2)
                    fold_preds[e].append(list(zip(pred1, pred2)))
                    fold_labels.append(list(zip(l1, l2)))
                    loss = self.pair_loss(emb_t1=emb1, emb_t2=emb2, delta_sigma=ds, pred_1=pred1, pred_2=pred2,
                                          labels_1=l1, labels_2=l2, lamb=lamb)
                    total_loss += float(loss.data)
                # print(' TEST\n', pred1[:20].detach().cpu().numpy().round(decimals=2)) # outside batch loop so it only prints once per epoch
                test_loss_vals.append(total_loss / 1)
                test_acc.append(correct / imgs_seen)
                torch.save(self.state_dict(), os.path.join(dir_destination, f'interm_model_lamb_{lamb}_epoch_{e+1}.pth'))
            FREQ = 1 # adjust as needed for verbosity
            if e % FREQ == 0:
                if verbose >= 1:
                    print(f'    Epoch {e+1} Results -> train loss = {round(train_loss_vals[-1], 6)} | valid loss = {round(valid_loss_vals[-1], 6)} | test loss = {round(test_loss_vals[-1], 6)}')
                    print(f'                     -> train  acc = {round(100*train_acc[-1], 2)}% | valid acc = {round(100*valid_acc[-1], 2)} | test acc = {round(100*test_acc[-1], 2)}%')
                    if verbose >= 2:
                        if len(train_acc) >= (FREQ + 1):
                            train_acc_delta = train_acc[-1] - train_acc[-FREQ-1]
                            test_acc_delta = test_acc[-1] - test_acc[-FREQ-1]
                            s1 = '+' if train_acc_delta > 0 else ''
                            s2 = '+' if test_acc_delta > 0 else ''
                            print(f'    Train Acc Delta: {s1}{round(100*train_acc_delta, 2)}% \n     Test Acc Delta: {s2}{round(100*test_acc_delta, 2)}%')
                            train_loss_delta = train_loss_vals[-1] - train_loss_vals[-FREQ-1]
                            test_loss_delta = test_loss_vals[-1] - test_loss_vals[-FREQ-1]
                            s1 = '+' if train_loss_delta > 0 else ''
                            s2 = '+' if test_loss_delta > 0 else ''            
                            print(f'    Train Loss Delta: {s1}{round(train_loss_delta, 6)} \n     Test Loss Delta: {s2}{round(test_loss_delta, 6)}')
        best_index = np.argmax(test_acc)
        print(f'[DONE] FOLD {fold} --- Best Epoch Results ({best_index + 1}) -> TRAIN ACC = {round(100*train_acc[best_index], 2)}% | TEST ACC = {round(100*test_acc[best_index], 2)}%\n')
        print([round(x, 4) for x in train_acc], '\n')
        print([round(x, 4) for x in valid_acc], '\n')
        print([round(x, 4) for x in test_acc], '\n')
        return fold_preds[best_index], test_acc, test_loss_vals, fold_labels

if __name__ == '__main__':
    image_dim = 64
    tau_dim = 2048
    alpha = args.lr
    epochs = args.epochs
    lamb = args.lamb
    batch_size = args.batch_size
    wd = args.wd
    seed = args.seed
    num_folds = args.num_folds
    verbose = args.verbose
    assert tau_dim >= 128
    model_args = (alpha, epochs, lamb, batch_size, wd, num_folds, verbose)
    dir_destination = f'epochs = {epochs} - alpha = {alpha} - batch_size = {batch_size} - lambda = {lamb} - seed {seed}'
    try:
        os.mkdir(dir_destination) # make new directory for results
    except:
        pass # directory already exists -- results will be overridden
    model = Net(image_dim=image_dim, tau_dim=tau_dim).double().cuda()
    image_pairs, label_pairs, sigma_pairs, delta_sigma_pairs, id_pairs = model.load_matched_data(sigma_feat='npz')
    ratio = model.ratio
    if verbose >= 1:
        print(f'Run Initialized: epochs = {epochs}; alpha = {alpha}; batch_size = {batch_size}; lambda = {lamb}; FOLDS = {num_folds}')
        print(f'Data Loaded - {2 * len(image_pairs)} matched patients')
        print(f'Model Initialized: image_dim = {image_dim}x{image_dim}x{image_dim}; tau_dim = {tau_dim}')
        if verbose >= 2:
            print(f'--> Starting Training')
    for k in range(num_folds):
        model = Net(image_dim=image_dim, tau_dim=tau_dim).double().cuda()
        model.ratio = ratio
        fold_preds, fold_accs, fold_losses, fold_labels = model.train_fold(image_pairs, label_pairs, sigma_pairs, 
                                                                           delta_sigma_pairs, model_args, fold=k, verbose=verbose)
        with open(f'{dir_destination}/final_preds_fold_{k}_{model_args}.pickle', 'wb') as handle:
            pk.dump(fold_preds, handle)
        with open(f'{dir_destination}/final_acc_fold_{k}_{model_args}.pickle', 'wb') as handle:
            pk.dump(fold_accs, handle)
        with open(f'{dir_destination}/final_loss_fold_{k}_{model_args}.pickle', 'wb') as handle:
            pk.dump(fold_losses, handle)
        with open(f'{dir_destination}/labels_fold_{k}_{model_args}.pickle', 'wb') as handle:
            pk.dump(fold_labels, handle)
        with open(f'{dir_destination}/ids_pidn_fold_{k}_{model_args}.pickle', 'wb') as handle:
            idx = np.arange(len(image_pairs))
            fold_mask = (idx % num_folds == k)
            pk.dump(id_pairs[fold_mask], handle)
        torch.save(model.state_dict(), os.path.join(dir_destination, f'trained_model_fold_{k}_{model_args}.pth'))