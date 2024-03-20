import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import warnings
import pickle as pk
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train')
parser.add_argument('--lamb', type=float, default=0, help='disentanglement factor')
parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training')
parser.add_argument('--wd', type=float, default=0.01, help='weight decay for optimizer')
parser.add_argument('--seed', type=int, default=1, help='seed for pair matching and model initialization')
parser.add_argument('--W', type=float, default=1.0, help='additional penalizing weighting for 1s over 0s, on top of ratio')
parser.add_argument('--th_ci', type=float, default=0.5, help='classification threshold on the CI axis (x-axis)')
parser.add_argument('--th_hiv', type=float, default=0.5, help='classification threshold on the HIV axis (y-axis)')
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

class PairedDataset(Dataset):

    def __init__(self, image_pairs, label_pairs, del_sigma_pairs, id_pairs):
        self.image_pairs = image_pairs
        self.label_pairs = label_pairs
        self.del_sigma_pairs = del_sigma_pairs
        self.id_pairs = id_pairs

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        return self.image_pairs[idx], self.label_pairs[idx], self.del_sigma_pairs[idx], self.id_pairs[idx]

class Net(nn.Module):

    def __init__(self, image_dim=64, tau_dim=2048, thresh_tuple=(0.5, 0.5), hand_split=None):
        """ image_dim (int): value 'D' such that all images are a 3D-volume DxDxD """
        """ tau_dim (int): number of dimensions to encode the tau-direction in """
        super(Net, self).__init__()
        self.device = device
        self.thresh = torch.tensor(thresh_tuple, device=self.device)
        self.dim = image_dim
        self.W = args.W
        self.ratio = None # ratio between label 0s and label 1s for the BCE loss weighting. !! dynamically populated at runtime !!
        self.hand_split = hand_split # separation index for (train+valid) | (test), marking the first index of the sequence of HAND samples at load-time !! dynamically populated at runtime if left None !!
        # Encoder from 3D-volume input to latent-space vector
        self.encode = nn.Sequential(
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
            nn.BatchNorm3d(2),                          
            nn.Flatten(start_dim=1),
        )

        self.first_linear = nn.Sequential(
            nn.Linear(in_features=128, out_features=32),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.Linear(in_features=32, out_features=1),
        ).to(self.device)

        self.second_linear = nn.Sequential(
            nn.Linear(in_features=128, out_features=32),
            nn.Dropout(0.5),            
            nn.LeakyReLU(),          
            nn.Linear(in_features=32, out_features=1),
        ).to(self.device)

    def forward(self, x):
        e = self.encode(x) ## Encode to latent-space
        y1 = nn.Sigmoid()(self.first_linear(e)) # first label classif
        y2 = nn.Sigmoid()(self.second_linear(e)) # second label classif
        y = torch.stack((y1, y2), dim=1).squeeze()
        return e, y
    
    def bce_loss(self, emb_t1, emb_t2, delta_sigma, pred_1, pred_2, labels_1, labels_2, lamb):
        bce = nn.BCELoss(reduction='none')
        first_bce = bce(pred_1, labels_1)
        second_bce = bce(pred_2, labels_2)        
        first_bce[labels_1 == 1] *= self.ratio
        second_bce[labels_2 == 1] *= self.ratio
        bce_loss = (first_bce + second_bce).sum()
        return bce_loss

    def repr_loss(self, emb_t1, emb_t2, delta_sigma, pred_1, pred_2, labels_1, labels_2, lamb):
        ones = torch.ones((1, 1, self.dim, self.dim, self.dim), device=self.device).double()
        tau = self.encode(ones).flatten()
        proj_e1_len = (emb_t1 @ tau) / tau.norm(p=1)
        proj_e2_len = (emb_t2 @ tau) / tau.norm(p=1)
        proj_len_diff = (proj_e2_len - proj_e1_len)
        disentangle_loss = (proj_len_diff - delta_sigma).nan_to_num(0).square().sum()
        return disentangle_loss
    
    def pair_loss(self, emb_t1, emb_t2, delta_sigma, pred_1, pred_2, labels_1, labels_2, lamb):
        bce = self.bce_loss(emb_t1, emb_t2, delta_sigma, pred_1, pred_2, labels_1, labels_2, lamb)
        if lamb == 0:
            return bce
        disentang = self.repr_loss(emb_t1, emb_t2, delta_sigma, pred_1, pred_2, labels_1, labels_2, lamb)
        return bce + lamb * disentang

    def count_correct_batch(self, batch_preds, batch_labels):
        corr = ((batch_preds >= self.thresh) == batch_labels).all(dim=1).sum()
        return int(corr.data)
    
    def balanced_accuracy(self, pair_preds, pair_labels):
        full_preds = torch.vstack([torch.vstack(tup) for tup in pair_preds[0]])
        full_labels = torch.vstack([torch.vstack(tup) for tup in pair_labels[0]])
        if len(pair_labels) > 1:
            for i in range(1, len(pair_labels)):
                full_preds = torch.vstack((full_preds, torch.vstack([torch.vstack(tup) for tup in pair_preds[i]])))
                full_labels = torch.vstack((full_labels, torch.vstack([torch.vstack(tup) for tup in pair_labels[i]])))
        pred_correctness = ((full_preds >= self.thresh) == full_labels).all(dim=1)
        accs = list()
        for label in full_labels.unique(dim=0):
            label_mask = (full_labels == label).all(dim=1)
            acc = int(pred_correctness[label_mask].sum()) / int(pred_correctness[label_mask].shape[0])
            accs.append(acc)
        return np.array(accs).mean()
    
    def load_matched_data(self, path_to_pickles='', sigma_feat='npz', seed=1):
        sigma_feat = sigma_feat.lower()
        assert sigma_feat in ['npz', 'age', 'gender'], f'Unsurported feature. must be age, npz, or gender'
        from sklearn.utils import shuffle
        # Read in the data
        with open(f'{path_to_pickles}matched_patients.pickle', 'rb') as handle:
            ucsf_df = pk.load(handle)
        if not self.hand_split:
            hand_split = len(ucsf_df) - len(ucsf_df[ucsf_df['label'].astype(str) == '[1 1]']) # remove the HAND samples
            self.hand_split = hand_split
        ids = np.array(ucsf_df['id'])
        images = np.stack(ucsf_df['image'])
        labels = np.array(ucsf_df['label'])
        sigmas = np.array(ucsf_df[sigma_feat])
        # set the weighting ratio for the BCE Loss
        lab = labels[:self.hand_split]
        num_ones = lab.sum().sum()
        num_zeros = 2 * int(lab.shape[0]) - num_ones
        self.ratio = self.W * (num_zeros / num_ones)
        # Apply deterministic shuffle in the list (meaning all lists are shuffled in a same random order)
        #   pair samples with sigma feat (NPZ) present together!
        has_sigma = ~np.isnan(sigmas)[:self.hand_split]
        ordering = np.hstack((
            shuffle(np.arange(self.hand_split)[has_sigma], random_state=seed),
            shuffle(np.arange(self.hand_split)[~(has_sigma)], random_state=seed),
            shuffle(np.arange(self.hand_split, len(images)), random_state=seed)
        ))
        ids = ids[ordering]
        images = images[ordering]
        labels = labels[ordering]
        sigmas = sigmas[ordering]
        delta_sigmas = np.diff(sigmas)
        return images, labels, sigmas, delta_sigmas, ids, self.hand_split
        
    def train_fold(self, all_images, all_labels, all_sigmas, all_delta_sigmas, model_args, fold, verbose):
        from sklearn.model_selection import StratifiedKFold
        alpha, epochs, lamb, batch_size, wd, num_folds, _ = model_args
        label_reference = {'[0 0]' : 'CTRL', '[0 1]' : 'HIV', '[1 0]' : 'MCI', '[1 1]' : 'HAND'}
        all_named_labels = np.array([label_reference[str(lab)] for lab in all_labels])
        skf = StratifiedKFold(n_splits=num_folds, shuffle=False) # shuffled is handled with matching at load-time
        for i, (train_mask, valid_mask) in enumerate(skf.split(all_images[:self.hand_split], all_named_labels[:self.hand_split])):
            if i == fold:
                break # retains train_mask and valid_mask for the specific fold --this is the endorsed documentation approach
        test_mask = np.arange(self.hand_split, len(all_images))
        # Make into pairs & split into different sets
        # --> TRAIN
        images = all_images[train_mask]
        labels = all_labels[train_mask]
        sigmas = all_sigmas[train_mask]
        num_pairs = len(labels) // 2
        train_images = torch.tensor([(images[2*i], images[2*i + 1]) for i in range(num_pairs)])
        train_labels = torch.tensor([(labels[2*i], labels[2*i + 1]) for i in range(num_pairs)]).double()
        train_del_sigmas = torch.tensor([sigmas[2*i] - sigmas[2*i + 1] for i in range(num_pairs)])
        train_id_pairs = torch.tensor([(ids[2*i], ids[2*i + 1]) for i in range(num_pairs)])
        train_dataset = PairedDataset(train_images, train_labels, train_del_sigmas, train_id_pairs)
        train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        # --> VALID
        images = all_images[valid_mask]
        labels = all_labels[valid_mask]
        sigmas = all_sigmas[valid_mask]
        num_pairs = len(labels) // 2
        valid_images = torch.tensor([(images[2*i], images[2*i + 1]) for i in range(num_pairs)])
        valid_labels = torch.tensor([(labels[2*i], labels[2*i + 1]) for i in range(num_pairs)]).double()
        valid_del_sigmas = torch.tensor([sigmas[2*i] - sigmas[2*i + 1] for i in range(num_pairs)])
        valid_id_pairs = torch.tensor([(ids[2*i], ids[2*i + 1]) for i in range(num_pairs)])
        valid_dataset = PairedDataset(valid_images, valid_labels, valid_del_sigmas, valid_id_pairs)
        valid_dl = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        # --> TEST
        images = all_images[test_mask]
        labels = all_labels[test_mask]
        sigmas = all_sigmas[test_mask]
        num_pairs = len(labels) // 2
        test_images = torch.tensor([(images[2*i], images[2*i + 1]) for i in range(num_pairs)])
        test_labels = torch.tensor([(labels[2*i], labels[2*i + 1]) for i in range(num_pairs)]).double()
        test_del_sigmas = torch.tensor([sigmas[2*i] - sigmas[2*i + 1] for i in range(num_pairs)])
        test_id_pairs = torch.tensor([(ids[2*i], ids[2*i + 1]) for i in range(num_pairs)])
        test_dataset = PairedDataset(test_images, test_labels, test_del_sigmas, test_id_pairs)
        test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        # Setup metrics & optimizer
        train_loss_vals, train_acc, valid_loss_vals, valid_acc, test_loss_vals, test_acc = [], [], [], [], [], []
        fold_train_preds, fold_valid_preds, fold_test_preds = dict(), dict(), dict()
        opt = optim.Adam(self.parameters(), lr=alpha, weight_decay=wd)
        # BEGIN TRAINING
        for e in tqdm(range(epochs), desc=f'Fold {fold}'):
            fold_train_preds[e] = []
            fold_train_labels = []
            total_loss, imgs_seen, correct = 0, 0, 0
            for i, batch in enumerate(train_dl):
                batch_images, batch_labels, batch_del_sigmas, batch_id_pairs = batch
                eff_batch_size = batch_images.shape[0]
                imgs_seen += 2 * eff_batch_size
                # shape: (batch_size, pair_position, image_dim, image_dim, image_dim)
                img1 = batch_images[:, 0, :, :, :].unsqueeze(1).to(self.device)
                img2 = batch_images[:, 1, :, :, :].unsqueeze(1).to(self.device)
                img_input = torch.cat((img1, img2))
                embeddings, class_preds = self.forward(img_input)
                pred1, pred2 = class_preds[:eff_batch_size], class_preds[eff_batch_size:]
                emb1, emb2 = embeddings[:eff_batch_size], embeddings[eff_batch_size:]
                ds = batch_del_sigmas.to(self.device)
                l1 = batch_labels[:, 0, :].to(self.device)
                l2 = batch_labels[:, 1, :].to(self.device)
                correct += self.count_correct_batch(pred1, l1)
                correct += self.count_correct_batch(pred2, l2)
                fold_train_preds[e].append(list(zip(pred1, pred2)))
                fold_train_labels.append(list(zip(l1, l2)))
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
            train_loss_vals.append(total_loss / imgs_seen)
            train_acc.append(self.balanced_accuracy(fold_train_preds[e], fold_train_labels))
            ## VALIDATION PORTION
            fold_valid_preds[e] = []
            fold_valid_labels = []
            if num_folds >= 2:
                with torch.no_grad():
                    total_loss, imgs_seen, correct = 0, 0, 0
                    for i, batch in enumerate(valid_dl):
                        batch_images, batch_labels, batch_del_sigmas, batch_id_pairs = batch
                        eff_batch_size = batch_images.shape[0]
                        imgs_seen += 2 * eff_batch_size
                        # shape: (batch_size, pair_position, image_dim, image_dim, image_dim)
                        img1 = batch_images[:, 0, :, :, :].unsqueeze(1).to(self.device)
                        img2 = batch_images[:, 1, :, :, :].unsqueeze(1).to(self.device)
                        img_input = torch.cat((img1, img2))
                        embeddings, class_preds = self.forward(img_input)
                        pred1, pred2 = class_preds[:eff_batch_size], class_preds[eff_batch_size:]
                        emb1, emb2 = embeddings[:eff_batch_size], embeddings[eff_batch_size:]
                        ds = batch_del_sigmas.to(self.device)
                        l1 = batch_labels[:, 0, :].to(self.device)
                        l2 = batch_labels[:, 1, :].to(self.device)
                        correct += self.count_correct_batch(pred1, l1)
                        correct += self.count_correct_batch(pred2, l2)
                        fold_valid_preds[e].append(list(zip(pred1, pred2)))
                        fold_valid_labels.append(list(zip(l1, l2)))
                        loss = self.pair_loss(emb_t1=emb1, emb_t2=emb2, delta_sigma=ds, pred_1=pred1, pred_2=pred2,
                                            labels_1=l1, labels_2=l2, lamb=lamb)
                        total_loss += float(loss.data)
                    valid_loss_vals.append(total_loss / imgs_seen)
                    valid_acc.append(self.balanced_accuracy(fold_valid_preds[e], fold_valid_labels))
            ## TESTING PORTION
            fold_test_preds[e] = []
            fold_test_labels = []
            with torch.no_grad():
                total_loss, imgs_seen, correct = 0, 0, 0
                for i, batch in enumerate(test_dl):
                    batch_images, batch_labels, batch_del_sigmas, batch_id_pairs = batch
                    eff_batch_size = batch_images.shape[0]
                    imgs_seen += 2 * eff_batch_size
                    # shape: (batch_size, pair_position, image_dim, image_dim, image_dim)
                    img1 = batch_images[:, 0, :, :, :].unsqueeze(1).to(self.device)
                    img2 = batch_images[:, 1, :, :, :].unsqueeze(1).to(self.device)
                    img_input = torch.cat((img1, img2))
                    embeddings, class_preds = self.forward(img_input)
                    pred1, pred2 = class_preds[:eff_batch_size], class_preds[eff_batch_size:]
                    emb1, emb2 = embeddings[:eff_batch_size], embeddings[eff_batch_size:]
                    ds = batch_del_sigmas.to(self.device)
                    l1 = batch_labels[:, 0, :].to(self.device)
                    l2 = batch_labels[:, 1, :].to(self.device)
                    correct += self.count_correct_batch(pred1, l1)
                    correct += self.count_correct_batch(pred2, l2)
                    fold_test_preds[e].append(list(zip(pred1, pred2)))
                    fold_test_labels.append(list(zip(l1, l2)))
                    loss = self.pair_loss(emb_t1=emb1, emb_t2=emb2, delta_sigma=ds, pred_1=pred1, pred_2=pred2,
                                          labels_1=l1, labels_2=l2, lamb=lamb)
                    total_loss += float(loss.data)
                # print(' TEST\n', pred1[:20].detach().cpu().numpy().round(decimals=2)) # outside batch loop so it only prints once per epoch
                test_loss_vals.append(total_loss / imgs_seen)
                test_acc.append(correct / imgs_seen)
            FREQ = 1 # adjust as needed for verbosity
            if e % FREQ == 0:
                if verbose >= 1:
                    print(f'    Epoch {e+1} Results -> train loss = {round(train_loss_vals[-1], 4)} | valid loss = {round(valid_loss_vals[-1], 4)} | test loss = {round(test_loss_vals[-1], 4)}')
                    print(f'                        -> train  acc = {round(100*train_acc[-1], 2)}% | valid acc = {round(100*valid_acc[-1], 2)}% | test acc = {round(100*test_acc[-1], 2)}%')
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
                            print(f'    Train Loss Delta: {s1}{round(train_loss_delta, 4)} \n     Test Loss Delta: {s2}{round(test_loss_delta, 4)}')
        best_index = np.argmax(valid_acc)
        torch.save(self.state_dict(), os.path.join(dir_destination, f'saved_model_fold_{fold}.pth'))
        print(f'[DONE] FOLD {fold} --- Best Valid Epoch Results ({best_index + 1}) -> Train B-Acc = {round(100*train_acc[best_index], 2)}% | Valid B-Acc = {round(100*valid_acc[best_index], 2)}% | Test Acc = {round(100*test_acc[best_index], 2)}%\n')
        if verbose >= 1:
            print([round(x, 4) for x in train_acc], '\n')
            print([round(x, 4) for x in valid_acc], '\n')
            print([round(x, 4) for x in test_acc], '\n')
        return fold_test_preds[best_index], test_acc, test_loss_vals, fold_test_labels, fold_valid_preds[best_index], fold_valid_labels, valid_id_pairs

if __name__ == '__main__':
    image_dim = 64
    tau_dim = 128
    alpha = args.lr
    epochs = args.epochs
    lamb = args.lamb
    batch_size = args.batch_size
    wd = args.wd
    seed = args.seed
    num_folds = args.num_folds
    verbose = args.verbose
    W = args.W
    threshs = (args.th_ci, args.th_hiv)
    model_args = (alpha, epochs, lamb, batch_size, wd, num_folds, verbose)
    dir_destination = f'alpha = {alpha} - epochs = {epochs} - lambda = {lamb} - batch_size = {batch_size} - wd = {wd} - num_folds = {num_folds} - W = {W} - seed = {seed}'
    try:
        os.mkdir(dir_destination) # make new directory for results
    except:
        pass # directory already exists -- results will be overridden
    model = Net(image_dim=image_dim, tau_dim=tau_dim, thresh_tuple=threshs).double().cuda()
    images, labels, sigmas, delta_sigmas, ids, hand_split = model.load_matched_data(sigma_feat='npz', seed=seed)
    ratio = model.ratio
    print(f'Run Initialized: {dir_destination}')
    if verbose >= 1:
        print(f'Data Loaded - {len(images)} matched patients')
        print(f'Model Initialized: image_dim = {image_dim}x{image_dim}x{image_dim}')
        if verbose >= 2:
            print(f'--> Starting Training')
    for k in range(num_folds):
        model = Net(image_dim=image_dim, tau_dim=tau_dim, thresh_tuple=threshs, hand_split=hand_split).double().cuda()
        model.ratio = ratio
        fold_test_preds, fold_test_accs, fold_test_losses, fold_test_labels, fold_valid_preds, fold_valid_labels, fold_ids = model.train_fold(images,
                                                                                                                                              labels,
                                                                                                                                              sigmas,
                                                                                                                                              delta_sigmas,
                                                                                                                                              model_args,
                                                                                                                                              fold=k,
                                                                                                                                              verbose=verbose)
        with open(f'{dir_destination}/final_test_preds_fold_{k}_{model_args}.pickle', 'wb') as handle:
            pk.dump(fold_test_preds, handle)
        with open(f'{dir_destination}/final_test_acc_fold_{k}_{model_args}.pickle', 'wb') as handle:
            pk.dump(fold_test_accs, handle)
        with open(f'{dir_destination}/final_test_loss_fold_{k}_{model_args}.pickle', 'wb') as handle:
            pk.dump(fold_test_losses, handle)
        with open(f'{dir_destination}/labels_test_fold_{k}_{model_args}.pickle', 'wb') as handle:
            pk.dump(fold_test_labels, handle)
        with open(f'{dir_destination}/ids_valid_fold_{k}_{model_args}.pickle', 'wb') as handle:
            pk.dump(fold_ids, handle)
        with open(f'{dir_destination}/final_valid_preds_fold_{k}_{model_args}.pickle', 'wb') as handle:
            pk.dump(fold_valid_preds, handle)
        with open(f'{dir_destination}/final_valid_labels_fold_{k}_{model_args}.pickle', 'wb') as handle:
            pk.dump(fold_valid_labels, handle)
        torch.save(model.state_dict(), os.path.join(dir_destination, f'trained_model_fold_{k}_{model_args}.pth'))