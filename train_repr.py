import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
import pickle as pk

from torch.utils.data import DataLoader

from tqdm import tqdm

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='ADNI')
parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 100)')
parser.add_argument('--batch_size', type=int, default=16, help='input batch size for training (default: 16)')
parser.add_argument('--lamb', type=float, default=1e-2, help='disentanglement factor (default: 0.5)')
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
torch.backends.cudnn.enabled = False # DO NOT REMOVE!

# DEBUG
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
        image_pairs = torch.tensor([(images[2*i], images[2*i + 1]) for i in range(num_pairs)])
        label_pairs = torch.tensor([(labels[2*i], labels[2*i + 1]) for i in range(num_pairs)]).double()
        sigma_pairs = torch.tensor([(sigmas[2*i], sigmas[2*i + 1]) for i in range(num_pairs)])
        delta_sigma_pairs = torch.tensor([sigmas[2*i] - sigmas[2*i + 1] for i in range(num_pairs)])
        id_pairs = torch.tensor([(ids[2*i], ids[2*i + 1]) for i in range(num_pairs)])
        print(f'image pair shape: {image_pairs.shape}')
        print(f'label pair shape: {label_pairs.shape}')
        return image_pairs, label_pairs, sigma_pairs, delta_sigma_pairs, id_pairs

class Net(nn.Module):

    def __init__(self, image_dim=64, tau_dim=512):
        """ image_dim (int): value 'D' such that all images are a 3D-volume DxDxD"""
        """ tau_dim (int): number of dimensions to encode the tau-direction in"""
        super(Net, self).__init__()
        self.device = device
        self.dim = image_dim
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
        e = self.encode(x) ## Encode to latent-space
        y = self.linear(e) ## Classification
        return e, y

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
        bce = nn.BCELoss()
        pred_1 = pred_1.nan_to_num(0.5).double()
        pred_2 = pred_2.nan_to_num(0.5).double()
        bce_loss_1 = bce(pred_1, labels_1)
        bce_loss_2 = bce(pred_2, labels_2)
        bce_loss = (bce_loss_1 + bce_loss_2)
        if lamb == 0:
            return bce_loss
        ones = torch.ones((1, 1, self.dim, self.dim, self.dim), device=self.device).double()
        tau = self.encode(ones).flatten()
        proj_e1_len = (emb_t1 @ tau) / tau.norm(p=1)
        proj_e2_len = (emb_t2 @ tau) / tau.norm(p=1)
        proj_len_diff = proj_e2_len - proj_e1_len
        disentangle_loss = (proj_len_diff - delta_sigma).nan_to_num(0).square().clamp(max=100).sum()
        return bce_loss + lamb*disentangle_loss

    def update_tau(self, eps=1e-4):
        ones = torch.ones((1, 1, self.dim, self.dim, self.dim), device=self.device).double()
        tau = self.encode(ones).flatten()
        tau[tau > 0] += eps
        tau[tau < 0] -= eps
        return tau

    def count_correct_batch(self, batch_preds, batch_labels):
        corr = (batch_preds.round() == batch_labels).all(dim=1).sum()
        return int(corr.data)

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
        opt = optim.SGD(self.parameters(), lr=alpha)
        fold_preds = dict()
        for e in tqdm(range(epochs), desc=f'Fold {fold + 1}'):
            total_loss = 0
            imgs_seen = 0
            correct = 0
            for i, batch in enumerate(train_dl):
                eff_batch_size = min(batch_size, batch.shape[0])
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
                correct += self.count_correct_batch(pred1, l1)
                correct += self.count_correct_batch(pred2, l2)
                opt.zero_grad()
                loss1 = self.pair_loss(emb_t1=emb1, emb_t2=emb2, delta_sigma=ds, pred_1=pred1, pred_2=pred2,
                                        labels_1=l1, labels_2=l2, lamb=lamb)
                loss1.backward()
                opt.step()
                total_loss = total_loss + float(loss1.data)
            p = nn.utils.parameters_to_vector(self.parameters())
            train_loss_vals.append(total_loss / imgs_seen)
            train_acc.append(correct / imgs_seen)
            ## TESTING PORTION
            fold_preds[e] = []
            with torch.no_grad():
                total_loss = 0
                imgs_seen = 0
                correct = 0
                for i, batch in enumerate(test_dl):
                    eff_batch_size = min(batch_size, batch.shape[0])
                    imgs_seen += 2 * eff_batch_size
                    img1 = batch[:, 0, :, :, :].unsqueeze(1).to(self.device)
                    img2 = batch[:, 1, :, :, :].unsqueeze(1).to(self.device)
                    img_input = torch.cat((img1, img2))
                    embeddings, class_preds = self.forward(img_input)
                    pred1, pred2 = class_preds[:eff_batch_size], class_preds[eff_batch_size:]
                    fold_preds[e].append(pred1)
                    fold_preds[e].append(pred2)
                    emb1, emb2 = embeddings[:eff_batch_size], embeddings[eff_batch_size:]
                    ds = test_del_sigmas[i*eff_batch_size : (i+1)*eff_batch_size].to(self.device)
                    l1 = test_labels[i*eff_batch_size : (i+1)*eff_batch_size, 0, :].to(self.device)
                    l2 = test_labels[i*eff_batch_size : (i+1)*eff_batch_size, 1, :].to(self.device)
                    correct += self.count_correct_batch(pred1, l1)
                    correct += self.count_correct_batch(pred2, l2)
                    loss = self.pair_loss(emb_t1=emb1, emb_t2=emb2, delta_sigma=ds, pred_1=pred1, pred_2=pred2,
                                          labels_1=l1, labels_2=l2, lamb=lamb)
                    total_loss += float(loss.data)
                test_loss_vals.append(total_loss / imgs_seen)
                test_acc.append(correct / imgs_seen)
            if e % 1 == 0: # adjust as needed for verbosity
                print(f'    Epoch {e+1} Results -> avg train loss = {round(train_loss_vals[-1], 4)} | avg test loss = {round(test_loss_vals[-1], 4)} | train acc = {round(100*train_acc[-1], 2)}% | test acc = {round(100*test_acc[-1], 2)}%')
                if len(train_acc) >= 2:
                    train_acc_delta = train_acc[-1] - train_acc[-2]
                    test_acc_delta = test_acc[-1] - test_acc[-2]
                    s1 = '+' if train_acc_delta > 0 else ''
                    s2 = '+' if test_acc_delta > 0 else ''
                    print(f'    Train Acc Delta: {s1}{round(100*train_acc_delta, 2)}% \n     Test Acc Delta: {s2}{round(100*test_acc_delta, 2)}%')
                    train_loss_delta = train_loss_vals[-1] - train_loss_vals[-2]
                    test_loss_delta = test_loss_vals[-1] - test_loss_vals[-2]
                    s1 = '+' if train_loss_delta > 0 else ''
                    s2 = '+' if test_loss_delta > 0 else ''            
                    print(f'    Train Loss Delta: {s1}{round(train_loss_delta, 6)} \n     Test Loss Delta: {s2}{round(test_loss_delta, 6)}')
        best_index = np.argmax(test_acc)
        print(f'[DONE] FOLD {fold + 1} --- Best Epoch Results -> TRAIN ACC = {round(100*train_acc[best_index], 2)}% | TEST ACC = {round(100*test_acc[best_index], 2)}%\n')
        return fold_preds[best_index], test_acc, test_loss_vals

if __name__ == '__main__':
    image_dim = 64
    tau_dim = 2048
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
    print(f'Model Initialized: image_dim = {image_dim}x{image_dim}x{image_dim}; tau_dim = {tau_dim}')
    print(f'--> Starting Training')
    for k in range(num_folds):
        model = Net(image_dim=image_dim, tau_dim=tau_dim).double().cuda()
        fold_preds, fold_accs, fold_losses = model.train_fold(image_pairs, label_pairs, sigma_pairs,
                                                              delta_sigma_pairs, model_args, fold=k)
        with open(f'final_preds_fold_{k}_{model_args}.pickle', 'wb') as handle:
            pk.dump(fold_preds, handle)
        with open(f'final_acc_fold_{k}_{model_args}.pickle', 'wb') as handle:
            pk.dump(fold_accs, handle)
        with open(f'final_loss_fold_{k}_{model_args}.pickle', 'wb') as handle:
            pk.dump(fold_losses, handle)
        with open(f'ids_pidn_fold_{k}_{model_args}.pickle', 'wb') as handle:
            idx = np.arange(len(image_pairs))
            fold_mask = (idx % num_folds == k)
            pk.dump(id_pairs[fold_mask], handle)
        torch.save(model.state_dict(), os.path.join('', f'trained_model_fold_{k}_{model_args}.pth'))
