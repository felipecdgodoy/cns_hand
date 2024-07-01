import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from model import fe
from dataloading import MRI_Dataset
import os
from torch.utils.data import DataLoader
import nibabel as nb
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff
from skimage.transform import resize

template = nb.load('/home/groups/kpohl/t1_data/hand/template.nii.gz')
template_img = np.resize(template.get_fdata(),(64,64,64))
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("device:",device)
num_classes = 2
target_dimensions = (64, 64, 64)
# Use scikit-image resize to resize the 3D image
template_img = resize(template_img, target_dimensions, mode='constant')

class net(nn.Module):
    def __init__(self,fe_path,class_path):
        super(net, self).__init__()

        self.feature_extractor = fe(trainset_size = 104, in_num_ch=1, img_size=(64, 64, 64), inter_num_ch=16,
                          fc_num_ch=16, kernel_size=3, conv_act='LeakyReLU',
                          fe_arch= 'fe1', dropout=0.2,
                          fc_dropout = 0.2, batch_size = 1).to(device)

        self.classifier = nn.Sequential(
                           nn.Linear(256, 128),
                           nn.Dropout(0.25),
                           nn.LeakyReLU(),
                           nn.Linear(128, 16),
                           nn.LeakyReLU(),
                           nn.Linear(16, 2),
                           ).to(device)
        
        
        self.feature_extractor.load_state_dict(torch.load(fe_path))
        self.classifier.load_state_dict(torch.load(class_path))

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x


def generate_saliency_map_3d_multilabel(model, input_data, num_classes):
    model.eval()

    input_data.requires_grad_()
    output = model(input_data)

    model.zero_grad()

    saliency_maps = []

    for target_label in range(num_classes):
        # Compute gradients with respect to input for each target class
        output[:, target_label].backward(retain_graph=True)

        # Get the gradients and normalize them
        gradients = input_data.grad.data
        gradients = gradients.abs().max(dim=1, keepdim=True)[0]

        # Reshape gradients to match input_data shape
        gradients = gradients.view(gradients.size(0), -1)

        # Normalize the gradients to [0, 1]
        gradients = (gradients - gradients.min()) / (gradients.max() - gradients.min() + 1e-10)

        # Convert the gradients to a numpy array
        gradients = gradients.cpu().numpy()

        saliency_maps.append(gradients.reshape((64,64,64)))

        # Reset gradients for the next target class
        model.zero_grad()

    return saliency_maps


# Instantiate the 3D model

# folder = '/scratch/users/jmanasse/lamb=0.1'   
# subfolders = ['floral_snowball_25', 'tough_wildflower_26', 'vital_pyramid_28','sandy_sky_29','expert_leaf_30']
folder = '/scratch/users/jmanasse'
subfolder = 'mri_ckpts'
# models =  'ublah_fearch_npz1_bz_32_epoch_100_lr_5e-05_do_0.2_fcdo_0.2_wd_0.5_alpha_[0.5, 0.5, 0.5, 0.5]_seed_1'
models = 'ublah_fearch_npz1_bz_16_epoch_100_lr_5e-05_do_0.2_fcdo_0.2_wd_0.5_alpha_[0.5, 0.5, 0.5, 0.5]_seed_1'
folds = ['fold_0','fold_1','fold_2','fold_3','fold_4']
fe_model = 'feature_extractor.pt'
class_model = 'classifier_ucsf.pt'

fe_paths = [os.path.join(folder,subfolder,models,fold,fe_model) for fold in folds]
class_paths = [os.path.join(folder,subfolder,models,fold,class_model) for fold in folds]

networks = [net(fe_path, class_path) for (fe_path, class_path) in zip(fe_paths, class_paths)]

average_saliency_map = [np.zeros((64,64,64))] * num_classes
#num_hand_samples = 0
num_samples = 0
for i,model in enumerate(networks):

    test_data = MRI_Dataset(fold = i, stage= 'original_test')

    final_test_loader = DataLoader(dataset=test_data,
                                batch_size=1, #64,
                                shuffle=False,
                                pin_memory=True,
                                num_workers=3)
    for (images, labels, actual_labels, ids, ages, genders,npzs)  in final_test_loader:
        num_samples += 1
        images = images.view(1,1,64,64,64).to(device).float()
        model.eval()
        output = model(images)
        # if output[:,0] < 0 or output[:,1] < 0:
        #     continue
        #num_hand_samples += 1

        # if output[0] > 0 and output[1] > 0: #if classified as HAND
        #     average_saliency_map_hand[0] = average_saliency_map_hand[0] + saliency_maps_multilabel[0]
        #     average_saliency_map_hand[1] = average_saliency_map_hand[1] + saliency_maps_multilabel[1]
        #     num_hand_samples += 1
        # elif output[0] > 0 and output[1] < 0: #if classified as CI
        # #     average_saliency_map_ci = average_saliency_map_ci + saliency_maps_multilabel[0]
        #     wrong_ci_samples += 1
        # elif output[0] < 0.5 and output[1] > 0: #if classified as HIV
        # #     average_saliency_map_hiv = average_saliency_map_hiv + saliency_maps_multilabel[1]
        #     wrong_hiv_samples += 1
        # elif output[0] < 0.5 and output[1] < 0.5: #if classified as CTRL
        #     wrong_ctrl_samples += 1

        # else:
        #     continue
        
        # Generate the saliency maps for each class in the multi-label setting
        saliency_maps_multilabel = generate_saliency_map_3d_multilabel(model, images, num_classes)

        # Update average maps
        average_saliency_map[0] = average_saliency_map[0] + saliency_maps_multilabel[0]
        average_saliency_map[1] = average_saliency_map[1] + saliency_maps_multilabel[1]
        #UNCOMMENT FOR HEATMAP
        # saliency_cmap = plt.get_cmap('viridis')
        # #generate HAND map = (s_hiv + s_ci)/2
        # saliency_hand = (saliency_maps_multilabel[0]+saliency_maps_multilabel[1])/2
        # #generate HIV map = max(s_hiv - s_ci,0)
        # saliency_hiv= np.maximum(saliency_maps_multilabel[1]-saliency_maps_multilabel[0], 0)
        # #generate MCI map = max(s_ci - s_hiv,0)
        # saliency_ci = np.maximum(saliency_maps_multilabel[0]-saliency_maps_multilabel[1], 0)

    #Normalize maps 
    # average_saliency_map[0] = average_saliency_map[0]/num_hand_samples
    # average_saliency_map[1] = average_saliency_map[1]/num_hand_samples
average_saliency_map[0] = average_saliency_map[0]/num_samples
average_saliency_map[1] = average_saliency_map[1]/num_samples
# Output directory to save saliency maps
output_directory = 'saliency_maps'
os.makedirs(output_directory, exist_ok=True)

# Save the saliency maps for each class
for i in range(num_classes):
    #UNCOMMENT FOR HEATMAP
    # # Normalize saliency map values to [0, 1]
    # normalized_saliency = saliency_maps_multilabel[i] / np.max(saliency_maps_multilabel[i])
    # # Map normalized saliency values to RGBA colors using the chosen colormap
    # saliency_colors = saliency_cmap(normalized_saliency)
    #  # Create the overlay by blending the template image and the saliency colors
    # overlay = (1 - normalized_saliency[..., None]) * template_img + normalized_saliency[..., None] * saliency_colors

    # # Save the overlay as an image (you can choose a different format)
    # overlay_path = os.path.join(output_directory, f'overlay_class_{i}.tif')
    # tiff.imsave(overlay_path, overlay)

    #AVERAGE IMAGE
    print("Shape of Saliency Map:", average_saliency_map[i].shape)
    print("Data Type of Saliency Map:", average_saliency_map[i].dtype)
    saliency_map_path = os.path.join(output_directory, 'saliency_map_class_{}.tif'.format(i))
    tiff.imsave(saliency_map_path, average_saliency_map[i].squeeze().astype(np.float32))

    #SINGLE IMAGE
    # print("Shape of Saliency Map:", saliency_maps_multilabel[i].shape)
    # print("Data Type of Saliency Map:", saliency_maps_multilabel[i].dtype)
    # saliency_map_path = os.path.join(output_directory, 'saliency_map_class_{}.tif'.format(i))
    # tiff.imsave(saliency_map_path, saliency_maps_multilabel[i].squeeze().astype(np.float32))

    print('Saliency maps saved to:', output_directory)



