import random
import pandas as pd
import numpy as np
import pickle as pk
import nibabel as nib

# set deterministic behavior
np.random.seed(0)
random.seed(0)

import warnings
warnings.filterwarnings('ignore')

def get_CE_label(numeric_label):
    """ map single digit label (0, 1, 2, 3) -> 2-bit binary label (00, 01, 10, 11)"""
    mapping = {0: np.array([0, 0]), 1: np.array([1, 0]), 2: np.array([0, 1]), 3: np.array([1, 1])}
    return mapping[numeric_label]

with open('new_ucsf.pickle', 'rb') as f:
    ucsf = pk.load(f)

ucsf = pd.DataFrame(ucsf, columns=['filename', 'label', 'dataset', 'all_dataset', 'id', 'age', 'gender', 'npz'])
ucsf['id'] = ucsf['id'].astype(int)
ucsf.loc[np.abs(ucsf['npz']) > 3.5, 'npz'] = np.nan
ucsf['label'] = ucsf['label'].apply(lambda x : get_CE_label(x))
ucsf = ucsf.drop_duplicates(subset=['id'])
ucsf = ucsf[['id', 'age', 'gender', 'npz', 'label', 'filename']]

patch_x, patch_y, patch_z = 64, 64, 64

images = list()

for filename in ucsf['filename']:
    img = nib.load(filename)
    img_data = img.get_fdata()
    img_data = img_data[0:patch_x, 0:patch_y, 0:patch_z]
    img_data = (img_data - np.mean(img_data)) / np.std(img_data)
    images.append(img_data)

ucsf['image'] = images

augm_hiv_block_1 = ucsf[ucsf['label'].astype(str) == '[0 1]']
augm_hiv_block_2 = ucsf[ucsf['label'].astype(str) == '[0 1]']
augm_hiv_block_3 = ucsf[ucsf['label'].astype(str) == '[0 1]']
augm_hiv_block_4 = ucsf[ucsf['label'].astype(str) == '[0 1]']

augm_hiv_block_1['image'] = augm_hiv_block_1['image'].apply(lambda img : img + np.random.normal(loc=0, scale=0.02, size=(patch_x, patch_y, patch_z)))
augm_hiv_block_2['image'] = augm_hiv_block_2['image'].apply(lambda img : img + np.random.normal(loc=0, scale=0.05, size=(patch_x, patch_y, patch_z)))
augm_hiv_block_3['image'] = augm_hiv_block_3['image'].apply(lambda img : img + np.random.normal(loc=0, scale=0.10, size=(patch_x, patch_y, patch_z)))
augm_hiv_block_4['image'] = augm_hiv_block_3['image'].apply(lambda img : img + np.random.normal(loc=0, scale=0.12, size=(patch_x, patch_y, patch_z)))

ucsf = pd.concat((augm_hiv_block_1, augm_hiv_block_2, augm_hiv_block_3, augm_hiv_block_4, ucsf))

print(f'Matched Patients: {len(images)}')
print(f'Generated Samples: {len(ucsf)}')

print(f"CTRL: {len(ucsf[ucsf['label'].astype(str) == '[0 0]'])}")
print(f"MCI: {len(ucsf[ucsf['label'].astype(str) == '[1 0]'])}")
print(f"HIV: {len(ucsf[ucsf['label'].astype(str) == '[0 1]'])}")
print(f"HAND: {len(ucsf[ucsf['label'].astype(str) == '[1 1]'])}")

with open('matched_patients.pickle', 'wb') as handle:
    pk.dump(ucsf, handle)
