import random
import pandas as pd
import numpy as np
import pickle as pk
import nibabel as nib
from tqdm import tqdm

# set deterministic behavior
np.random.seed(0)
random.seed(0)

COMMON_SIZE = 96 # dim for image 3D-volume (DxDxD)
SAVE_PATH = '/scratch/groups/kpohl/matched_patients.pickle'

def get_CE_label(numeric_label):
    """ map single digit label (0, 1, 2, 3) -> 2-bit binary label (00, 01, 10, 11)"""
    mapping = {0: np.array([0, 0]), 1: np.array([1, 0]), 2: np.array([0, 1]), 3: np.array([1, 1])}
    return mapping[numeric_label]

with open('new_ucsf.pickle', 'rb') as f:
    ucsf = pk.load(f)

ucsf = pd.DataFrame(ucsf, columns=['filename', 'label', 'dataset', 'all_dataset', 'id', 'age', 'gender', 'npz'])
### Update to high-definition images
ucsf['filename'] = ucsf['filename'].apply(lambda path : path.replace('img_64_', 'img_orig_'))
###
ucsf['id'] = ucsf['id'].astype(int)
ucsf.loc[np.abs(ucsf['npz']) > 3.5, 'npz'] = np.nan
ucsf['label'] = ucsf['label'].apply(lambda x : get_CE_label(x))
ucsf = ucsf.drop_duplicates(subset=['id'])
ucsf = ucsf[['id', 'age', 'gender', 'npz', 'label', 'filename']]

paths = ucsf['filename']
n = len(ucsf)
x, y, z = nib.load(paths[0]).get_fdata().shape
ucsf['image'] = [np.zeros((x, y, z))] * n

patch_x = (x - COMMON_SIZE) // 2
patch_y = (y - COMMON_SIZE) // 2
patch_z = (z - COMMON_SIZE) // 2

for i in tqdm(range(len(paths)), desc='processing images'):
    filename = paths[i]
    img = nib.load(filename)
    img_data = img.get_fdata() # original size: 138 x 176 x 138
    img_data = img_data[patch_x : -patch_x, patch_y : -patch_y, patch_z : -patch_z]
    img_data = (img_data - np.mean(img_data)) / np.std(img_data)
    ucsf.loc[ucsf['filename'] == filename, 'image'] = [img_data]

print(f'Matched Patients: {len(ucsf)}')

with open(SAVE_PATH, 'wb') as handle:
    pk.dump(ucsf, handle)

print(f"CTRL: {len(ucsf[ucsf['label'].astype(str) == '[0 0]'])}")
print(f"MCI: {len(ucsf[ucsf['label'].astype(str) == '[1 0]'])}")
print(f"HIV: {len(ucsf[ucsf['label'].astype(str) == '[0 1]'])}")
print(f"HAND: {len(ucsf[ucsf['label'].astype(str) == '[1 1]'])}")

print(f"Successfully Saved: {SAVE_PATH}")
