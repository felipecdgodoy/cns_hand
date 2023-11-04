import pandas as pd
import numpy as np
import pickle as pk
import nibabel as nib

def get_CE_label(numeric_label):
    """ map single digit label (0, 1, 2, 3) -> 2-bit binary label (00, 01, 10, 11)"""
    mapping = {0: np.array([0, 0]), 1: np.array([1, 0]), 2: np.array([0, 1]), 3: np.array([1, 1])}
    return mapping[numeric_label]

with open('all_ucsf.pickle', 'rb') as f:
    ucsf = pk.load(f)
ucsf = pd.DataFrame(ucsf, columns=['filename', 'label', 'dataset', 'all_dataset', 'id', 'age', 'gender'])
ucsf['id'] = ucsf['id'].astype(int)
ucsf = ucsf.rename(columns={'age':'norm_age'})
ucsf['label'] = ucsf['label'].apply(lambda x : get_CE_label(x))

df = pd.read_csv('ucsf_npz_scores.csv')
df = df.drop(columns=['gender'])
df.loc[np.abs(df['npz']) > 3.5, 'npz'] = np.nan

matched = df.merge(ucsf, left_on='id', right_on='id')
matched = matched.drop_duplicates(subset=['id'])
matched = matched[['id', 'age', 'gender', 'npz', 'label', 'filename']]

images = list()
patch_x, patch_y, patch_z = 64, 64, 64
for filename in ucsf['filename']:
    img = nib.load(filename)
    img_data = img.get_fdata()
    img_data = img_data[0:patch_x, 0:patch_y, 0:patch_z]
    img_data = (img_data - np.mean(img_data)) / np.std(img_data)
    images.append(img_data)

matched['image'] = images

with open('matched_patients.pickle', 'wb') as handle:
    pk.dump(matched, handle)

