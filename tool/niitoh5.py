import gzip
import nibabel as nib
import numpy as np
import h5py
import os

nii_files = []
for dirpath, sf, files in os.walk('test'):
    for file in files:
        nii_files.append(os.path.join(dirpath,
                         file))

for i in nii_files:
    decompressed_file = gzip.open(i)
    out_path = i.replace('/','_')[:-3]
    with open('brain_nii/' + out_path, 'wb') as outfile:
        outfile.write(decompressed_file.read())

# load all files in the folder and save them as a .h5 file
def save_large_dataset(file_name, variable):
    h5f = h5py.File(os.path.join('brain_h5', file_name) + '.h5', 'w')
    h5f.create_dataset('variable', data=variable)
    h5f.close()


indir = 'brain_nii/'
Xs = []
for root, dirs, filenames in os.walk(indir):
    for f in filenames:
        if '.nii' == f[-4:]:
            img = nib.load(indir + f)
            data = img.dataobj  # Get the data object
            data = data[:-1, :-1, :-1]  # Clean the last dimension for a high GCD (all values are 0)
            X = np.expand_dims(data, -1)
            X = X / np.max(X)
            X = X.astype('float32')
            X = np.expand_dims(X, 0)
            print('Shape: ', X.shape)
            Xs.append(X)

    Xa = np.vstack(Xs)
    save_large_dataset('1', Xa)

# Finally, we can use the following code to load up the data:
# def load_large_dataset(file_name):
#     h5f = h5py.File(file_name + '.h5','r')
#     variable = h5f['variable'][:]
#     h5f.close()
#     return variable