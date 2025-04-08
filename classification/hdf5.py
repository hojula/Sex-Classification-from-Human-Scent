
import os
import h5py
from time import sleep
import torch
import numpy as np
from tqdm import tqdm, trange

def save_hdf5(tensor, tensor_name, hdf5_dir,file_name):
    hdf5_filepath = os.path.join(hdf5_dir, file_name)
    group = tensor_name[0]

    if group in ['F', 'M']:
        identifier = ''
        for char in tensor_name[1:]:
            if char.isdigit():
                identifier += char
            else:
                break

        if tensor.is_cuda:
            tensor = tensor.cpu().numpy()
        else:
            tensor = tensor.numpy()

        chunk_size = (tensor.shape[0], 10, 10)

        success = False
        while not success:
            try:
                with h5py.File(hdf5_filepath, 'a') as f:
                    # Check if group exists, otherwise create it
                    if group not in f:
                        f.create_group(group)
                    if identifier not in f[group]:
                        f[group].create_group(identifier)
                    # Check if dataset exists, otherwise create it
                    if tensor_name not in f[group][identifier]:
                        dset = f[group][identifier].create_dataset(
                            name=tensor_name,
                            data=tensor,
                            chunks=chunk_size
                        )
                success = True
            except Exception as e:
                print(f"HDF5 Error while processing {tensor_name}: {e}")
                sleep(1)
    #print(f"Dataset {tensor_name} saved to {hdf5_filepath}")

def load_hdf5(tensor_name, hdf5_dir, file_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hdf5_filepath = os.path.join(hdf5_dir, file_name)
    group = tensor_name[0]

    if group in ['F', 'M']:
        identifier = ''
        for char in tensor_name[1:]:
            if char.isdigit():
                identifier += char
            else:
                break

        success = False
        while not success:
            try:
                with h5py.File(hdf5_filepath, 'r') as f:
                    # Check if group and identifier exist
                    if group in f and identifier in f[group] and tensor_name in f[group][identifier]:
                        dset = f[group][identifier][tensor_name]
                        tensor_data = dset[:]
                        # Convert numpy array to torch tensor
                        tensor = torch.tensor(tensor_data, device=device)
                        success = True
                        return tensor
                    else:
                        raise KeyError(f"Dataset {tensor_name} not found in file.")
            except Exception as e:
                print(f"HDF5 Error while loading {tensor_name}: {e}")
                sleep(1)


if __name__ == '__main__':
    dir = '/mnt/data/vrg/spetlrad/data/scent/dataset/warped/fcn_plus_cos_all_tic'
    files = os.listdir(dir)
    for i in trange(len(files)):
        file = files[i]
        tensor = torch.load(os.path.join(dir, file))
        save_hdf5(tensor, file.replace('.pt', ''),dir, file.replace('.pt', '.h5'))
        #remove pt
        os.remove(os.path.join(dir, file))
        #tensor = load_hdf5(file, '/mnt/data/vrg/spetlrad/data/scent/dataset/warped/test', 'test.h5')