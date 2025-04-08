import os
import random
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision import transforms
from einops import rearrange
import hdf5



class GenderSplitDataset(Dataset):
    def __init__(self, root_dir, gender_and_identity_sorted_dict, master_mask_filepath, transform=None,
                 cache_dir=None, USING_TIC=0, transform_shape=None, normalize=False,
                 allready_registered=False):
        global lru_first_col, lru_last_col, lru_first_row, lru_last_row, lru_first_dim
        self.transform = transform
        self.root_dir = root_dir
        self.cache_dir = cache_dir
        self.transform_shape = transform_shape

        # Dictionary to hold the split data
        self.data = gender_and_identity_sorted_dict
        self.mean = 0.0
        self.std = 0.0
        self.normalize = normalize
        self.already_registered = allready_registered

        self.all_images = [(group, identifier, filename)
                           for group in self.data
                           for identifier in self.data[group]
                           for filename in self.data[group][identifier]]

        # self.all_images = self.all_images[:4]

        self.master_mask = self.transform(Image.open(master_mask_filepath).convert('F'))[:, :, ::5].float()
        # I HAVE ABSOLUTELY NO CLUE WHY THIS IS NECESSARY NOW
        self.master_mask = torch.flip(self.master_mask, [1])
        self.master_mask -= self.master_mask.min()
        self.master_mask /= self.master_mask.max()
        # find first and last non-zero column and row
        col_sums = (self.master_mask[0].sum(0) > 0).float().nonzero()
        row_sums = (self.master_mask[0].sum(1) > 0).float().nonzero()

        lru_first_col = self.first_col = col_sums[0]
        lru_last_col = self.last_col = col_sums[-1]
        lru_first_row = self.first_row = row_sums[0]
        lru_last_row = self.last_row = row_sums[-1]

        # when USING TIC SET TO 0 ELSE 29
        if USING_TIC == 0:
            lru_first_dim = self.first_dim = 0
        else:
            lru_first_dim = self.first_dim = 29

    def compute_mean_std(self):
        for _, _, image_filename in self.all_images:
            if '.pt' == image_filename[-3:]:
                image = self.load(self.load_tensor, image_filename)
            else:
                image = self.load(self.load_image, image_filename)
            image = image.type(torch.float32)
            self.mean += image.mean()
            self.std += image.std()
        self.mean /= len(self.all_images)
        self.std /= len(self.all_images)
        return self.mean, self.std

    def __len__(self):
        return len(self.all_images)

    def load_tensor(self, filepath):
        image = torch.load(filepath, weights_only=False)
        if self.already_registered:
            image = image[self.first_dim:, self.first_row:self.last_row, self.first_col:self.last_col]
            image = image.type(torch.float32)
            image *= self.master_mask
        _, h, w = self.master_mask.shape
        larger = torch.zeros((1, h, w))
        _, h, w = image.shape
        larger[:, :h, :w] = image
        image = larger.float()
        return image

    def load_image(self, img_path):
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        image = image[:, :, ::5]
        image *= self.master_mask
        if self.transform_shape:
            image = self.transform_shape(image)
        return image

    def load_tensor_h5(self, img_path):
        file_name = img_path.split('/')[-1]
        dir = img_path.split('/')[:-1]
        dir = '/'.join(dir)
        name = file_name.split('.')[0]
        image = hdf5.load_hdf5(name, dir, file_name)
        if self.already_registered:
            image = image.type(torch.float32)
            image *= self.master_mask
            image = image[self.first_dim:, self.first_row:self.last_row, self.first_col:self.last_col]
            # print(self.master_mask.shape)
        # _, h, w = self.master_mask.shape
        # larger = torch.zeros((1, h, w))
        # _, h, w = image.shape
        # larger[:, :h, :w] = image
        # image = larger.float()
        return image

    def __getitem__(self, idx):
        # Get the image path based on the index
        group, identifier, image_filename = self.all_images[idx]

        if '.pt' == image_filename[-3:]:
            image = self.load(self.load_tensor, image_filename)
        elif '.h5' == image_filename[-3:]:
            image = self.load(self.load_tensor_h5, image_filename)
        else:
            image = self.load(self.load_image, image_filename)

        if self.normalize:
            image = (image - self.mean) / self.std
        group = 1.0 if group == 'M' else 0.0
        # Return the image and its corresponding group and identifier
        return image, group, identifier, image_filename, [], idx

    def load(self, loading_function, filename):
        root_image_filepath = os.path.join(self.root_dir, filename)
        if '.h5' in filename:
            filename_cache = filename.replace('.h5', '.pt')
        else:
            filename_cache = filename
        cached_filepath = os.path.join(self.cache_dir, filename_cache)
        if self.cache_dir is not None:
            if not os.path.exists(self.cache_dir):
                os.makedirs(self.cache_dir, exist_ok=True)
            if os.path.exists(cached_filepath):
                image = torch.load(cached_filepath, weights_only=False)
            else:
                image = loading_function(root_image_filepath)
                torch.save(image, cached_filepath)
        else:
            image = loading_function(root_image_filepath)

        return image

    def num_identities(self):
        return sum(len(self.data[group]) for group in self.data)


class BalancedSampler(Sampler):
    def __init__(self, dataset, sampler_size=None):
        super().__init__()
        self.dataset = dataset
        self.samples = {'F': {}, 'M': {}}
        self.sampler_size = sampler_size

        # Organize dataset indices by group and identifier
        for idx in range(len(dataset)):
            group, identifier, _ = dataset.all_images[idx]
            if identifier not in self.samples[group]:
                self.samples[group][identifier] = []
            self.samples[group][identifier].append(idx)

        # Create a list of balanced indices, sampling one per identifier
        self.balanced_indices = []

    def __len__(self):
        return len(self.balanced_indices)

    def __iter__(self):
        self.balanced_indices = []
        self.min_samples = min(len(self.samples['F']), len(self.samples['M']))
        self.f_ids = random.sample(list(self.samples['F'].keys()), self.min_samples)
        self.m_ids = random.sample(list(self.samples['M'].keys()), self.min_samples)
        for f_id, m_id in zip(self.f_ids, self.m_ids):
            if self.sampler_size is not None:
                if self.sampler_size > len(self.samples['F'][f_id]):
                    self.balanced_indices.extend(random.sample(self.samples['F'][f_id], len(self.samples['F'][f_id])))
                else:
                    self.balanced_indices.extend(random.sample(self.samples['F'][f_id], self.sampler_size))
                if self.sampler_size > len(self.samples['M'][m_id]):
                    self.balanced_indices.extend(random.sample(self.samples['M'][m_id], len(self.samples['M'][m_id])))
                else:
                    self.balanced_indices.extend(random.sample(self.samples['M'][m_id], self.sampler_size))
            else:
                self.balanced_indices.extend(self.samples['F'][f_id])
                self.balanced_indices.extend(self.samples['M'][m_id])

        return iter(self.balanced_indices)

    def num_identities(self):
        return len(self.f_ids) + len(self.m_ids)
