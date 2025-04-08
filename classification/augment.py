import torch
import torchvision.transforms.functional as TF
import random
import imgaug.augmenters as iaa
from einops import rearrange


def shift_by_random_pixels(tensors, max_shift=5):
    batch_size, channels, height, width = tensors.shape
    transformed_tensors = []
    for i in range(batch_size):
        shift_x = random.randint(-max_shift, max_shift)
        shift_y = random.randint(-max_shift, max_shift)
        transformed_tensors.append(TF.affine(tensors[i], angle=0, translate=(shift_x, shift_y), scale=1, shear=0))
    return torch.stack(transformed_tensors)


def do_augmentations(tensors):
    original_device = tensors.device
    elastic_transform = iaa.ElasticTransformation(alpha=(0, 5.0), sigma=0.25)
    # elastic_transform = iaa.ElasticTransformation(alpha=(0, 25), sigma=1.25)
    batch_size, channels, height, width = tensors.shape
    tensors = tensors.cpu().numpy()
    tensors = rearrange(tensors, 'b c h w -> b h w c')
    augmented_tensors = elastic_transform(images=tensors)
    augmented_tensors = rearrange(augmented_tensors, 'b h w c -> b c h w')
    augmented_tensors = torch.from_numpy(augmented_tensors)
    augmented_tensors = augmented_tensors.to(original_device)
    return augmented_tensors


def do_piecewise_affine(tensors):
    original_device = tensors.device
    piecewise_affine = iaa.PiecewiseAffine(scale=(0.01, 0.05))
    batch_size, channels, height, width = tensors.shape
    tensors = tensors.cpu().numpy()
    tensors = rearrange(tensors, 'b c h w -> b h w c')
    augmented_tensors = piecewise_affine(images=tensors)
    augmented_tensors = rearrange(augmented_tensors, 'b h w c -> b c h w')
    augmented_tensors = torch.from_numpy(augmented_tensors)
    augmented_tensors = augmented_tensors.to(original_device)
    return augmented_tensors


def add_gaussian_noise(tensors, mean=0., std=1.):
    noise = torch.randn_like(tensors) * std + mean
    return tensors + noise