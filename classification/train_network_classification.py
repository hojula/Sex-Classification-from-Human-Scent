import importlib
import os

from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
import torch.optim as optim
from torchvision.transforms import transforms
from tqdm import tqdm, trange
from omegaconf import OmegaConf
from models import LinearProjectionModel1, LeNet5
from statistics import Statistics
import torch as th

import argparse

from augment import shift_by_random_pixels
from help_functions import create_kfold_splits
from genderSplitDataset import GenderSplitDataset
from help_functions import plot_loss_and_error
from help_functions import print_statistics
from genderSplitDataset import BalancedSampler


def dev(device=None):
    """
    Get the device to use for torch.distributed.
    """
    if device is None:
        if th.cuda.is_available():
            return th.device(int(os.environ['LOCAL_RANK']))
        return th.device("cpu")
    return th.device(device)


def parse_args():
    parser = argparse.ArgumentParser(description="Training job configuration")

    # Add arguments
    parser.add_argument('conf_path', type=str, help='Path to the configuration file')
    parser.add_argument('job_id', type=str, help='Job ID')
    parser.add_argument('--clear_cache', action='store_true', help='Clear cache directory')
    # Parse the arguments
    args = parser.parse_args()
    print(args)

    return args


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def calculate_loss(criterion, model, logits, labels, l1_lambda=0.1, l2_lambda=0.1, l1_bool=False, l2_bool=False):
    if l1_bool:
        l1_norm = sum(p.abs().sum() for p in model.parameters())
    if l2_bool:
        l2_norm = sum(p.pow(2).sum() for p in model.parameters())
    if l1_bool and l2_bool:
        loss = criterion(logits, labels) + l1_lambda * l1_norm + l2_lambda * l2_norm
    elif l1_bool:
        loss = criterion(logits, labels) + l1_lambda * l1_norm
    elif l2_bool:
        loss = criterion(logits, labels) + l2_lambda * l2_norm
    else:
        loss = criterion(logits, labels)
    return loss


def validate_model_one_epoch(model, criterion, val_loader, val_dataset, split, validation_loss, validation_error,
                             currentStatistics, epoch):
    model.eval()
    val_loss = 0.0
    val_err = 0.0
    val_total_images = 0
    device = dev()
    with torch.no_grad():
        for images, labels, ids, file_name, grid, _ in val_loader:
            val_total_images += len(images)
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = calculate_loss(criterion, model, logits, labels)
            # for each image in images it counts how many samples are from the same identity
            identity_counts = torch.tensor(
                [len(val_dataset.data['M' if group == 1.0 else 'F'][id.item()]) for group, id in
                 zip(labels, ids)]
            ).to(device)

            # sums errors for each image -> could be more than 1
            val_err += ((torch.round(torch.sigmoid(logits)) != labels) / identity_counts).sum().item()
            val_loss += (loss / identity_counts).sum().item()
            true_labels = labels.cpu().numpy()
            predicted_labels = torch.round(torch.sigmoid(logits)).cpu().numpy()
            group = ['M' if group == 1.0 else 'F' for group in labels]
            group_and_identity = [group + str(id.item()) for group, id in zip(group, ids)]
            currentStatistics.count_statistic(predicted_labels, true_labels, group_and_identity)

    # divide by number of identities in the validation set
    val_loss /= val_dataset.num_identities()
    validation_loss[split][epoch] = val_loss
    val_err /= val_dataset.num_identities()
    validation_error[split][epoch] = val_err
    currentStatistics.validation_loss = val_loss
    currentStatistics.validation_error = val_err

    return val_loss, val_err, 0, 0


def train_model_one_epoch(config, model, criterion, optimizer, trn_loader, trn_dataset, split, training_loss,
                          training_error, currentStatistics, sampler_size=None, sampler=None):
    currentStatistics.init_new_epoch()
    device = dev()
    model.train()
    running_loss = 0.0
    trn_loss = 0.0
    trn_err = 0.0
    id_correct_incorrect = {}
    total_images = 0
    for images, labels, ids, file_name, grid, idx in trn_loader:
        total_images += len(images)
        images, labels = images.to(device), labels.to(device)
        if config.affine_shift_max_px > 0:
            # images = add_gaussian_noise(images)
            images = shift_by_random_pixels(images, config.affine_shift_max_px)
            # images = do_augmentations(images)
            # images = do_piecewise_affine(images)

        optimizer.zero_grad()
        logits = model(images)
        loss = calculate_loss(criterion, model, logits, labels)
        predictions = torch.round(torch.sigmoid(logits))
        for label, id, prediction in zip(labels, ids, predictions):
            group = 'M' if label == 1.0 else 'F'
            key = group + str(id.item())
            if key not in id_correct_incorrect:
                id_correct_incorrect[key] = {"Correct": 0, "Incorrect": 0}
            if label == prediction:
                id_correct_incorrect[key]["Correct"] += 1
            else:
                id_correct_incorrect[key]["Incorrect"] += 1
        if sampler_size is None:
            identity_counts = torch.tensor(
                [len(trn_dataset.data['M' if group == 1.0 else 'F'][id.item()]) for group, id in
                 zip(labels, ids)]
            ).to(device)
        else:
            identity_counts = torch.tensor([sampler_size for _ in range(len(labels))]).to(device)
        trn_err += ((torch.round(torch.sigmoid(logits)) != labels) / identity_counts).sum().item()
        trn_loss += (loss / identity_counts).sum().item()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    trn_loss /= sampler.num_identities()
    training_loss[split].append(trn_loss)
    trn_err /= sampler.num_identities()
    training_error[split].append(trn_err)
    return trn_loss, trn_err

def do_setup():
    device = dev()
    args = parse_args()
    config = OmegaConf.load(args.conf_path)
    # clear cache
    if args.clear_cache and os.path.exists(config['cache_dir']):
        for file in os.listdir(config['cache_dir']):
            os.remove(os.path.join(config['cache_dir'], file))
    BATCH_SIZE = config.batch_size
    model_save_dir = os.path.join(config['save_dir'], args.job_id)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    root_dir = config.data_dir
    master_mask_filepath = config.master_mask_filepath
    transform_img_to_tensor = transforms.Compose([transforms.ToTensor()])
    transform_tensor_shape = transforms.Compose([transforms.Resize((224, 224))])
    # create splits probably only in rank 0
    subsample = config.subsample
    if subsample==-1:
        subsample = None
    splits = create_kfold_splits(root_dir, subsample=subsample, n_splits=config.num_folds)
    num_splits = len(splits)
    errors = [float('inf')] * num_splits
    errors_train = [float('inf')] * num_splits
    best_models = [None] * num_splits
    statistics_array = [Statistics() for _ in range(num_splits)]
    # some parameters
    num_epochs = config.training_params.epochs
    plot_after_num_epoch = config.training_params.plot_after
    save_after_num_epoch = config.training_params.save_after
    validate_after_num_epoch = config.training_params.valid_after
    training_loss = [[] for _ in range(num_splits)]
    training_error = [[] for _ in range(num_splits)]
    validation_loss = [{} for _ in range(num_splits)]
    validation_error = [{} for _ in range(num_splits)]
    sampler_size = config.training_params.sampler_size
    return BATCH_SIZE, best_models, config, device, errors, errors_train, master_mask_filepath, model_save_dir, num_epochs, num_splits, plot_after_num_epoch, root_dir, sampler_size, save_after_num_epoch, splits, statistics_array, training_error, training_loss, transform_img_to_tensor, transform_tensor_shape, validate_after_num_epoch, validation_error, validation_loss



def main():
    print("CUDA available: ", torch.cuda.is_available())
    BATCH_SIZE, best_models, config, device, errors, errors_train, master_mask_filepath, model_save_dir, num_epochs, num_splits, plot_after_num_epoch, root_dir, sampler_size, save_after_num_epoch, splits, statistics_array, training_error, training_loss, transform_img_to_tensor, transform_tensor_shape, validate_after_num_epoch, validation_error, validation_loss = do_setup()
    # training loop
    for i, (train_files, val_files) in enumerate(splits):
        currentStatistics = statistics_array[i]
        # Prepare dataset and dataloaders
        trn_dataset = GenderSplitDataset(root_dir, train_files, master_mask_filepath,
                                         transform=transform_img_to_tensor, cache_dir=config['cache_dir'],
                                         transform_shape=transform_tensor_shape, normalize=False)
        sampler = BalancedSampler(trn_dataset, sampler_size)
        dataloader = DataLoader(trn_dataset, batch_size=BATCH_SIZE, sampler=sampler)

        val_dataset = GenderSplitDataset(root_dir, val_files, master_mask_filepath,
                                         transform=transform_img_to_tensor, cache_dir=config['cache_dir'],
                                         transform_shape=transform_tensor_shape, normalize=False)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
        print("Number train images: ", len(trn_dataset))
        print("Number val images: ", len(val_dataset))
        # prepare model
        model_params = config.get("model_params", dict())
        model = get_obj_from_str(config["model_classname"])(**model_params).to(device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(model.parameters(), lr=config.training_params.learning_rate_model)
        plot_dir = config['plots_dir']
        for epoch in trange(num_epochs, desc=f"Training Split {i + 1}/{num_splits}"):
            trn_loss, trn_err = train_model_one_epoch(config, model, criterion, optimizer, dataloader, trn_dataset, i,
                                                      training_loss,
                                                      training_error,
                                                      currentStatistics, sampler_size=sampler_size, sampler=sampler)
            if (epoch % validate_after_num_epoch == 0 or epoch == num_epochs - 1):
                val_loss, val_err, number_of_certains, number_of_uncertains = validate_model_one_epoch(model,
                                                                                                       criterion,
                                                                                                       val_loader,
                                                                                                       val_dataset,
                                                                                                       i,
                                                                                                       validation_loss,
                                                                                                       validation_error,
                                                                                                       currentStatistics,
                                                                                                       epoch)
                if val_err < errors[i]:
                    errors[i] = val_err
                    errors_train[i] = trn_err
                    best_models[i] = model.state_dict()
                    for k, v in best_models[i].items():
                        best_models[i][k] = v.cpu()
                    currentStatistics.update_best()

                print(f"EPOCH {epoch} Training Loss: {trn_loss:.2f} Training Error: {trn_err:.2f} "
                      f"Validation Loss: {val_loss:.2f} Validation Error: {val_err:.2f} "
                      f"Best Validation Error: {errors[i]:.2f} Number of uncertain: {number_of_uncertains} ")

            if epoch % plot_after_num_epoch == 0:
                plot_loss_and_error(training_loss[i], training_error[i], validation_loss[i], validation_error[i],
                                    plot_dir, i + 1)
            if epoch % save_after_num_epoch == 0:
                torch.save(model.state_dict(), os.path.join(model_save_dir, f"model_fold_{i}.pt"))
    print_statistics(errors_train, errors, statistics_array)

if __name__ == '__main__':
    main()
