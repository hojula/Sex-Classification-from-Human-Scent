from matplotlib import pyplot as plt
from omegaconf import OmegaConf
import random

from torch.utils.data import Dataset, DataLoader

import torch
from einops import rearrange
import numpy as np
import os
from help_functions import from_times_pixels
from help_functions import load_compound_and_times
from help_functions import set_system
from help_functions import load_folds
from help_functions import filter_files

from models import MyModel
from tqdm import tqdm, trange


def create_kfold_datasets(dir_path, k):
    measurements = os.listdir(dir_path)
    random.seed(42)
    random.shuffle(measurements)

    if k == 1:
        return [(measurements, [])]

    kfold_datasets = []
    for i in range(k):
        test_data = measurements[i::k]  # Select test data for this fold
        train_data = [m for m in measurements if m not in test_data]  # Select training data
        kfold_datasets.append((train_data, test_data))

    return kfold_datasets


class CompoundsDataset(Dataset):
    def __init__(self, dir_path, measurements, config, fold=1):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cfg_numbers_ori = OmegaConf.load(config.constants.compound_numbers_path)
        self.cfg_numbers = {}
        k = 0
        for compound, number in self.cfg_numbers_ori.items():
            if compound in config.calibration_compounds:
                self.cfg_numbers[compound] = k
                k += 1
        self.cfg_number_to_compound = {v: k for k, v in self.cfg_numbers.items()}
        reference_spectra = torch.zeros(len(self.cfg_numbers), 801, device=device)
        for compound in self.cfg_numbers:
            reference = torch.load(
                config.constants.folds_references_dir + 'fold' + str(fold) + '/' + compound + '.pt').to(
                device)
            reference = rearrange(reference, 'l h w -> (l h w)')
            compound_number = self.cfg_numbers[compound]
            reference_spectra[compound_number] = reference
        self.data = {}
        self.number_of_compounds = len(config.calibration_compounds)
        self.compounds_numbers = {}
        self.all_data = []
        system_counts = {}
        for measurement in measurements:
            compounds = os.listdir(os.path.join(dir_path, measurement))
            system_num = int(measurement.split('_')[1])
            if system_num not in system_counts:
                system_counts[system_num] = 0
            system_counts[system_num] += 1
            _, rest = set_system(system_num, fold=fold)
            x_six_seconds, y_six_seconds, x_eight_seconds, y_eight_seconds, x_ten_seconds, y_ten_seconds = rest
            width = x_six_seconds + x_eight_seconds + x_ten_seconds
            height = y_ten_seconds
            prefix_end = measurement.index("_M") if "_M" in measurement else measurement.index("_F")
            system_prefix = measurement[:prefix_end]  # Extracts "system_3"
            txt_file = measurement
            compounds_times = load_compound_and_times(txt_file, config.constants.dir_txt)
            system_number = int(system_prefix.split('_')[1])
            all_compounds_reference_positions_times, rest = set_system(system_number)
            x_six_seconds, y_six_seconds, x_eight_seconds, y_eight_seconds, x_ten_seconds, y_ten_seconds = rest
            all_compounds_reference_positions_pixels = from_times_pixels(
                compounds_times,
                system_number,
                x_six_seconds, y_six_seconds, x_eight_seconds,
                y_eight_seconds, x_ten_seconds, y_ten_seconds, -500)
            for compound_file in compounds:
                try:
                    compound_name = compound_file.replace('.pt', '')
                except IndexError:
                    continue
                if compound_name not in config.calibration_compounds:
                    continue
                # Load data for the compound
                data_path = os.path.join(dir_path, measurement, compound_file)
                data = torch.load(data_path).to(device)
                data = data.float()
                data = torch.nn.functional.normalize(data, dim=0)
                t1, t2 = all_compounds_reference_positions_pixels[compound_name]
                # convert to tensor of same type as data
                t1_f = float(t1)
                t2_f = float(t2)
                # normalize time to 0-1
                t1_f = t1_f / (width - 1)
                t2_f = t2_f / (height - 1)
                t1_tensor = torch.full(data.shape, t1_f, device=device)
                t2_tensor = torch.full(data.shape, t2_f, device=device)
                # add time t1 and t2 as new channels
                # 801,c,1
                data_good = torch.cat((data, t1_tensor, t2_tensor), dim=1)
                data_good = data_good.float()
                # get refernece for both objects
                # c, 801, 1
                data_good = rearrange(data_good, 'h c w -> c h w')
                # new_element = (compound_name, (data, cosine_similarities))
                new_element = (compound_name, data_good)
                self.all_data.append(new_element)

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        compound_name, data = self.all_data[idx]
        label = torch.zeros(len(self.cfg_numbers), device=device)
        label[self.cfg_numbers[compound_name]] = 1
        return label, data, True


def run_one_fold(m_ids_train, m_ids_test, f_ids_train, f_ids_test, files_all, device, config, best_val_accuracy_splits,
                 bestl_models, i, save_dir):
    if config.constants.fold_num != -1 and i != config.constants.fold_num:
        i += 1
        return best_val_accuracy_splits, bestl_models, i
    print("Fold", i)
    print("M ids", m_ids_train)
    print("F ids", f_ids_train)
    print("All files", len(files_all))
    files = filter_files(files_all, m_ids_train, f_ids_train)
    train_dataset = CompoundsDataset(config.constants.dir_all_spectra, files, config, fold=i)
    if config.constants.testing_while_training_cnn:
        test_data = filter_files(files_all, m_ids_test, f_ids_test)
        test_dataset = CompoundsDataset(config.constants.dir_all_spectra, test_data, config,
                                        fold=i)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    print("Size of training data", len(train_dataset))
    print("Size of testing data", len(test_dataset))
    output_size = len(test_dataset.cfg_numbers)
    model = MyModel(output_size, 4).to(device)
    # prefer noise
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    for epoch in trange(2):
        model.train()
        train_loss = 0
        validation_loss = 0
        for labels, data, real in train_loader:
            labels = labels.float()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        model.eval()
        train_correct = 0
        train_total = 0
        for labels, data, real in train_loader:
            labels = labels.float()
            output = model(data)
            loss = criterion(output, labels)
            train_loss += loss.item()
            output = output[real]
            labels = labels[real]
            train_correct += (output.argmax(1) == labels.argmax(1)).sum().item()
            train_total += len(labels)
        if config.constants.testing_while_training_cnn:
            with torch.no_grad():
                test_correct = 0
                test_total = 0
                for labels, data, real in test_loader:
                    labels = labels.float()
                    output = model(data)
                    loss = criterion(output, labels)
                    validation_loss += loss.item()
                    output = output[real]
                    labels = labels[real]
                    test_correct += (output.argmax(1) == labels.argmax(1)).sum().item()
                    test_total += len(labels)
            print(
                f"Epoch {epoch}: Train accuracy: {train_correct / train_total}, Train loss: {train_loss / len(train_loader)}, Validation accuracy: {test_correct / test_total}, Validation loss: {validation_loss / len(test_loader)}")
            if test_correct / test_total > best_val_accuracy_splits[i - 1]:
                best_val_accuracy_splits[i - 1] = test_correct / test_total
                bestl_models[i - 1] = model
        else:
            print(f"Epoch {epoch}: Train accuracy: {train_correct / train_total}")

    torch.save(bestl_models[i - 1], os.path.join(save_dir, 'model_fcn_fold' + str(i) + '.pt'))
    i += 1
    return best_val_accuracy_splits, bestl_models, i


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = OmegaConf.load("config.yaml")
    save_dir = config.constants.save_dir + '/cnn'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    num_folds = config.constants.num_folds
    m_folds = load_folds(config.constants.f_folds_path)
    f_folds = load_folds(config.constants.m_folds_path)
    m_testing = [m_folds[i] for i in range(num_folds)]
    f_testing = [f_folds[i] for i in range(num_folds)]
    m_training = [np.concatenate([m_folds[j] for j in range(num_folds) if j != i]) for i in range(num_folds)]
    f_training = [np.concatenate([f_folds[j] for j in range(num_folds) if j != i]) for i in range(num_folds)]
    best_val_accuracy_splits = [0 for i in range(num_folds)]
    bestl_models = [None for i in range(num_folds)]
    i = 1
    files_all = os.listdir(config.constants.dir_all_spectra)
    for m_ids_train, m_ids_test, f_ids_train, f_ids_test in zip(m_training, m_testing, f_training, f_testing):
        best_val_accuracy_splits, bestl_models, i = run_one_fold(m_ids_train, m_ids_test, f_ids_train, f_ids_test,
                                                                 files_all, device, config, best_val_accuracy_splits,
                                                                 bestl_models, i, save_dir)


if __name__ == "__main__":
    main()
