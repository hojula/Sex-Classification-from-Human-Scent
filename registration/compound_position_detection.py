import matplotlib.pyplot as plt
import os
import torch
from omegaconf import OmegaConf
from help_functions import set_system
from help_functions import from_times_pixels
from help_functions import find_positions
from match_txt_pt import find_match
from help_functions import load_compound_and_times
from help_functions import load_tensor_and_system
from help_functions import find_positions_cnn
from help_functions import load_folds
from help_functions import filter_files
from tqdm import tqdm, trange
from help_functions import plot_spectrogram
from einops import rearrange
import yaml
from help_functions import reshape_results
from help_functions import compute_avg_results

import numpy as np
from settings_compound_position_detection import *


def find(file_txt, t1_shift=-500, load_dir_txt=None, load_dir=None, config=None, settings_array=None, fold=None,
         cnn_model=None, reference_spectra=None, compound_numbers_cnn=None):
    file = find_match(file_txt, load_dir)
    all_compounds_txt_positions_times = load_compound_and_times(file_txt, load_dir_txt)
    spectrogram_image, system_number = load_tensor_and_system(file, t1_shift, dir_path=load_dir)
    all_compounds_reference_positions_times, rest = set_system(system_number, fold=fold)
    x_six_seconds, y_six_seconds, x_eight_seconds, y_eight_seconds, x_ten_seconds, y_ten_seconds = rest
    if fold is None:
        all_compounds_reference_positions_pixels = from_times_pixels(all_compounds_reference_positions_times,
                                                                     system_number,
                                                                     x_six_seconds, y_six_seconds, x_eight_seconds,
                                                                     y_eight_seconds, x_ten_seconds, y_ten_seconds,
                                                                     t1_shift)
    else:
        # saved as pixels already
        all_compounds_reference_positions_pixels = all_compounds_reference_positions_times
    all_compounds_txt_positions_pixels = from_times_pixels(all_compounds_txt_positions_times, system_number,
                                                           x_six_seconds, y_six_seconds, x_eight_seconds,
                                                           y_eight_seconds,
                                                           x_ten_seconds, y_ten_seconds, t1_shift)
    compounds_diff = {}
    all_compounds_reference_positions_pixels_backup = all_compounds_reference_positions_pixels.copy()
    for settings in settings_array:
        # filter only significant compounds if wanted
        all_compounds_reference_positions_pixels = all_compounds_reference_positions_pixels_backup.copy()
        if settings.only_significant:
            new_dict = {}
            for compound in all_compounds_reference_positions_pixels:
                if compound in config.calibration_compounds:
                    new_dict[compound] = all_compounds_reference_positions_pixels[compound]
            all_compounds_reference_positions_pixels = new_dict
        if settings.metric != 'cnn':
            avg_shift, compounds_pixels, shifts_for_compounds = find_positions(all_compounds_reference_positions_pixels,
                                                                               spectrogram_image, config, settings,
                                                                               fold, cnn_model, reference_spectra,
                                                                               compound_numbers_cnn,
                                                                               spectrum_dir=config.constants.avg_spectra_dir)
        else:
            compounds_pixels = find_positions_cnn(spectrogram_image, cnn_model, reference_spectra, compound_numbers_cnn)
        compounds_differences = {}
        for compound in compounds_pixels:
            if compound in all_compounds_txt_positions_pixels:
                t1 = abs(compounds_pixels[compound][0] - all_compounds_txt_positions_pixels[compound][0])
                t2 = abs(compounds_pixels[compound][1] - all_compounds_txt_positions_pixels[compound][1])
                t_vector = np.array([t1.cpu().numpy(), t2.cpu().numpy()])
                l1_norm = np.linalg.norm(t_vector, ord=1)
                compounds_differences[compound] = (l1_norm, t1.item(), t2.item())
        compounds_diff[settings] = compounds_differences
    return compounds_diff


def process_all_files(files, dir_txt, dir_pt, config, fold, t1_shift=-500, settings_array=None, train=False,
                      channels=801, save_path=os.getcwd()):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    compounds_diferences_avg_settings = {}
    compounds_diferences_avg_settings_without_banned = {}
    with open(config.constants.compound_numbers_cnn_path, 'r') as file:
        compound_numbers_cnn = yaml.load(file, Loader=yaml.FullLoader)
    reference_spectra = torch.zeros(len(compound_numbers_cnn), channels, device=device)
    for compound in compound_numbers_cnn:
        reference = torch.load(config.constants.folds_references_dir + 'fold' + str(fold) + '/' + compound + '.pt').to(
            device)
        compound_number = compound_numbers_cnn[compound]
        reference = rearrange(reference, 'l h w -> (l h w)')
        reference_spectra[compound_number] = reference
    cnn_model = torch.load(config.constants.folds_references_dir + 'fold' + str(fold) + '/cnn/model_fcn.pt').to(device)
    banned_compounds = ['Heptacosane', 'Nonacosane']
    for i in trange(len(files)):
        file = files[i]
        if file.endswith('.txt'):
            compounds_diferences_settings = find(file, t1_shift, dir_txt, dir_pt, config,
                                                 settings_array, fold, cnn_model, reference_spectra,
                                                 compound_numbers_cnn)
            for settings in settings_array:
                if settings not in compounds_diferences_avg_settings:
                    compounds_diferences_avg_settings[settings] = []
                    compounds_diferences_avg_settings_without_banned[settings] = []
                for compound, values in compounds_diferences_settings[settings].items():
                    compounds_diferences_avg_settings[settings].append(values)
                    if compound not in banned_compounds:
                        compounds_diferences_avg_settings_without_banned[settings].append(values)
    avg_diff_settings = {}
    for settings in compounds_diferences_avg_settings.keys():
        l1_norm_mean = np.mean([x[0] for x in compounds_diferences_avg_settings[settings]])
        l1_norm_median = np.median([x[0] for x in compounds_diferences_avg_settings[settings]])
        t1_mean = np.mean([x[1] for x in compounds_diferences_avg_settings[settings]])
        t1_median = np.median([x[1] for x in compounds_diferences_avg_settings[settings]])
        t2_median = np.median([x[2] for x in compounds_diferences_avg_settings[settings]])
        t2_mean = np.mean([x[2] for x in compounds_diferences_avg_settings[settings]])
        avg_diff_settings[settings] = (l1_norm_mean, l1_norm_median, t1_mean, t1_median, t2_mean, t2_median)
    avg_diff_settings_without_banned = {}
    for settings in compounds_diferences_avg_settings_without_banned.keys():
        l1_norm_mean = np.mean([x[0] for x in compounds_diferences_avg_settings_without_banned[settings]])
        l1_norm_median = np.median([x[0] for x in compounds_diferences_avg_settings_without_banned[settings]])
        t1_mean = np.mean([x[1] for x in compounds_diferences_avg_settings_without_banned[settings]])
        t1_median = np.median([x[1] for x in compounds_diferences_avg_settings_without_banned[settings]])
        t2_median = np.median([x[2] for x in compounds_diferences_avg_settings_without_banned[settings]])
        t2_mean = np.mean([x[2] for x in compounds_diferences_avg_settings_without_banned[settings]])
        avg_diff_settings_without_banned[settings] = (
            l1_norm_mean, l1_norm_median, t1_mean, t1_median, t2_mean, t2_median)
    return avg_diff_settings, avg_diff_settings_without_banned


def run_test(config, settings_array, m_training, f_training, m_testing, f_testing):
    files_all = os.listdir(config.constants.dir_txt)
    num_folds = config.constants.num_folds
    if not os.path.exists(config.constants.save_dir):
        os.makedirs(config.constants.save_dir)
    training_results, training_results_without_banned = run_set(config, f_training, files_all, m_training,
                                                                settings_array)
    testing_results, testing_results_without_banned = run_set(config, f_testing, files_all, m_testing,
                                                                settings_array)
    avg_results_training = compute_avg_results(training_results, num_folds)
    avg_results_testing = compute_avg_results(testing_results, num_folds)
    avg_results_training_without_banned = compute_avg_results(training_results_without_banned,
                                                              num_folds)
    avg_results_testing_without_banned = compute_avg_results(testing_results_without_banned,
                                                             num_folds)

    print("Training results")
    print(avg_results_training)
    print("Testing results")
    print(avg_results_testing)
    with open(config.constants.save_dir + 'training_results.txt', 'w') as file:
        file.write(str(avg_results_training))
    print("Training results without banned")
    print(avg_results_training_without_banned)
    print("Testing results without banned")
    print(avg_results_testing_without_banned)
    with open(config.constants.save_dir + 'training_results_without_banned.txt', 'w') as file:
        file.write(str(avg_results_training_without_banned))


def run_set(config, f_training, files_all, m_training, settings_array):
    i = 1
    training_results = {}
    training_results_without_banned = {}
    for m_ids, f_ids in zip(m_training, f_training):
        if config.constants.fold_num != -1 and i != config.constants.fold_num:
            i += 1
            continue
        print("Fold", i)
        print("M ids", m_ids)
        print("F ids", f_ids)
        print("All files", len(files_all))
        files = filter_files(files_all, m_ids, f_ids)
        files = files[:2]
        diff, diff_without_banned = process_all_files(files, config.constants.dir_txt, config.constants.dir_pt,
                                                      config, i,
                                                      settings_array=settings_array, train=True,
                                                      save_path=config.constants.save_dir)
        results = reshape_results(diff)
        results_without_banned = reshape_results(diff_without_banned)
        training_results[i] = results
        training_results_without_banned[i] = results_without_banned
        i += 1
    return training_results, training_results_without_banned

def main():
    config = OmegaConf.load('config.yaml')
    print("Config loaded")
    settings_array = set_settings_for_metric_test()
    # settings_array = set_settings_for_box_test()
    # settings_array = set_settings_for_significant_compounds_test()
    # settings_array = set_settings_svm()
    # settings_array = set_settings_for_weighting_test_24()
    # settings_array = set_settings_for_weighting_test()
    #settings_array = set_settings_cnn()
    m_folds = load_folds(config.constants.f_folds_path)
    f_folds = load_folds(config.constants.m_folds_path)
    num_folds = config.constants.num_folds
    m_testing = [m_folds[i] for i in range(num_folds)]
    f_testing = [f_folds[i] for i in range(num_folds)]
    m_training = [np.concatenate([m_folds[j] for j in range(num_folds) if j != i]) for i in range(num_folds)]
    f_training = [np.concatenate([f_folds[j] for j in range(num_folds) if j != i]) for i in range(num_folds)]
    print("Settings loaded")
    run_test(config, settings_array, m_training, f_training, m_testing, f_testing)

if __name__ == '__main__':
    main()
