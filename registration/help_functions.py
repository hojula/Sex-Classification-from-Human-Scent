import matplotlib.pyplot as plt
import pandas as pd
from omegaconf import OmegaConf
import os
import torch
import numpy as np
import matplotlib
import cv2
from einops import rearrange
import h5py
from time import sleep
import yaml

SCAN_LENGHT_SYSTEM_1 = 714800
SCAN_LENGHT_SYSTEM_2 = 715600
SCAN_LENGHT_SYSTEM_3 = 715600


def set_system_1():
    scan_length = SCAN_LENGHT_SYSTEM_1
    y_six_seconds = 1200  # 6/0.005 (scan_duration)
    x_six_seconds = 200
    # y_eight_seconds = 1600  # 8/0.005 (scan_duration)
    y_eight_seconds = 1200
    x_eight_seconds = 149
    y_ten_seconds = 2000  # 10/0.005 (scan_duration)
    x_ten_seconds = (scan_length - x_six_seconds * y_six_seconds
                     - x_eight_seconds * y_eight_seconds) // y_ten_seconds
    return x_six_seconds, y_six_seconds, x_eight_seconds, y_eight_seconds, x_ten_seconds, y_ten_seconds


def set_system_2():
    scan_length = SCAN_LENGHT_SYSTEM_2
    y_six_seconds = 1200  # 6/0.005 (scan_duration)
    x_six_seconds = 200
    y_eight_seconds = 1600  # 8/0.005 (scan_duration)
    x_eight_seconds = 111
    y_ten_seconds = 2000  # 10/0.005 (scan_duration)
    x_ten_seconds = (scan_length - x_six_seconds * y_six_seconds
                     - x_eight_seconds * y_eight_seconds) // y_ten_seconds
    return x_six_seconds, y_six_seconds, x_eight_seconds, y_eight_seconds, x_ten_seconds, y_ten_seconds


def set_system_3():
    scan_length = SCAN_LENGHT_SYSTEM_3
    y_six_seconds = 1200  # 6/0.005 (scan_duration)
    x_six_seconds = 200
    y_eight_seconds = 1600  # 8/0.005 (scan_duration)
    x_eight_seconds = 111
    y_ten_seconds = 2000  # 10/0.005 (scan_duration)
    x_ten_seconds = (scan_length - x_six_seconds * y_six_seconds
                     - x_eight_seconds * y_eight_seconds) // y_ten_seconds
    return x_six_seconds, y_six_seconds, x_eight_seconds, y_eight_seconds, x_ten_seconds, y_ten_seconds


def set_system_weird():
    scan_length = SCAN_LENGHT_SYSTEM_2
    y_six_seconds = 1200  # 6/0.005 (scan_duration)
    x_six_seconds = 0
    y_eight_seconds = 1600  # 8/0.005 (scan_duration)
    x_eight_seconds = 150
    y_ten_seconds = 2000  # 10/0.005 (scan_duration)
    x_ten_seconds = (scan_length - x_six_seconds * y_six_seconds
                     - x_eight_seconds * y_eight_seconds) // y_ten_seconds
    return x_six_seconds, y_six_seconds, x_eight_seconds, y_eight_seconds, x_ten_seconds, y_ten_seconds


def set_system(system_number, weird_files=None, fold=None):
    reference_compounds_system1 = OmegaConf.load('compounds_system1.yaml')
    reference_compounds_system2 = OmegaConf.load('compounds_system2.yaml')
    reference_compounds_system3 = OmegaConf.load('compounds_system3.yaml')

    if fold is None:
        if system_number == 1:
            return reference_compounds_system1, set_system_1()
        elif system_number == 2:
            return reference_compounds_system2, set_system_2()
        elif system_number == 3:
            return reference_compounds_system3, set_system_3()
        else:
            return reference_compounds_system3, set_system_weird()
    else:
        system_number_str = system_number
        if system_number == -1:
            system_number_str = 3
        with open(f'times_system{system_number_str}.yaml', 'r') as file:
            reference_compounds = file.read()
            reference_compounds = yaml.load(reference_compounds, Loader=yaml.FullLoader)
        ref_c = {}
        for c, val in reference_compounds.items():
            t1 = int(val[0])
            t2 = int(val[1])
            ref_c[c] = (t1, t2)
        reference_compounds = ref_c
        if system_number == 1:
            return reference_compounds, set_system_1()
        elif system_number == 2:
            return reference_compounds, set_system_2()
        elif system_number == 3:
            return reference_compounds, set_system_3()
        else:
            return reference_compounds, set_system_weird()


def load_tensor_and_system(file, t1_shift, dir_path=None):
    T1_TIME_SYSTEM_1 = 497
    T1_TIME_SYSTEM_2 = 460
    T1_TIME_SYSTEM_3 = 460
    spectrogram_image = torch.load(os.path.join(dir_path, file),
                                   map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    if (int)(spectrogram_image.shape[2]) == T1_TIME_SYSTEM_1:
        system_number = 1
    elif (int)(spectrogram_image.shape[2]) == T1_TIME_SYSTEM_2:
        if 'system2' in file:
            system_number = 2
        else:
            system_number = 3
    else:
        system_number = -1
    return spectrogram_image, system_number


def extract_spectrums(spectrogram_image, compounds_pixels):
    compounds_spectrum = {}
    for c in compounds_pixels:
        x = compounds_pixels[c][0]
        y = compounds_pixels[c][1]
        spectrum = spectrogram_image[:, y:y + 1, x:x + 1].clone()
        compounds_spectrum[c] = spectrum
    return compounds_spectrum


def from_times_pixels(compounds: dict, system_number, x_six_seconds, y_six_seconds, x_eight_seconds, y_eight_seconds,
                      x_ten_seconds, y_ten_seconds, t1_shift=0, t1_offset=0, t2_offset=0):
    compounds_pixels = {}
    if system_number == 1:
        min_ten_seconds = x_six_seconds * 6 + x_eight_seconds * 6
        for compound in compounds:
            t1 = compounds[compound]['t1'] + t1_offset + t1_shift
            t2 = compounds[compound]['t2'] + t2_offset
            t2 *= 1000
            t2_pixels = t2 // 5
            if t1 < min_ten_seconds:
                t1_pixels = t1 // 6
            else:
                t1 = t1 - min_ten_seconds
                t1_pixels = (x_six_seconds + x_eight_seconds) + t1 // 10
            t2_pixels = int(t2_pixels)
            t1_pixels = int(t1_pixels)
            compounds_pixels[compound] = (t1_pixels, t2_pixels)
    elif system_number == -1:
        min_ten_seconds = x_eight_seconds * 8
        for compound in compounds:
            t1 = compounds[compound]['t1'] + t1_offset + t1_shift
            t2 = compounds[compound]['t2'] + t2_offset
            t2 *= 1000
            t2_pixels = t2 // 5
            if t1 < min_ten_seconds:
                t1_pixels = t1 // 8
            else:
                t1 = t1 - min_ten_seconds
                t1_pixels = x_eight_seconds + t1 // 10
            t2_pixels = int(t2_pixels)
            t1_pixels = int(t1_pixels)
            compounds_pixels[compound] = (t1_pixels, t2_pixels)
    else:
        min_ten_seconds = x_six_seconds * 6 + x_eight_seconds * 8
        min_eight_seconds = x_six_seconds * 6
        for compound in compounds:
            t1 = compounds[compound]['t1'] + t1_offset + t1_shift
            t2 = compounds[compound]['t2'] + t2_offset
            t2 *= 1000
            t2_pixels = t2 // 5
            if t1 < min_eight_seconds:
                t1_pixels = t1 // 6
            elif t1 < min_ten_seconds:
                t1 = t1 - min_eight_seconds
                t1_pixels = x_six_seconds + t1 // 8
            else:
                t1 = t1 - min_ten_seconds
                t1_pixels = (x_six_seconds + x_eight_seconds) + t1 // 10
            t2_pixels = int(t2_pixels)
            t1_pixels = int(t1_pixels)
            compounds_pixels[compound] = (t1_pixels, t2_pixels)
    return compounds_pixels


def find_positions_cnn(spectrogram_image, model, reference_spectra, compound_numbers, batch_size=32768):
    spectrogram_image = spectrogram_image.float()
    spectrogram_image = torch.nn.functional.normalize(spectrogram_image, dim=0)
    channels, height, width = spectrogram_image.shape
    x_coords = torch.arange(width).unsqueeze(0).expand(height, width).unsqueeze(0).float().to(
        spectrogram_image.device)
    y_coords = torch.arange(height).unsqueeze(1).expand(height, width).unsqueeze(0).float().to(
        spectrogram_image.device)
    x_coords = x_coords / (width - 1)
    y_coords = y_coords / (height - 1)
    x_coords = x_coords.expand(channels, height, width)
    y_coords = y_coords.expand(channels, height, width)
    image_with_coords = torch.stack((spectrogram_image, x_coords, y_coords), dim=0)
    image_flattened = rearrange(image_with_coords, 'c s h w -> (h w) c s 1')  # Shape: (height * width, 3, 801, 1)
    model.eval()
    logits_batches = []
    with torch.no_grad():
        for i in range(0, len(image_flattened), batch_size):
            data = image_flattened[i:i + batch_size]
            logits_batch = model(data)
            logits_batches.append(logits_batch)
    logits = torch.cat(logits_batches, dim=0)
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    probabilities = rearrange(probabilities, '(h w) c -> c h w', h=height, w=width)
    mask = torch.ones_like(probabilities[0])
    mask[:, :75] = 0
    mask[:, -75:] = 0
    mask[:300, :] = 0
    mask[-300:, :] = 0
    probabilities = probabilities * mask
    # Get predicted classes
    predicted_classes = probabilities.argmax(0)

    num_classes = len(compound_numbers)
    max_cords = {}

    # Iterate over classes to find max coordinates
    for i in range(num_classes):
        mask_local = (predicted_classes == i)
        spectrum = reference_spectra[i].unsqueeze(1).unsqueeze(2)
        spectrum = torch.nn.functional.normalize(spectrum, dim=0)
        compound_name = list(compound_numbers.keys())[list(compound_numbers.values()).index(i)]
        if mask_local.sum() > 0:
            cosine_similarities = spectrum * spectrogram_image
            cosine_similarities = cosine_similarities.sum(dim=0)
            regulatization = 1
            cosine_similarities = cosine_similarities * mask_local * mask
            # uncomment for only probabilities or cosine similarities
            # aggregated_value_per_pixel = probabilities[i]
            # aggregated_value_per_pixel = cosine_similarities * regulatization
            aggregated_value_per_pixel = probabilities[i] + cosine_similarities * regulatization
            index_of_max = torch.argmax(aggregated_value_per_pixel)
            max_value = torch.max(aggregated_value_per_pixel)
            x, y = index_of_max % width, index_of_max // width
            max_cords[compound_name] = (x, y)
    return max_cords


def find_positions(compounds: dict, spectrogram_image: torch.Tensor, config, settings, fold=None, cnn_model=None,
                   reference_spectra=None, compound_numbers_cnn=None,
                   spectrum_dir=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    avg_shift_x = 0
    avg_shift_y = 0
    counter = 0
    ref_compounds_pixels = {}
    shifts_for_compounds = {}
    product_svm = None
    for compound in compounds:
        if fold is None:
            spectrum = torch.load(os.path.join(spectrum_dir, f'{compound}.pt'), weights_only=False).to(device)
        else:
            spectrum = torch.load(
                config.constants.folds_references_dir + 'fold' + str(fold) + '/' + compound + '.pt').to(device)
            svm_model = torch.load(
                config.constants.folds_references_dir + 'fold' + str(fold) + '/svm/l2/model_c100.pt')
        x_compound = compounds[compound][0]
        y_compound = compounds[compound][1]
        new_x_compound, new_y_compound, _, product, product_svm = find_position(spectrogram_image,
                                                                                spectrum, x_compound, y_compound,
                                                                                settings,
                                                                                svm_model=svm_model,
                                                                                compound=compound,
                                                                                cnn_model=cnn_model,
                                                                                reference_spectra=reference_spectra,
                                                                                compound_numbers_cnn=compound_numbers_cnn,
                                                                                svm_product=product_svm)
        if new_x_compound == -1 and new_y_compound == -1:
            print(f"Compound {compound} not found")
            continue
        ref_compounds_pixels[compound] = (new_x_compound, new_y_compound)
        shift_x = new_x_compound - x_compound
        shift_y = new_y_compound - y_compound
        shifts_for_compounds[compound] = (shift_x, shift_y)
        avg_shift_x += shift_x
        avg_shift_y += shift_y
        counter += 1
    if counter == 0:
        return None, None, None
    avg_shift_x = avg_shift_x / counter
    avg_shift_y = avg_shift_y / counter
    avg_shift_x = int(avg_shift_x.item())
    avg_shift_y = int(avg_shift_y.item())
    avg_tup = (avg_shift_x, avg_shift_y)
    return avg_tup, ref_compounds_pixels, shifts_for_compounds


def find_position(spectrogram_image, spectrum, x_compound, y_compound, settings, compound_index=None,
                  dot_product_threshold=None, svm_model=None, compound=None, cnn_model=None,
                  reference_spectra=None, compound_numbers_cnn=None, svm_product=None):
    metric = settings.metric
    x_left = x_compound - settings.box_add_x
    x_right = x_compound + 1 + settings.box_add_x
    y_down = y_compound - settings.box_add_y
    y_up = y_compound + 1 + settings.box_add_y
    if x_left < 0:
        x_left = 0
    if x_right > spectrogram_image.shape[2] - 1:
        x_right = spectrogram_image.shape[2] - 1
    if y_down < 0:
        y_down = 0
    if y_up > spectrogram_image.shape[1] - 1:
        y_up = spectrogram_image.shape[1] - 1
    channels, height, width = spectrogram_image.shape
    if svm_product is None or settings.box_add_x != -1:
        if compound_index is None:
            if metric != 'cnn_box':
                if settings.box_add_x != -1:
                    cut_spectrogram_image = spectrogram_image[:,
                                            y_down:y_up,
                                            x_left:x_right].clone()
                else:
                    cut_spectrogram_image = spectrogram_image.clone()
            else:
                spectrogram_image = spectrogram_image.float()
                spectrogram_image = torch.nn.functional.normalize(spectrogram_image, dim=0)
                channels, height, width = spectrogram_image.shape
                x_coords = torch.arange(width).unsqueeze(0).expand(height, width).unsqueeze(0).float().to(
                    spectrogram_image.device)
                y_coords = torch.arange(height).unsqueeze(1).expand(height, width).unsqueeze(0).float().to(
                    spectrogram_image.device)
                x_coords = x_coords / (width - 1)
                y_coords = y_coords / (height - 1)
                x_coords = x_coords.expand(channels, height, width)
                y_coords = y_coords.expand(channels, height, width)
                image_with_coords = torch.stack((spectrogram_image, x_coords, y_coords), dim=0)
                image_with_coords = image_with_coords[:, :, y_down:y_up, x_left:x_right].clone()
                height, width = image_with_coords.shape[2], image_with_coords.shape[3]
                cut_spectrogram_image = spectrogram_image[:, y_down:y_up, x_left:x_right].clone()
                cut_spectrogram_image = torch.nn.functional.normalize(cut_spectrogram_image, dim=0)
                if cut_spectrogram_image.shape[0] > spectrum.shape[0]:
                    cut_spectrogram_image = cut_spectrogram_image[:spectrum.shape[0], :, :]
                elif cut_spectrogram_image.shape[0] < spectrum.shape[0]:
                    spectrum = spectrum[:cut_spectrogram_image.shape[0], :, :]
        else:
            compound_index = [int(index) for index in compound_index]
            cut_spectrogram_image = spectrogram_image[compound_index,
                                    y_down:y_up,
                                    x_left:x_right].clone()
            spectrum = spectrum[compound_index, :, :]
    if metric != 'cnn_box' and (svm_product is None or settings.box_add_x != -1):
        cut_spectrogram_image = cut_spectrogram_image.double()
        spectrum = spectrum.double()
        original_cut_spectrogram_image = cut_spectrogram_image.clone()
        cut_spectrogram_image = torch.nn.functional.normalize(cut_spectrogram_image, dim=0)
        spectrum = torch.nn.functional.normalize(spectrum, dim=0)
        if cut_spectrogram_image.shape[0] > spectrum.shape[0]:
            cut_spectrogram_image = cut_spectrogram_image[:spectrum.shape[0], :, :]
        elif cut_spectrogram_image.shape[0] < spectrum.shape[0]:
            spectrum = spectrum[:cut_spectrogram_image.shape[0], :, :]
    if metric == 'dot':
        product = spectrum * cut_spectrogram_image
        product = product.sum(dim=0)
    elif metric == 'l1':
        product = torch.abs(spectrum - cut_spectrogram_image)
        product = product.sum(dim=0)
    elif metric == 'l2':
        product = (spectrum - cut_spectrogram_image) ** 2
        product = product.sum(dim=0)
        product = product.sqrt()
    elif metric == 'l_inf':
        product = torch.abs(spectrum - cut_spectrogram_image)
        product = product.max(dim=0).values
    elif metric == 'svm':
        # hopefully this works
        config_numbers = OmegaConf.load('compound_numbers.yaml')
        # returns signed distance to the hyperplane
        if svm_product is None or settings.box_add_x != -1:
            h, w = cut_spectrogram_image.shape[1], cut_spectrogram_image.shape[2]
            cut_spectrogram_image = rearrange(cut_spectrogram_image, 'c h w -> (h w) c')
            cut_spectrogram_image = cut_spectrogram_image.cpu().numpy()
            product_svm = svm_model.decision_function(cut_spectrogram_image)
        else:
            h, w = svm_product.shape[1], svm_product.shape[2]
            product_svm = rearrange(svm_product, 'c h w -> (h w) c')
        compound_index = config_numbers[compound]
        product = product_svm[:, compound_index]
        product_svm = rearrange(product_svm, '(h w) c -> c h w', h=h)
        # print(product_svm.shape)
        product = rearrange(product, '(h w) -> h w', h=h)
        product = torch.tensor(product)
    elif metric == 'cnn_box':
        image_flattened = rearrange(image_with_coords, 'c s h w -> (h w) c s 1')  # Shape: (height * width, 3, 801, 1)
        cnn_model.eval()
        logits_batches = []
        with torch.no_grad():
            data = image_flattened
            logits_batch = cnn_model(data)
            logits_batches.append(logits_batch)

        logits = torch.cat(logits_batches, dim=0)
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        probabilities = rearrange(probabilities, '(h w) c -> c h w', h=height, w=width)
        # Get predicted classes
        predicted_classes = probabilities.argmax(0)
        compound_number = compound_numbers_cnn[compound]
        mask_local = (predicted_classes == compound_number)
        regulatization = 1
        probabilites = probabilities[compound_number] * mask_local
        cosine = torch.nn.functional.cosine_similarity(spectrum, cut_spectrogram_image, dim=0)
        cosine = cosine * mask_local
        product = probabilites + cosine * regulatization

    else:
        raise ValueError(f"Metric {metric} not supported")

    if settings.weighted:
        center_x = (x_right - x_left) // 2
        center_y = (y_up - y_down) // 2

        y_indices, x_indices = torch.meshgrid(torch.arange(product.shape[0]), torch.arange(product.shape[1]))
        x_distance = torch.abs(x_indices - center_x)  # 1 pixel = 6s
        y_distance = torch.abs(y_indices - center_y) * 6  # 1 pixel = 0.005s
        x_distance = x_distance
        y_distance = y_distance
        weighted_distances = x_distance + y_distance
        # weighted_distances = weighted_distances / torch.max(weighted_distances)
        penalty = settings.weighting_function(weighted_distances)
        max_penalty = torch.max(penalty)
        penalty = penalty / max_penalty
        scaling_factor = 1
        penalty = scaling_factor * penalty
        # print penalty device
        penalty = penalty.to(product.device)
        product -= penalty

    if metric[0] != 'l':
        index_of_max_dot_product = torch.argmax(product)
    else:
        index_of_max_dot_product = torch.argmin(product)
    value_of_max_dot_product = product.flatten()[index_of_max_dot_product]
    if not settings.weighted and (
            dot_product_threshold is not None and value_of_max_dot_product < dot_product_threshold):
        return -1, -1, -1
    y_pos = index_of_max_dot_product // product.shape[1]
    x_pos = index_of_max_dot_product % product.shape[1]
    if settings.box_add_x != -1:
        new_x_compound = x_compound - settings.box_add_x + x_pos
        new_y_compound = y_compound - settings.box_add_y + y_pos
    else:
        new_x_compound = x_pos
        new_y_compound = y_pos
    if metric != 'svm':
        return new_x_compound, new_y_compound, value_of_max_dot_product.item(), product, None
    else:
        return new_x_compound, new_y_compound, value_of_max_dot_product.item(), product, product_svm


def load_compound_and_times(file, tmp_dir) -> (int, dict):
    dir_path = tmp_dir
    separator = "\t"
    compounds_times_dict = {}
    file_name = os.path.splitext(file)[0]
    system_number = int(file_name.split("_")[1])
    data_frame = pd.read_csv(dir_path + "/" + file, sep=separator)
    for index, row in data_frame.iterrows():
        name = row[0]
        if name == '1-Octanol, 2-hexyl-':
            name = '2-Hexyl-1-octanol'
        if name == 'Benzenoic acid, tetradecyl ester':
            name = 'Benzoic acid, tetradecyl ester'
        if name == "5,10-Diethoxy-2,3,7,8-tetrahydro-1H,6H-dipyrrolo[1,2-a:1',2'-d]pyrazine":
            name = "5,10-Diethoxy-2,3,7,8-tetrahydro-1H,6H-dipyrrolo[1,2-a_1',2'-d]pyrazine"
        compounds_times_dict[name] = {'t1': int(row[1]), 't2': float(row[2].replace(',', '.'))}
    return compounds_times_dict


def apply_color_map(grayscale_image):
    cm = matplotlib.colormaps['viridis']
    return cm(grayscale_image)[..., :3] * 255


def plot_spectrogram(spectrogram_image: np.ndarray, save_path: str = None,
                     invert_color_channels: bool = False, channel_order: str = 'CHW',
                     compounds_pixels_predicted: list = None,
                     x_shift=0,
                     y_shift=0, compounds_pixels_txt=None):
    spectrogram_image = spectrogram_image.sum(0)
    spectrogram_image = spectrogram_image - spectrogram_image[spectrogram_image > 0].min()
    spectrogram_image[spectrogram_image < 0] = 0
    spectrogram_image = spectrogram_image[::-1]
    viz = spectrogram_image[np.newaxis]
    original_height = viz.shape[1]
    original_width = viz.shape[2]
    if channel_order == 'CHW':
        viz = viz.transpose((1, 2, 0))
    elif channel_order != 'HWC':
        raise RuntimeError(f'Unknown channel order {channel_order}!')
    if invert_color_channels:
        viz = viz[..., ::-1]
    viz = np.log(viz + 1e-10)
    min_grather_zero = viz[viz > 0].min()
    viz[viz < 0] = min_grather_zero
    viz = viz / viz.max()
    viz = apply_color_map(viz)
    viz = np.squeeze(viz, axis=2)
    result = np.zeros((viz.shape[0], 5 * viz.shape[1], 3))
    for i in range(5):
        result[:, i::5, :] = viz
    viz = result
    if compounds_pixels_predicted is not None:
        for compound in compounds_pixels_predicted:
            if compound in compounds_pixels_txt:
                x_compound_predicted, y_compound_predicted = compounds_pixels_predicted[compound]
                x_compound_predicted += x_shift
                y_compound_predicted += y_shift
                x_compound_predicted *= 5
                x_compound_predicted += 2
                y_compound_predicted = original_height - y_compound_predicted - 1
                # red circle for predicted
                x_compound_predicted = int(x_compound_predicted)
                y_compound_predicted = int(y_compound_predicted)
                viz = cv2.circle(viz, (x_compound_predicted, y_compound_predicted), radius=4, color=(255, 0, 0),
                                 thickness=1)
                # blue circle for txt
                x_compound_txt, y_compound_txt = compounds_pixels_txt[compound]
                x_compound_txt *= 5
                x_compound_txt += 2
                y_compound_txt = original_height - y_compound_txt - 1
                x_compound_txt = int(x_compound_txt)
                y_compound_txt = int(y_compound_txt)
                viz = cv2.circle(viz, (x_compound_txt, y_compound_txt), radius=4, color=(0, 0, 0), thickness=1)
                # plot line between txt and predicted
                viz = cv2.line(viz, (x_compound_predicted, y_compound_predicted), (x_compound_txt, y_compound_txt),
                               color=(0, 0, 0), thickness=1)
                # plot text
                font = cv2.FONT_HERSHEY_SIMPLEX
                text = str(compound)
                text_position = (x_compound_predicted + 5, y_compound_predicted)
                font_scale = 0.5
                font_thickness = 1
                font_color = (255, 255, 255)
                viz = cv2.putText(viz, text, text_position, font, font_scale, font_color, font_thickness)

    print("Saving image to", save_path)
    cv2.imwrite(save_path, viz[..., ::-1])


def plot_np_array_with_values(np_array: np.ndarray, save_path: str = None):
    plt.figure(figsize=(8, 6))
    cax = plt.imshow(np_array, cmap='viridis', aspect='auto')
    plt.title('Squalene cosine similarity')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(cax, label='Value')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def load_folds(fold_file):
    with open(fold_file, "r") as file:
        data = yaml.safe_load(file)
    return data


def filter_files(txt_files, m_ids, f_ids):
    filtred_files = []
    for file in txt_files:
        who = file.split('_')[3]
        gender = who[0]
        id = who[1:]
        if gender == 'M' and id in m_ids:
            filtred_files.append(file)
        elif gender == 'F' and id in f_ids:
            filtred_files.append(file)
    return filtred_files

def reshape_results(dict_of_results):
    results = {}
    for key, value in dict_of_results.items():
        l1_norm = value[0]
        l1_norm_median = value[1]
        t1_mean = value[2]
        t1_median = value[3]
        t2_mean = value[4]
        t2_median = value[5]
        key = str(key)
        results[key] = {}
        results[key]['l1_norm'] = l1_norm
        results[key]['l1_norm_median'] = l1_norm_median
        results[key]['t1_mean'] = t1_mean
        results[key]['t1_median'] = t1_median
        results[key]['t2_mean'] = t2_mean
        results[key]['t2_median'] = t2_median
    return results



def compute_avg_results(results, num_folds):
    avg_results = {}
    collected_results = {}
    for fold in results:
        for setting in results[fold]:
            if setting not in collected_results:
                collected_results[setting] = {
                    'l1_norm': [],
                    't1_mean': [],
                    't2_mean': []
                }
            collected_results[setting]['l1_norm'].append(results[fold][setting]['l1_norm'])
            collected_results[setting]['t1_mean'].append(results[fold][setting]['t1_mean'])
            collected_results[setting]['t2_mean'].append(results[fold][setting]['t2_mean'])

    # Calculate mean and standard deviation using numpy
    for setting, errors in collected_results.items():
        avg_results[setting] = {
            'l1_norm': np.mean(errors['l1_norm']),
            'l1_norm_std': np.std(errors['l1_norm'], ddof=0),
            't1_mean': np.mean(errors['t1_mean']),
            't1_mean_std': np.std(errors['t1_mean'], ddof=0),
            't2_mean': np.mean(errors['t2_mean']),
            't2_mean_std': np.std(errors['t2_mean'], ddof=0)
        }

    return avg_results

