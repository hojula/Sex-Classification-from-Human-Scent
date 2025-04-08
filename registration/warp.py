from compound_position_detection import Settings

import matplotlib
import os
import torch
from omegaconf import OmegaConf
from help_functions import set_system
from help_functions import from_times_pixels
from help_functions import find_positions
from help_functions import load_compound_and_times
from help_functions import load_tensor_and_system
from help_functions import find_positions_cnn
from einops import rearrange
from scipy.interpolate import LinearNDInterpolator
import torch.nn.functional as F
import cv2

import numpy as np
import warnings
from help_functions import apply_color_map
import yaml


def warp_tensor(reference_tensor, tensor_to_warp, reference_compound_pixels, compounds_pixels, name,
                save_dir=None, save_grids=False, suffix=None):
    reference_tensor = reference_tensor.cpu().numpy()
    tensor_to_warp = tensor_to_warp.cpu().numpy()
    C, H, W = tensor_to_warp.shape
    t1_ref, t2_ref = [], []
    t1, t2 = [], []
    t1_ref_shift, t2_ref_shift = [], []
    compound_order = []
    for compound in compounds_pixels:
        if compound not in reference_compound_pixels:
            continue
        compound_order.append(compound)
        reference_t1 = reference_compound_pixels[compound][0]
        reference_t2 = reference_compound_pixels[compound][1]
        t1_ref.append(reference_t1)
        t2_ref.append(reference_t2)
        measure_t1 = compounds_pixels[compound][0].cpu()
        measure_t2 = compounds_pixels[compound][1].cpu()
        t1.append(measure_t1)
        t2.append(measure_t2)
        t1_shift = reference_t1 - measure_t1
        t2_shift = reference_t2 - measure_t2
        t1_ref_shift.append(t1_shift)
        t2_ref_shift.append(t2_shift)

    def preproces_tensor_to_tic(tensor):
        tensor = tensor.sum(0, keepdims=True).transpose((1, 2, 0))
        # print('Check if tensor is the same', tensor_to_warp is tensor)
        tensor = tensor - tensor[tensor > 0].min()
        tensor[tensor < 0] = 0
        tensor = np.log(tensor + 1e-10)
        min_grather_zero = tensor[tensor > 0].min()
        tensor[tensor < 0] = min_grather_zero
        tensor = tensor / tensor.max()
        tensor = apply_color_map(tensor)
        tensor = np.squeeze(tensor)
        return tensor

    tic_to_warp = preproces_tensor_to_tic(tensor_to_warp)
    for i in range(len(t1)):
        cv2.circle(tic_to_warp, (int(t1[i]), int(t2[i])), radius=4, color=(255, 0, 0), thickness=1)

    reference_tic = preproces_tensor_to_tic(reference_tensor)
    c = 0
    for i in range(len(t1_ref)):
        # print("Reference", t1_ref[i], t2_ref[i])
        cv2.circle(reference_tic, (int(t1_ref[i]), int(t2_ref[i])), radius=4, color=(0, 0, 255), thickness=1)
        c += 1
    # print("Number of compounds", c)

    t1_ref.extend([0, 0, W - 1, W - 1])
    t2_ref.extend([0, H - 1, H - 1, 0])
    t1.extend([0, 0, W - 1, W - 1])
    t2.extend([0, H - 1, H - 1, 0])

    t1_ref, t2_ref = np.array(t1_ref), np.array(t2_ref)
    t1, t2 = np.array(t1), np.array(t2)
    x_coords, y_coords = np.meshgrid(np.linspace(0, W, W), np.linspace(0, H, H))
    x_coords_flat = x_coords.flatten()
    y_coords_flat = y_coords.flatten()

    reference_points = np.vstack((t1_ref, t2_ref)).T  # Original grid points
    new_positions = np.vstack((t1, t2)).T  # Modified grid points

    # Interpolate new x and y coordinates for the whole grid
    interp_x = LinearNDInterpolator(reference_points, new_positions[:, 0])
    interp_y = LinearNDInterpolator(reference_points, new_positions[:, 1])

    # Interpolated coordinates for the entire grid
    new_x_coords_flat = interp_x(x_coords_flat, y_coords_flat)
    new_y_coords_flat = interp_y(x_coords_flat, y_coords_flat)

    # Create new grid with normalized coordinates for grid_sample
    new_grid = np.stack([new_x_coords_flat, new_y_coords_flat], axis=1)
    new_grid = new_grid.reshape(H, W, 2)
    new_grid = torch.tensor(new_grid, dtype=torch.float32).unsqueeze(0)

    # Normalize grid to [-1, 1] range
    new_grid[..., 0] = (new_grid[..., 0] / (W - 1)) * 2 - 1
    new_grid[..., 1] = (new_grid[..., 1] / (H - 1)) * 2 - 1

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    new_grid = new_grid.to(device)

    if save_grids:
        grid_dir = os.path.join(save_dir, 'grids')
        if not os.path.exists(grid_dir):
            os.makedirs(grid_dir)
        torch.save(new_grid, os.path.join(grid_dir, name + '.pt'))
        print("Saved grid to", os.path.join(grid_dir, name + '.pt'))

    def warp_tensor(tensor_to_warp, new_grid):
        tensor_to_warp_t = torch.tensor(tensor_to_warp, device=device)[None].float()
        warped_tensor = F.grid_sample(tensor_to_warp_t, new_grid, mode='bilinear', align_corners=True)
        warped_tensor = warped_tensor.squeeze(0).cpu().numpy()
        warped_tensor = np.nan_to_num(warped_tensor, nan=0)
        return warped_tensor

    tic_to_warp = tic_to_warp.transpose((2, 0, 1))
    # reference_tic = reference_tic.transpose((2, 0, 1))
    warped_tensor = warp_tensor(tensor_to_warp, new_grid)
    tic_to_warp = warp_tensor(tic_to_warp, new_grid).transpose((1, 2, 0)).astype(np.uint8)
    # reference_tic = warp_tensor(reference_tic, new_grid).transpose((1, 2, 0)).astype(np.uint8)

    reference_tic = reference_tic[::-1]
    tic_to_warp = tic_to_warp[::-1]

    def scale_t1(viz):
        result = np.zeros((viz.shape[0], 5 * viz.shape[1], 3))
        for i in range(5):
            result[:, i::5, :] = viz
        return result

    # make all same size)
    if reference_tic.shape[1] > tic_to_warp.shape[1]:
        larger = np.zeros_like(reference_tic)
        h, w, c = tic_to_warp.shape
        larger[:h, :w] = tic_to_warp
        tic_to_warp = larger
        larger = np.zeros_like(reference_tensor)
        larger[:, :h, :w] = warped_tensor
        warped_tensor = larger
    elif reference_tic.shape[1] < tic_to_warp.shape[1]:
        smaller = np.zeros_like(reference_tic)
        h, w, c = reference_tic.shape
        smaller[:, :] = tic_to_warp[:, :h, :w]
        reference_tic = smaller
        smaller = np.zeros_like(warped_tensor)
        smaller[:, :, :] = warped_tensor[:, :h, :w]
        warped_tensor = smaller

    reference_tic = scale_t1(reference_tic)
    tic_to_warp = scale_t1(tic_to_warp)
    # blend with original image
    blended = cv2.addWeighted(reference_tic, 0.5, tic_to_warp, 0.5, 0)
    save_dir = os.path.join(save_dir, 'viz')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(os.path.join(save_dir, 'blend')):
        os.makedirs(os.path.join(save_dir, 'blend'))
    if not os.path.exists(os.path.join(save_dir, 'warp')):
        os.makedirs(os.path.join(save_dir, 'warp'))
    cv2.imwrite(os.path.join(save_dir, 'blend', name + '_blended.png'), blended[..., ::-1])
    # cv2.imwrite(os.path.join(save_dir, 'reference.png'), reference_tic[..., ::-1])
    cv2.imwrite(os.path.join(save_dir, 'warp', name + '_warped.png'), tic_to_warp[..., ::-1])
    return torch.from_numpy(warped_tensor)


def find(file, spectrogram_image=None, t1_shift=-500, load_dir=None, config=None, settings=None, fold=None,
         cnn_model=None, reference_spectra=None, compound_numbers_cnn=None,
         all_compounds_reference_positions_pixels=None):
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
                                                                           compound_numbers_cnn)
    else:
        compounds_pixels = find_positions_cnn(spectrogram_image, cnn_model, reference_spectra, compound_numbers_cnn)
    return compounds_pixels


def process_one_file(file, settings, t1_shift, dir_txt, dir_pt, config, fold, cnn_model, reference_spectra,
                     reference_positions, reference_tensor,
                     save_dir=os.path.join(os.getcwd(), 'warped_tensors'), compound_numbers_cnn=None):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_dir_registered = os.path.join(save_dir, 'registered')
    if not os.path.exists(save_dir_registered):
        os.makedirs(save_dir_registered)
    if not os.path.exists(os.path.join(save_dir_registered, 'without_banned')):
        os.makedirs(os.path.join(save_dir_registered, 'without_banned'))
    if not os.path.exists(os.path.join(save_dir_registered, 'with_banned')):
        os.makedirs(os.path.join(save_dir_registered, 'with_banned'))
    print("Processing", file)
    banned_compounds = ['Heptacosane', 'Nonacosane']
    reference_positions_without_banned = {k: v for k, v in reference_positions.items() if k not in banned_compounds}
    spectrogram_image, system_number = load_tensor_and_system(file, t1_shift, dir_path=dir_pt)
    all_compounds_reference_positions_pixels, rest = set_system(system_number, fold=fold)
    for i, set in enumerate(settings):
        compounds_position = find(file, spectrogram_image=spectrogram_image, t1_shift=t1_shift,
                                  load_dir=dir_pt, config=config,
                                  settings=set, fold=fold, cnn_model=cnn_model,
                                  reference_spectra=reference_spectra,
                                  compound_numbers_cnn=compound_numbers_cnn,
                                  all_compounds_reference_positions_pixels=all_compounds_reference_positions_pixels)
        compounds_position_without_banned = {k: v for k, v in compounds_position.items() if k not in banned_compounds}
        warped_tensor = warp_tensor(reference_tensor, spectrogram_image, reference_positions,
                                    compounds_position,
                                    file.replace('.pt', ''), save_dir=save_dir, save_grids=False,
                                    suffix=set.metric + '_all')
        expected_shape = (801, 2000, 497)
        assert warped_tensor.shape == expected_shape, f"Expected shape {expected_shape}, got {warped_tensor.shape}"
        torch.save(warped_tensor, os.path.join(save_dir_registered+'/with_banned', file))
        warped_tensor = warp_tensor(reference_tensor, spectrogram_image,
                                    reference_positions_without_banned,
                                    compounds_position_without_banned,
                                    file.replace('.pt', ''),
                                    save_dir=save_dir,
                                    save_grids=False,
                                    suffix=set.metric + '_without_banned')
        expected_shape = (801, 2000, 497)
        assert warped_tensor.shape == expected_shape, f"Expected shape {expected_shape}, got {warped_tensor.shape}"
        torch.save(warped_tensor, os.path.join(save_dir_registered+'/without_banned', file))


def procces_all_files(files, dir_txt, dir_pt, config, fold, t1_shift=-500, settings_array=None, channels=801):
    with open(config.constants.compound_numbers_cnn_path, 'r') as file:
        compound_numbers_cnn = yaml.load(file, Loader=yaml.FullLoader)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    reference_spectra = torch.zeros(len(compound_numbers_cnn), channels, device=device)
    for compound in compound_numbers_cnn:
        reference = torch.load(config.constants.folds_references_dir + 'fold' + str(fold) + '/' + compound + '.pt').to(
            device)
        compound_number = compound_numbers_cnn[compound]
        reference = rearrange(reference, 'l h w -> (l h w)')
        reference_spectra[compound_number] = reference
    cnn_model = torch.load(config.constants.folds_references_dir + 'fold' + str(fold) + '/cnn/model_fcn.pt').to(device)
    reference_positions = load_reference_positions(dir_txt=dir_txt)
    reference_tensor = torch.load(os.path.join(dir_pt, config.constants.reference_sample)).to(device)
    print("Reference tensor and positions loaded")
    #files = files[:2]
    for file in files:
        process_one_file(file, settings_array, t1_shift, dir_txt, dir_pt, config, fold, cnn_model, reference_spectra,
                         reference_positions, reference_tensor, compound_numbers_cnn=compound_numbers_cnn,
                         save_dir=os.path.join(config.constants.save_dir, 'warped_tensors'))


def set_settings(metric='dot'):
    if metric == 'dot':
        settings = Settings(5, 50, 'dot', True, lambda x: x ** 4, True)
    elif metric == 'cnn':
        settings = Settings(None, None, 'cnn', None, None, True)
    return [settings]


def load_reference_positions(reference_file='system_1_target_F4_05.txt', reference_system_number=1, t1_shift=-500,
                             dir_txt=None):
    compounds_times = load_compound_and_times(reference_file, dir_txt)
    _, rest = set_system(reference_system_number, fold=1)
    x_six_seconds, y_six_seconds, x_eight_seconds, y_eight_seconds, x_ten_seconds, y_ten_seconds = rest
    reference_compounds_pixels = from_times_pixels(compounds_times, int(reference_system_number),
                                                   x_six_seconds,
                                                   y_six_seconds,
                                                   x_eight_seconds, y_eight_seconds, x_ten_seconds,
                                                   y_ten_seconds,
                                                   t1_shift)
    return reference_compounds_pixels

def main():
    warnings.filterwarnings("ignore")
    config = OmegaConf.load('config.yaml')
    files = os.listdir(config.constants.dir_pt)
    files.remove(config.constants.reference_sample)
    settings_array = set_settings(metric=config.constants.warp_metric)
    procces_all_files(files, config.constants.dir_txt, config.constants.dir_pt,
                      config, config.constants.fold_num, settings_array=settings_array,
                      t1_shift=config.constants.t1_shift)


if __name__ == '__main__':
    main()

