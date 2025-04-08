from matplotlib import pyplot as plt
import os
import random
from tqdm import tqdm, trange


def create_kfold_splits(root_dir, extensions=('png', 'jpg', 'jpeg', 'bmp', 'tiff', 'pt', 'h5'), n_splits=10,
                        subsample=None):
    data = {'F': {}, 'M': {}}

    # Load all images and split into groups and subsets
    counter = 0
    for filename in tqdm(os.listdir(root_dir)):
        if filename.endswith(extensions):
            group = filename[0]
            if group in ['F', 'M']:
                identifier = ''
                for char in filename[1:]:
                    if char.isdigit():
                        identifier += char
                    else:
                        break

                identifier = int(identifier) if identifier else None
                # if identifier is not None and identifier > 20:
                #    continue

                if identifier is not None:
                    if identifier not in data[group]:
                        data[group][identifier] = []
                    data[group][identifier].append(filename)
                    counter += 1
                else:
                    print(f"Invalid identifier for {filename}")
    # Prepare list of identities for both groups
    f_ids = list(data['F'].keys())
    m_ids = list(data['M'].keys())

    # Shuffle identities for random splitting
    random.seed(42)
    random.shuffle(f_ids)
    random.shuffle(m_ids)

    # Split identities into n_splits folds
    f_fold_size = len(f_ids) // n_splits
    m_fold_size = len(m_ids) // n_splits

    f_folds = [f_ids[i * f_fold_size:(i + 1) * f_fold_size] for i in range(n_splits)]
    m_folds = [m_ids[i * m_fold_size:(i + 1) * m_fold_size] for i in range(n_splits)]

    # Adjust last fold to include any remaining identities
    f_folds[-1].extend(f_ids[n_splits * f_fold_size:])
    m_folds[-1].extend(m_ids[n_splits * m_fold_size:])

    # Generate the splits
    splits = []
    for i in trange(n_splits):
        train_files, val_files = {'F': {}, 'M': {}}, {'F': {}, 'M': {}}

        for j in range(n_splits):
            if j == i:
                for fid in f_folds[j]:
                    if fid not in val_files['F']:
                        val_files['F'][fid] = []
                    if subsample is None:
                        val_files['F'][fid].extend(data['F'][fid])
                    else:
                        val_files['F'][fid].extend(data['F'][fid][:subsample])
                for mid in m_folds[j]:
                    if mid not in val_files['M']:
                        val_files['M'][mid] = []
                    if subsample is None:
                        val_files['M'][mid].extend(data['M'][mid])
                    else:
                        val_files['M'][mid].extend(data['M'][mid][:subsample])
            else:
                for fid in f_folds[j]:
                    if fid not in train_files['F']:
                        train_files['F'][fid] = []
                    if subsample is None:
                        train_files['F'][fid].extend(data['F'][fid])
                    else:
                        train_files['F'][fid].extend(data['F'][fid][:subsample])
                for mid in m_folds[j]:
                    if mid not in train_files['M']:
                        train_files['M'][mid] = []
                    if subsample is None:
                        train_files['M'][mid].extend(data['M'][mid])
                    else:
                        train_files['M'][mid].extend(data['M'][mid][:subsample])

        splits.append((train_files, val_files))

    return splits


    # Prepare list of identities for both groups
    f_ids = list(data['F'].keys())
    m_ids = list(data['M'].keys())

    # Shuffle identities for random splitting
    random.seed(42)
    random.shuffle(f_ids)
    random.shuffle(m_ids)

    # Split identities into n_splits folds
    f_fold_size = len(f_ids) // n_splits
    m_fold_size = len(m_ids) // n_splits

    f_folds = [f_ids[i * f_fold_size:(i + 1) * f_fold_size] for i in range(n_splits)]
    m_folds = [m_ids[i * m_fold_size:(i + 1) * m_fold_size] for i in range(n_splits)]

    # Adjust last fold to include any remaining identities
    f_folds[-1].extend(f_ids[n_splits * f_fold_size:])
    m_folds[-1].extend(m_ids[n_splits * m_fold_size:])

    # Generate the splits
    splits = []
    for i in trange(n_splits):
        train_files, val_files = {'F': {}, 'M': {}}, {'F': {}, 'M': {}}

        for j in range(n_splits):
            if j == i:
                for fid in f_folds[j]:
                    if fid not in val_files['F']:
                        val_files['F'][fid] = []
                    if subsample is None:
                        val_files['F'][fid].extend(data['F'][fid])
                    else:
                        val_files['F'][fid].extend(data['F'][fid][:subsample])
                for mid in m_folds[j]:
                    if mid not in val_files['M']:
                        val_files['M'][mid] = []
                    if subsample is None:
                        val_files['M'][mid].extend(data['M'][mid])
                    else:
                        val_files['M'][mid].extend(data['M'][mid][:subsample])
            else:
                for fid in f_folds[j]:
                    if fid not in train_files['F']:
                        train_files['F'][fid] = []
                    if subsample is None:
                        train_files['F'][fid].extend(data['F'][fid])
                    else:
                        train_files['F'][fid].extend(data['F'][fid][:subsample])
                for mid in m_folds[j]:
                    if mid not in train_files['M']:
                        train_files['M'][mid] = []
                    if subsample is None:
                        train_files['M'][mid].extend(data['M'][mid])
                    else:
                        train_files['M'][mid].extend(data['M'][mid][:subsample])

        splits.append((train_files, val_files))

    return splits

def plot_loss_and_error(training_loss, training_error, validation_loss, validation_error, save_dir, split):
    plt.plot(training_loss, label='Training Loss')
    epochs = validation_loss.keys()
    values = validation_loss.values()
    plt.plot(epochs, values, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir, f"loss_{split}.png"))
    plt.clf()
    plt.plot(training_error, label='Training Error')
    epochs = validation_error.keys()
    values = validation_error.values()
    plt.plot(epochs, values, label='Validation Error')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.ylim(0, 0.6)
    plt.legend()
    plt.savefig(os.path.join(save_dir, f"error_{split}.png"))
    plt.clf()

def print_statistics(errors_train,errors,statistics_array):
    print("-" * 20)
    avg_err = torch.tensor(errors).mean().item()
    avg_train_err = torch.tensor(errors_train).mean().item()
    print(f'Average crossval validation error: {avg_err}')
    print(f'Average crossval training error: {avg_train_err}')
    print(f'Std crossval validation error : {torch.tensor(errors).std().item()}')
    print(f'Std crossval training error : {torch.tensor(errors_train).std().item()}')
    men_correct = sum(statistics.best_validation_matrix["M"]["Correct"] for statistics in statistics_array)
    men_incorrect = sum(statistics.best_validation_matrix["M"]["Incorrect"] for statistics in statistics_array)
    women_correct = sum(statistics.best_validation_matrix["F"]["Correct"] for statistics in statistics_array)
    women_incorrect = sum(
        statistics.best_validation_matrix["F"]["Incorrect"] for statistics in statistics_array)
    print("Men: Correct: " + str(men_correct) + " Incorrect: " + str(men_incorrect))
    print("Women: Correct: " + str(women_correct) + " Incorrect: " +
          str(women_incorrect))
    id_corect_incorrect = {}
    for statistics in statistics_array:
        for key in statistics.best_id_correct_incorrect_validation:
            if key not in id_corect_incorrect:
                id_corect_incorrect[key] = {"Correct": 0, "Incorrect": 0}
            id_corect_incorrect[key]["Correct"] += statistics.best_id_correct_incorrect_validation[key][
                "Correct"]
            id_corect_incorrect[key]["Incorrect"] += statistics.best_id_correct_incorrect_validation[key][
                "Incorrect"]
    id_corect_incorrect = statistics_array[0].sort_by_key(id_corect_incorrect)
    for key in id_corect_incorrect:
        print("ID: " + key + " Correct: " + str(id_corect_incorrect[key]["Correct"]) + " Incorrect: " + str(
            id_corect_incorrect[key]["Incorrect"]))
    print("-" * 20)
