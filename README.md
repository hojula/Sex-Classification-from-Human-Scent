# Compound Position Detection and Warping

## Overview
This project is structured into two main tasks, each in separate subdirectories with their own configurations:

1. **Registration** (`registration/`): 
   - **Compound Position Detection** (`compound_position_detection.py`): Detects compound positions from input measurements.
   - **CNN Training for Detection** (`train_network_compound_position_detection.py`): Trains a CNN to improve the detection accuracy of compound positions.
   - **Warping to Canonical Frame** (`warp.py`): Aligns measurements using detected positions.
   - **Additional Files**: Includes `settings_compound_position_detection.py`, `help_functions.py`, `match_txt_pt.py`, and other YAML configuration files.
   - **Configuration**: `registration/config.yaml` and other YAML configuration files.

2. **Identification** (`classification/`): 
   - **Sex Classification** (`train_network_classification.py`): Trains a model to classify sex based on registered measurements. 
   - **Job Submission** (`submit.py`): Manages execution inside a Singularity container.
   - **Singularity Execution** (`rci_job.sh`): Runs the job on SLURM.
   - **Singularity Build** (`singularity_build.sh`): Builds the container.
   - **Additional Files**: Includes additional YAML configuration files required for identification.
   - **Configuration**: `identification/config.yaml` and other YAML configuration files.

The **Sex Classification task** runs **only within a Singularity container** and must be executed via `submit.py`.

---
## Installation & Requirements
### Data download
Moq data can be downloaded from the following link: [Data](https://drive.google.com/file/d/1zdGoYDdVfvjwzLeZuqGdK6n_2W_eT6rc/view?usp=drive_link)
### Registration Task
To replicate the exact environment for **registration**, use the Conda environment file:
```sh
conda env create -f environment.yaml
conda activate <env-name>
```
**Note:** This Conda environment is only for **registration** and does not apply to the **identification** task.

### Classification Task
Classification task runs **exclusively** inside a Singularity container. Ensure the container is built before running the task.

---
## Usage
### 1. **Compound Position Detection (Registration Task)**
To detect compound positions, run:
```sh
python registration/compound_position_detection.py
```
Settings for compound position detection can be modified in:
```sh
registration/settings_compound_position_detection.py
```

### 2. **Train CNN for Compound Detection**
To train the CNN model for improving detection, run:
```sh
python registration/train_network_compound_position_detection.py
```
This script loads data, applies k-fold validation, and trains the model.

### 3. **Warp Measurements to Canonical Frame**
To warp measurements based on detected positions, run:
```sh
python registration/warp.py
```
This script aligns detected compound positions to a canonical reference frame.

### 4. **Sex Classification (Classification Task - Requires Singularity)**
#### Step 1: Configure `singularity_cu.def`
Before building the Singularity container, ensure the following line is present in the `singularity_cu.def` file to include required dependencies:
```
%files
    /home/example_user/example_project/Pipfile /opt/proj/Pipfile
```

#### Step 2: Build the Singularity Container
Once the `singularity_cu.def` file is correctly configured, build the container using:
```sh
bash classification/singularity_build.sh
```
The `singularity_build.sh` file **must include the following settings**:
```sh
#!/bin/bash

PROJECT_DIR="/mnt/example/path/to/project"

singularity build "${PROJECT_DIR}/example_container.sif" "singularity_cu.def"
```

#### Step 3: Ensure Correct SLURM Job Settings
Before running the classification task, set the following in `rci_job.sh`:
```sh
#set your working directory
cd "/home/example_user/example_project/classification/" || exit

#set your python path
export PYTHONPATH="/home/example_user/example_project/:${PYTHONPATH}"

#set your singularity image name
SINGULARITY_IMAGE_NAME="example_singularity"
```

#### Step 4: Run the Task Inside the Container
Once the SLURM job is correctly set, submit and execute the classification task by running:
```sh
python classification/submit.py
```
This script will automatically execute within the Singularity container and use SLURM for scheduling.

---
## Configuration Files
Each task has its own configuration file:
- **Registration Task**: `registration/config.yaml` (and additional YAML configuration files in `registration/`)
- **Classification Task**: `classification/config.yaml` (and additional YAML configuration files in `classification/`)

### Configuration File Details
#### **`registration/config.yaml`**
```yaml
constants:
  dir_txt: '/path/to/example/txt'          # Directory containing text files with compound data
  dir_pt: '/path/to/example/pt'            # Directory containing preprocessed data files
  num_folds: 5                    # Number of folds for cross-validation
  warp_metric: 'dot'              # Metric used for warping
  folds_references_dir: '/path/to/example/folds/'  # Directory containing fold-specific references
```
- **`folds_references_dir`**: This directory **must** contain subdirectories structured as follows:
```
/path/to/example/folds/
  ├── fold{num}/                     # Contains reference compound spectra for each fold
  │   ├── cnn/
  │   │   ├── model_fcn.py           # Trained CNN model
  │   │   ├── (other model files)
  │   ├── svm/l2/
  │   │   ├── model_c100.pt          # Trained SVM model
```

#### **`classification/config.yaml`**
```yaml
batch_size: 16                     # Number of samples per batch
root_dir: '/path/to/example/root'
save_dir: '/path/to/example/models/'
plots_dir: '/path/to/example/plots/'
data_dir: '/path/to/example/warped/data/'  # Path to registered data
affine_shift_max_px: 5              # Maximum affine shift applied during training
subsample: -1                       # Use -1 to include all data
num_folds: 10                       # Number of folds for training
training_params:
    learning_rate_model: 3e-4       # Learning rate for model training
    epochs: 100                     # Number of training epochs
    plot_after: 10                  # Generate plots after this many epochs
    save_after: 10                  # Save model after this many epochs
    valid_after: 1                   # Validate after each epoch
    sampler_size: 1                  # Number of samples per identity in balanced training
```

---
## Citation & License
This project is licensed under the Apache-2.0 License. You may use, distribute, and modify this software under the terms of the Apache License, Version 2.0.
A copy of the license can be found in the LICENSE file or at: [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)

---
## Authors & Acknowledgments
Developed by **Jan Hlavsa**, **Ing. Bc. Radim Špetlík**, and **prof. Ing. Jiří Matas, Ph.D.** at the **Czech Technical University in Prague**.

Special thanks to contributors and libraries used in this project.
