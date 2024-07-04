# Pneumonia Detection with Deep Learning

## Overview

This project aims to detect pneumonia in chest X-ray images using transfer learning with models like ResNet and VGG-16. The pipeline includes data augmentation, handling class imbalance, and evaluation using various metrics.

## Project Structure

- `data_loader.py`: Contains code for loading and transforming the dataset.
- `model.py`: Defines the model architecture.
- `train.py`: Contains the training loop.
- `test.py`: Contains the testing loop.
- `inference.py`: Contains functions for running inference on new data.
- `visualization.py`: Contains functions for visualizing results.
- `utils.py`: Utility functions.
- `run.py`: Main script to run the project.
- `config.yaml`: Configuration file.

## Setup

### 1. Clone the Repository

Clone the repository to your local machine:

```bash
git clone https://github.com/Cognitive_System/ResNet18_Augmentation.git
cd ResNet18_Data_Augmentation


2. **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

3. **Set up the directory structure:**

    Ensure your dataset is organized as follows:

    ```
    data/
    ├── train/
    ├── val/
    └── test/
    ```
4. **Adjust paths in `config.yaml` as needed.** Below is an example configuration:

    ```yaml
    model:
  name: resnet18
  pretrained: true

data:
  train_data_dir: 'C:\Users\deepl\PycharmProjects2\pythonProject\Data_Raw\chest_xray\chest_xray\train'
  val_data_dir: 'C:\Users\deepl\PycharmProjects2\pythonProject\Data_Raw\chest_xray\chest_xray\val'
  test_data_dir: 'C:\Users\deepl\PycharmProjects2\pythonProject\Data_Raw\chest_xray\chest_xray\test'
  classes: [ 'NORMAL', 'PNEUMONIA' ]
  mean: [0.485, 0.456, 0.406]
  std: [0.229, 0.224, 0.225]


training:
  batch_size: 32
  checkpoint_dir: 'C:\Users\deepl\PycharmProjects\pythonProject\ResNet18\Resnet_templateproject_updated\checkpoints'
  learning_rate: 0.001
  num_epochs: 5
  seed: 42
  k_folds: 5


5. **Running the Project

    To train and evaluate the model, run:
    ```bash
    python run.py


6. **Accessing Plots for ResNet18 with k-fold
    You can access the generated plots at the following link:

    Results and Plots: https://drive.google.com/drive/folders/1-4l3idNk2s6Be4-GcxY5tvEJagBHTSV0

     Please use these following (username:deeplearningproject34@gmail.com)and (password:deeplearningTinaRohanAnil) if required.


7. **Explanation of Changes:
    1. **Added Better Structure and Formatting:**
    - Improved the readability of the setup instructions.
    - Added sections for better organization.
    - Highlighted important commands and instructions.

    2. **Added Example Usage:**
    - Included an example of how to call visualization functions in the `run.py` script.

    3. **Added Section for External Plots:**
    - Provided detailed instructions on how to add external plots from a personal computer.

    4. **Provided Link for Results:**
    - Included a link to access the results and plots with a placeholder for credentials if required.

