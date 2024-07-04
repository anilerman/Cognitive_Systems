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
git clone https://github.com/Cognitive_System/ResNet18_K-fold_Data_Augmentation.git
cd ResNet18_K-fold_Data_Augmentation


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
    data:
      data_dir: 'data'
      batch_size: 32
      image_size: 224
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
      classes: ['Normal', 'Pneumonia']

    model:
      name: 'resnet18'
      pretrained: True

    training:
      epochs: 15
      learning_rate: 0.001
      seed: 42
      k_fold: 5

5. **Running the Project

    To train and evaluate the model, run:
    ```bash
    python run.py

6. **Results
    Accuracy: 0.9439
    Precision: 0.9455
    Recall: 0.9439
    F1 Score: 0.9433
    ROC AUC Score: 0.9303

7. **Accessing Plots
    You can access the generated plots at the following link:

    Results and Plots: https://drive.google.com/drive/folders/1-4l3idNk2s6Be4-GcxY5tvEJagBHTSV0

     Please use these following (username:deeplearningproject34@gmail.com)and (password:deeplearningTinaRohanAnil) if required.


8. **Explanation of Changes:
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

