Chest X-Ray Classification with AlexNet
This repository contains a deep learning project for classifying chest X-ray images using the AlexNet architecture. The project includes data preprocessing, model training, evaluation, and visualization of results.

Table of Contents
Installation
Usage
Project Structure
Model Training
Evaluation
Visualization
Contributing
License
Installation
Clone the repository:

sh
Copy code
git clone https://github.com/anilerman/Cognitive_Systems.git
cd chest-xray-classification
Create a virtual environment and activate it:

sh
Copy code
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
Install the required packages:

sh
Copy code
pip install -r requirements.txt
Usage
Ensure your dataset is organized into train, val, and test directories.
Update the dataset paths in the main.py file:
python
Copy code
train_dir = 'path/to/train'
val_dir = 'path/to/val'
test_dir = 'path/to/test'
Run the main script to train and evaluate the model:
sh
Copy code
python main.py
Project Structure
css
Copy code
chest-xray-classification/
├── data_loader.py
├── model.py
├── train.py
├── test.py
├── inference.py
├── graph.py
├── histogram_equalization.py
├── contrast_enhancement.py
├── main.py
├── requirements.txt
└── README.md
data_loader.py: Contains functions to load and preprocess the data.
model.py: Defines the AlexNet model architecture.
train.py: Contains the training loop and learning rate scheduler.
test.py: Evaluates the model on the test set.
inference.py: Functions for model inference and metric calculations.
graph.py: Functions for plotting training history, predictions, and confusion matrix.
histogram_equalization.py: Implements histogram equalization for image preprocessing.
contrast_enhancement.py: Implements contrast enhancement for image preprocessing.
main.py: Main script to run the training and evaluation.
requirements.txt: Lists the required Python packages.
README.md: Project documentation.
Model Training
The train.py script trains the AlexNet model on the provided dataset. The script performs the following steps:

Loads the dataset using get_data_loaders from data_loader.py.
Defines the loss function and optimizer.
Trains the model for a specified number of epochs.
Saves the best model based on validation accuracy.
Example training command:

sh
Copy code
python train.py
Evaluation
The test.py script evaluates the trained model on the test set and calculates various metrics such as accuracy, precision, recall, F1-score, and ROC-AUC.

Example evaluation command:

sh
Copy code
python test.py
Visualization
The graph.py script provides functions to visualize the training history, model predictions, and confusion matrix.

plot_training_history: Plots the training and validation loss and accuracy.
plot_predictions: Visualizes sample predictions from the model.
plot_confusion_matrix: Plots the confusion matrix for the test set.
Contributing
Contributions are welcome! Please submit a pull request or open an issue to discuss any changes.

License
This project is licensed under the MIT License. See the LICENSE file for details.