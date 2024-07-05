
### README

# DenseNet Model Training and Evaluation

This repository contains a series of Jupyter notebooks for training, tuning, and evaluating DenseNet models. Each notebook focuses on different aspects of the model development process, including data augmentation, hyperparameter tuning, and model evaluation using confusion matrices.

## Notebooks Overview

1. **DenseNet_With_Aug_ConfusionMatrix.ipynb**
   - **Description**: This notebook covers the training of a DenseNet model with data augmentation and includes the generation of a confusion matrix to evaluate the model's performance.
   - **Parameters**: 
     - `batch_size`: 32
     - `gamma`: 0.1
     - `learning_rate (lr)`: 0.01
     - `momentum`: 0.8
     - `step_size`: 10

2. **DenseNet_Hypertunning.ipynb**
   - **Description**: This notebook is dedicated to hyperparameter tuning of the DenseNet model. Various configurations are tested to find the optimal set of hyperparameters for the best model performance.

3. **DenseNet_V2.ipynb**
   - **Description**: The second version of the DenseNet model training. This notebook may include improvements or modifications based on the results obtained from the initial training and hyperparameter tuning.

4. **DenseNet_Without_Aug_ConfusionMatrix.ipynb**
   - **Description**: This notebook focuses on training the DenseNet model without data augmentation. It also includes a confusion matrix to evaluate the model's performance under these conditions.

5. **DenseNet_Best_Model_Save.ipynb**
   - **Description**: This notebook is used to save the best performing DenseNet model. The model is saved after evaluating its performance using various metrics.

## Getting Started

To run the notebooks, follow these steps:

1. **Clone the repository**:
    ```sh
    git clone <repository_url>
    cd <repository_name>
    ```

2. **Install the required dependencies**:
    ```sh
    pip install tensorflow
    pip install numpy
    pip install matplotlib
    pip install scikit-learn
    pip install jupyter
    ```

3. **Open the Jupyter notebooks**:
    ```sh
    jupyter notebook
    ```

4. **Run the notebooks**: Open each notebook in the Jupyter interface and run the cells to train, tune, and evaluate the DenseNet models.

## Dependencies

- Python 3.8
- Jupyter Notebook
- TensorFlow / Keras
- NumPy
- Matplotlib
- Scikit-learn

## Results and Evaluation

Each notebook includes detailed descriptions and visualizations of the training process, model performance metrics, and confusion matrices. These provide insights into how the model performs under different conditions and configurations.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes. Ensure that your code follows the established coding standards and includes appropriate documentation.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

This README provides an overview of the notebooks and instructions on how to use them for training and evaluating DenseNet models. Each notebook is self-contained with detailed comments and explanations to guide you through the process.
