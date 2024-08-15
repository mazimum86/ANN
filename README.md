# Artificial Neural Network (ANN)

This repository contains the implementation of an Artificial Neural Network (ANN) for solving a classification problem. ANNs are powerful machine learning models inspired by the human brain and are widely used for various tasks, including image recognition, speech processing, and more.

## Contents

- **ann.py**: The main script that implements the ANN model using TensorFlow/Keras.
- **data.csv**: The dataset used to train and test the ANN model.
- **model.h5**: The saved trained model.
- **results/**: A directory containing visualizations and performance metrics of the ANN model.
- **requirements.txt**: A list of Python dependencies required to run the code.

## Implementation Details

The implementation includes the following key steps:

1. **Data Preprocessing**: 
    - Loading and splitting the dataset (`data.csv`) into training and testing sets.
    - Feature scaling to normalize the input data for better convergence.

2. **Building the ANN Model**:
    - The ANN is built using TensorFlow/Keras, consisting of input, hidden, and output layers.
    - The hidden layers use activation functions like ReLU, while the output layer uses softmax (for multi-class classification) or sigmoid (for binary classification).

3. **Training the Model**:
    - The model is trained using a specified number of epochs and a batch size.
    - During training, the model's performance is evaluated using metrics like accuracy and loss.

4. **Evaluating the Model**:
    - After training, the model is evaluated on the test set.
    - The confusion matrix, accuracy score, and loss curves are generated to assess performance.

5. **Saving the Model**:
    - The trained model is saved as `model.h5` for future use.

## Usage

To use this code, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/mazimum86/ANN.git
    ```
2. Navigate to the repository directory:
    ```bash
    cd ANN
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. Run the ANN model training and evaluation:
    ```bash
    python ann.py
    ```

## Dependencies

The following Python packages are required to run the code:

- tensorflow
- keras
- numpy
- pandas
- matplotlib
- scikit-learn

These dependencies are listed in the `requirements.txt` file and can be installed using `pip`.

## Results

The output includes:

- **Confusion Matrix**: A matrix displaying the classification accuracy across classes.
- **Accuracy Score**: The overall accuracy of the model on the test set.
- **Loss Curves**: Plots showing the loss reduction over training epochs.

## Contributing

Contributions are welcome! If you have suggestions for improvements or additional features, feel free to submit a pull request or open an issue.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
