# Implementing a Feed Forward Neural Network (FFNN) using Keras and TensorFlow to classify images of handwritten Arabic characters

## Part 1: Data Preprocessing
### Dataset:

The dataset consists of Arabic handwritten characters, with images in 28x28 pixel resolution.
The dataset contains:

**TrainImgs.csv**: Training images.

**TrainLabels.csv**: Corresponding labels for training images.

**TestImgs.csv**: Testing images.

**TestLabels.csv**: Corresponding labels for testing images.

The data should be reshaped to 32x32 pixels and normalized to a scale of [0, 1].
### Data Preparation:

Split the dataset into training and testing sets.

Apply One-Hot Encoding to the labels to make them suitable for classification.

## Part 2: Feed Forward Neural Network (FFNN) Implementation
### Model Construction:

Implement the Feed Forward Neural Network (FFNN) using Keras and TensorFlow.

Design a simple FFNN architecture, where the input layer processes the flattened image data (1D array from 2D images), and subsequent layers consist of fully connected (dense) layers.

### Training Configuration:

Use Stochastic Gradient Descent (SGD) as the optimizer for training.

Key hyperparameters:

**Learning Rate**: 0.005

**Epochs**: 100

**Batch Size**: 32

**Activation Functions**: Use ReLU for hidden layers and Softmax for the output layer.

The weights should be initialized using normal or uniform distribution.

## Part 3: Model Training and Evaluation
### Training Process:

Train the FFNN on the preprocessed training data using SGD and monitor the loss and accuracy during each epoch.

Plot the accuracy and loss over epochs for both the training and validation datasets.

### Testing and Optimization:

Test the trained model on the testing dataset and evaluate its performance using metrics like accuracy, precision, recall, and F1-score.

Experiment with different batch sizes (16, 32, 64, 128) and compare the performance.

Tweak the learning rate, activation functions, and weight initialization strategies to optimize model performance.

## Part 4: Advanced Techniques
### Optimizers:

Experiment with different optimizers, such as Adam with varying momentum parameters (0.5, 0.9, 0.98).

Compare the results of using Adam with SGD and analyze their effect on the model’s convergence and performance.
### Regularization:

Implement Dropout (with a dropout rate of 0.1) to prevent overfitting and improve generalization.

Use regularization techniques to fine-tune the model and avoid overfitting.
