# MNIST Digit Classification using Machine Learning 
![image](https://github.com/deeparsh7/DigitClassification/assets/121679549/a9ca4a99-b7d8-48fd-9eb7-684bbf624ead)


This repository contains a simple image classification project that demonstrates the process of building and training a machine learning model to classify handwritten digits from the MNIST dataset.

## Project Overview

In this project, we utilize the TensorFlow library to create an image classification model. The goal is to accurately classify images of handwritten digits into their respective classes (0-9). We follow these main steps:

1. **Loading and Preprocessing Data**: The MNIST dataset is loaded using TensorFlow's built-in dataset module. The pixel values of the images are normalized to a range between 0 and 1, and the images are reshaped to match the expected input shape of the model.

2. **Model Architecture**: We build a simple convolutional neural network (CNN) using TensorFlow's Sequential model. The model consists of convolutional layers, max-pooling layers, and dense layers to learn and classify the digit images.

3. **Model Training**: The model is compiled using the Adam optimizer and the sparse categorical cross-entropy loss function. It is then trained on the training data for a specified number of epochs. After training, the model's accuracy is evaluated on the test data.

## Instructions to Run the Code

To run the code in this repository and reproduce the results, follow these steps:

1.Go to the GitHub repository containing the code: [https://github.com/your_username/mnist-digit-classification](https://github.com/deeparsh7/DigitClassification)

2.Click on the mnist_classification.ipynb notebook file.

3.On the top-left corner of the notebook page, you'll see a "Open in Colab" button. Click on it to open the notebook in Google Colab.

4.Open in Colab

5.Google Colab will open the notebook in a new tab. 

6. Observe the output of each cell to track the progress and results of the training process.

## Functionality

The main functionality of this project revolves around building, training, and evaluating an image classification model using the MNIST dataset. Here's a breakdown of how the project works:

1. **Loading and Preprocessing Data**: The MNIST dataset, comprising images of handwritten digits, is loaded using TensorFlow. The pixel values of the images are normalized to a range between 0 and 1, and the images are reshaped to match the expected input shape of the model.

2. **Model Architecture**: A convolutional neural network (CNN) model is constructed using TensorFlow's Sequential API. The model consists of convolutional layers, max-pooling layers, and dense layers. This architecture enables the model to learn and extract relevant features from the digit images.

3. **Model Training**: The model is compiled using the Adam optimizer and the sparse categorical cross-entropy loss function. It is then trained on the training data for a specified number of epochs. During training, the model learns to make accurate predictions by adjusting its internal parameters.

4. **Evaluation and Results**: After training, the model's accuracy is evaluated on a separate test dataset. The test accuracy provides insight into the model's generalization performance on unseen data. The achieved accuracy is showcased, allowing you to assess the effectiveness of the trained model.

## Project Results

After training the model on the MNIST dataset, you can expect to achieve a test accuracy of around [Achieved accuracy here - 98%].

## Time and Space Complexity

In terms of efficiency, the **time complexity** of the model training primarily depends on the number of epochs and the size of the dataset. 

As for the **space complexity**, it is determined by the number of model parameters and intermediate tensors created during the forward and backward passes.

 While both time and space complexity are important considerations, in this project, achieving a high accuracy is usually the primary focus.

## Conclusion

This project serves as a beginner-friendly example of building an image classification model using machine learning techniques. 

## Contact
as7112@srmist.edu.in,arshdeep72140@gmail.com

