# Multi level perceptron neural network
### _Neural network model built to model pan evaporation at a site in San Diego, USA (1987-1990)_

## Table of Contents

- [Introduction](#introduction)
- [Data Pre-processing](#data-pre-processing)
  - [Handling Non-Numeric & Missing Values](#handling-non-numeric--missing-values)
  - [Outlier Removal](#outlier-removal)
  - [Seasonality Removal](#seasonality-removal)
  - [Data Normalization](#data-normalistation)
  - [Identifying Relevant Predictors](#identifying-relevant-predictors)
  - [Data Splitting](#data-splitting)
- [Multi Layer Perceptron Implementation](#multi-later-perceptron-implementation)
  - [Network Structure](#network-structure)
  - [Technologies and Libraries](#technologies-and-libraries)
  - [Backpropagation Implementation](#backpropagation-implementation)
  - [Network Training](#network-training)
  - [Network Evaluation](#network-evaluation)

## Introduction
Given a set of data monitoring environmental factors at a site in San Diego, the aim of this project is to produce a neural network without any machine learning libraries capable of making predictions for pan evaportaion given the set of input variables. This project was completed as part of my coursework for my 2nd year AI methods module at University.

## Data pre-processing
Data pre-processing is essential for preparing raw data for usage within the multi-level perceptron (MLP), neural network, so by improving data quality and addressing any issues with the data, more accurate and reliable predictions may be obtained from the MLP. In this section I will outline all steps that I undertook to prepare the data on daily environmental factors, for usage in my network.

### Handling non numeric & Missing values
Firstly, to plot the initial data, a number of steps must be taken to resolve any issues that may arise due to the presence of null values. As seen below, any rows that contain non-numeric values are replaced with null values. Next, all rows that contain null values will be removed, which leaves only the rows containing clean data.

<p align="center">
  <img src="https://github.com/NeilVaria/Neural-network/assets/60001894/389691b4-ef2d-4543-b4f5-16e1fe028d4e" alt="Handling Non-Numeric & Missing Values" height="200" width="auto">
</p>

<p align="center">
  <img src="https://github.com/NeilVaria/Neural-network/assets/60001894/70b5e1a9-a021-4aba-89ff-2fc74db2cebf" alt="Handling Non-Numeric & Missing Values">
</p>

In the dataset provided, there is an index that will cause unnecessary data loss (see above), as lines 2-7, 9 and 11 contain non-numeric values (the index) and therefore will be removed. I manually removed the index from the dataset, to preserve the rows that would have been removed. After removing all non-numeric values and any rows that contain missing data, the initial dataset can be plotted on a line graph:

<p align="center">
  <img src="https://github.com/NeilVaria/Neural-network/assets/60001894/3b782284-1286-4268-b971-ab0ab8d34917" alt="Line Graph">
</p>


### Outlier removal
Outliers need to be removed from the dataset as they can skew a models understanding of underlying data distribution. I decided to use the Interquartile range method (IQR) method as seen in the following snippet, due to its lower sensitivity to extreme values, as apposed to a method such as the standard deviation method which is dependant on the mean. 

<p align="center">
  <img src="https://github.com/NeilVaria/Neural-network/assets/60001894/27ffb661-b261-4af9-91a3-9b99d448a3eb" alt="Outlier Removal" height="200" width="auto">
</p>


Since the mean can be heavily influenced by extreme values, some outliers may not be removed using the standard deviation method. After removing outliers, the dataset can be plotted as follows:

<p align="center">
  <img src="https://github.com/NeilVaria/Neural-network/assets/60001894/5e4d3263-f216-4fe0-9dde-718c95262c96" alt="Outlier Removal">
</p>


### Seasonality removal
As seen in the plotted dataset features, there is a recurring pattern that is not relevant to the problem being modelled, therefore the seasonality must be removed, as seen in the following code snippet:

<p align="center">
  <img src="https://github.com/NeilVaria/Neural-network/assets/60001894/f1b8e0d7-4851-4626-beb8-bceb3f123499" alt="Seasonality Removal" height="200" width="auto">
</p>

<p align="center">
  <img src="https://github.com/NeilVaria/Neural-network/assets/60001894/644f06b7-5c7f-4515-b1a8-fcac223b6143" alt="Seasonality Removal" height="75" width="auto">
</p>


There are a number of steps that are taken to remove the seasonality from the dataset:
1. Calculate the rolling average of the data using the given window size (15) , which calculates the mean value of each data point and its neighbours within the given window size, to smoothen the data.
2. The seasonality is then removed by subtracting the rolling average from the original data, as the rolling average should represent long term trends.
3. Then to centre the data around the same mean as before, the average is added back onto the data.

This results in the seasonal component being removed from the data, that can be plotted as follows:

<p align="center">
  <img src="https://github.com/NeilVaria/Neural-network/assets/60001894/06fee765-76d6-432c-bf7f-c2f9d6296b0c" alt="Seasonality Removed Data">
</p>


### Data normalistation
As we are using a sigmoid transfer function for the output node, we need to standardise our data between 0-1. Furthermore, data normalisation helps to speed up ANN training and can improve model performance. as seen in the following code snippet, min-max scaling is used to normalise the dataset:

<p align="center">
  <img src="https://github.com/NeilVaria/Neural-network/assets/60001894/dc2e5cc7-d90c-47c8-b06c-630e86f95b83" alt="Data Normalisation">
</p>


- For each feature, find the minimum value (min) and the maximum value (max) in the dataset.
- For each data point, subtract the minimum value from the current value and divide the result by the range (max - min) of the feature. This will scale the data point to a value between 0 and 1.

After the dataset has been normalised it can be plotted as follows:

<p align="center">
  <img src="https://github.com/NeilVaria/Neural-network/assets/60001894/2b26669f-2f47-41d3-915f-6a503971c235" alt="Normalised Data">
</p>


### Identifying relevant predictors

To identify the predictors for PanE, you can calculate the correlation coefficient between each predictor and the target variable. A correlation closer to -1 or 1 indicates a strong correlation between the predictor and the target variable (in this case PanE).

<p align="center">
  <img src="https://github.com/NeilVaria/Neural-network/assets/60001894/879986aa-7b93-4527-babd-81da6ffc5bc7" alt="Identifying Relevant Predictors">
</p>


Using the NumPy library the function in Figure 12 calculates the correlation coefficient using Pearson’s R (see above). 
Pearson’s R can be represented with: r = (∑(x_i - µX)(y_i - µY)) / √(∑(x_i - µX)² ∑(y_i - µY)²)

The results of function are as follows:

<p align="center">
  <img src="https://github.com/NeilVaria/Neural-network/assets/60001894/46228fc6-f361-498b-a6d3-1ec496c6ad37" alt="Correlation Coefficients" height="200" width="auto">
</p>


Typically values that are ± 0.5 to 0 are considered to have a weak correlation, therefore parameters T, W and DSP should be disregarded.
However, when removing the uncorrelated values, performance of the MLP is significantly reduced.

The following results are obtained when using all of the predictors:

<p align="center">
  <img src="https://github.com/NeilVaria/Neural-network/assets/60001894/3a60be92-3750-42b5-8018-3ad4c2d89fdd" alt="Using All Predictors">
</p>

The following results are obtained when only using SR and DRH:

<p align="center">
  <img src="https://github.com/NeilVaria/Neural-network/assets/60001894/1cb71d46-789a-4b08-b0a6-e676e5f8f4b7" alt="Using Only SR and DRH">
</p>


A lower Mean Squared Error (MSE) and Mean Absolute Error (MAE) indicates better performance, and a R-squared value as close to 1 as possible indicates good performance of an MLP. I will expand on this later on. As seen above, when only using SR and DRH, the R-Squared value is lower than when all predictors are being used. Furthermore, both the MSE and MAE values are significantly higher. Therefore, I will be selecting every parameter to train the ANN.

### Data splitting
There are 3 data sets that need to be created for training and evaluating the MLP: Calibration, Validation and Testing Sets. The calibration set is used to train the MLP. The validation set is to ensure that overfitting does not occur during training. The testing set is used to evaluate MLP performance, by testing the MLP with data it has not seen before. I decided to use a 60:20:20 split for my datasets, as I found it to be a good balance between having enough data to train the MLP and having a good amount of data to test the MLP with.

<p align="center">
  <img src="https://github.com/NeilVaria/Neural-network/assets/60001894/dec168ee-29ff-4951-8a86-f817f29143be" alt="Data Splitting">
</p>

<p align="center">
  <img src="https://github.com/NeilVaria/Neural-network/assets/60001894/accca63d-11a5-4855-9b34-0a2ce33e498e" alt="Data Splitting">
</p>

The above code snippets demonstrate how I split the data into the 3 sets. Within the split data function. The data is randomised before being split into each set, which ensures that any trend is removed from the data. Furthermore, it ensures that each set represents data acrossr a range of dates instead of only data containing sequential dates.


## Multi later perceptron implementation

### Network structure

<p align="center">
  <img src="https://github.com/NeilVaria/Neural-network/assets/60001894/ce0c0313-bfb2-429b-8855-35e31d90dae1" alt="Network Structure">
</p>

The above diagram represents the layout of the network. I use 5 inputs (T, W, SR, DSP, DRH, PanE), 1 hidden layer containing 10 hidden nodes and one output node (PanE).

### Technologies and libraries
I am using Python for my MLP implementation, due to Python’s extensive support for data analysis.Python has a large ecosystem of libraries, which makes it ideal for implementing the MLP.
I am using the following libraries for my MLP implementation:
- “pandas”: This library is used for data manipulation and analysis. I am using it to read the dataset from an Excel file, perform data cleaning and pre-processing, and handle the data in a DataFrame format. Pandas is ideal for working with tabular data, making it easy to manipulate columns, rows, and perform various calculations.
- “NumPy”: This library is used for numerical computing and provides support for arrays and matrices. It is used in the code to perform various calculations, such as computing dot products and other matrix operations. NumPy allows for efficient numerical computations and is widely used in scientific computing.
- “matplotlib”: This library is used for data visualization. In the code, I am using it to create various line plots to visualize different graphs and data. Matplotlib's pyplot module provides a simple interface for creating graphs and customizing their appearance.

### Backpropagation implementation
The below snippet shows the sigmoid activation function for given variable “x”, that introduces non-linearity into the neural network and maps input values to range (0,1). It uses the following formula: x * (1 - x).

<p align="center">
  <img src="https://github.com/NeilVaria/Neural-network/assets/60001894/a7c2dda0-b21a-45ea-8099-5e54f616eb87" alt="Sigmoid Activation Function">
</p>

The snippet below shows the sigmoid derivative function for given variable “x”, that is used during backpropagation to update the weights of the neural network. It uses the following formula: x * (1 - x).

<p align="center">
  <img src="https://github.com/NeilVaria/Neural-network/assets/60001894/235809e5-2ab1-4855-aa91-ff86afb57ddc" alt="Sigmoid Derivative Function">
</p>

<p align="center">
  <img src="https://github.com/NeilVaria/Neural-network/assets/60001894/afe199e1-2e19-4ceb-92ae-13cb78798b3e" alt="Backpropagation Implementation">
</p>


The above snippet shows the initialise weights function, that initialises weights and biases for all cells in the neural network based on the provided information about the number of input nodes, hidden nodes and output nodes. It assigns random small weights for input and output hidden connections.

<p align="center">
  <img src="https://github.com/NeilVaria/Neural-network/assets/60001894/1b1432a1-7410-4293-b20a-b37e70eff8be" alt="Initialise Weights Function">
</p>

The above snippet demonstrates my implementation of the backpropagation algorithm. It uses the calibration and validation datasets to train the MLP and implements early stopping to stop training if validation loss has not improved for a defined number of epochs.

The steps of training are as follows:
- Initialise weights and biases using the initialise_weights function.
- Start the main loop for the given number of epochs.
- Perform a forward pass through the network, computing weighted sums and activations for every node.
- Perform a backward pass, computing errors and deltas for output and hidden layers.
- Update the weights and biases based on calculated deltas and learning rate.
- Calculate the validation error using the predict function.
- Check if the validation loss has improved, update the best weights, and reset the counter for epochs without improvement. If not, increment the counter.
- Stop training if the validation loss has not improved for a defined number of epochs


### Network training:
I use my backpropagation algorithm to train my MLP as seen below:

<p align="center">
  <img src="https://github.com/NeilVaria/Neural-network/assets/60001894/ffd63136-6265-4148-b0af-bc2c83a0f4d4" alt="Network Training">
</p>


I have found the selected parameters to be the most effective for my MLP.
- hidden_nodes – changing the number of hidden nodes does not have a dramatic effect on the output, with lower number producing a slightly higher epoch error (around 0.035).
- learning_rate – any other learning rate has a dramatic effect on the effectiveness of the MLP, so 0.1 is the best value.
- epochs – early stopping usually stops training at around 200 epochs anyway so isn’t necessary to use any more epochs for the MLP.
- patience- the number of epochs to wait where the validation loss has not increased. 50 is a reasonable number for 800 epochs.

### Network evaluation
The network does diplay promising results, but given more time I would have preferred to implement the additional modifications (momentum, annealing, weight decay).

Firstly, a plot of the actual vs predicted values can be found below:

<p align="center">
  <img src="https://github.com/NeilVaria/Neural-network/assets/60001894/399eb209-99f6-4ef8-bf9a-3214e4763ce6" alt="Actual vs Predicted">
</p>


MSE is calculated by taking the average of the squared differences between the actual and predicted values. Lower value indicates better fit of the model to the data. MAE is calculated by taking the average of the absolute differences between the actual and predicted values. A smaller MAE indicates a better fit of the model to the data. R-squared (Coefficient of determination) is a statistical measure that represents the proportion of the variance in the dependent variable explained by the independent variables in a regression model. It ranges from 0 to 1, with 0 indicating that the model does not explain any variance in the data and 1 indicating that the model perfectly explains the variance.

The statistical results of the network performance can be found as follows:
- Mean Squared Error (MSE) on testing data: 0.001708271378289038
- Mean Absolute Error (MAE) on testing data: 0.030453582085664913
- R-squared (R2) on testing data: 0.9425650034848617
