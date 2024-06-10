import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Convert the date format


def convert_date(date_str):
    month = int(date_str[:-4])
    day = int(date_str[-4:-2])
    year = int(date_str[-2:]) + 1900
    return pd.Timestamp(year=year, month=month, day=day)


# Function to remove outliers from the dataset using the IQR method
def remove_outliers(data, columns):
    for col in columns:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
    return data


# Function to plot all the data on the same graph
def plot_data(data, title):
    plt.figure(figsize=(20, 10))
    i = 0
    for col in data.columns:
        plt.plot(data.index, data[col], label=col)
        i += 1
    plt.xlabel("Date")
    plt.title(title)
    plt.legend()
    plt.show()


# Function to calculate the rolling average of the data, used to remove seasonality
def calculate_rolling_average(data, window_size):
    rolling_average = data.rolling(window=window_size).mean()
    average = data.mean()
    difference = data - rolling_average
    difference = difference + average
    return difference


# Function to normalize the data using min-max scaling
def min_max_scaling(data):
    return (data - data.min()) / (data.max() - data.min())


# Function to split the data into calibration, testing and validation sets
def split_data(data, calibration_ratio, testing_ratio):
    shuffled_indices = np.random.permutation(len(data))
    calibration_set_size = int(len(data) * calibration_ratio)
    testing_set_size = int(len(data) * testing_ratio)

    calibration_indices = shuffled_indices[:calibration_set_size]
    testing_indices = shuffled_indices[calibration_set_size:
                                       calibration_set_size + testing_set_size]
    validation_indices = shuffled_indices[calibration_set_size +
                                          testing_set_size:]

    return data.iloc[calibration_indices], data.iloc[testing_indices], data.iloc[validation_indices]


# Function to computer the sigmoid activation function for a given input
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Function to compute the derivative of the sigmoid function for a given input
def sigmoid_derivative(x):
    return x * (1 - x)


# Function to initialize weights and biases for all cells according to the provided information
def initialise_weights(input_nodes, hidden_nodes, output_nodes):
    # Assign random small weights for input-hidden connections
    weights_input_hidden = np.random.uniform(
        -2 / input_nodes, 2 / input_nodes, (input_nodes, hidden_nodes))
    # Assign random small weights for hidden-output connections
    weights_hidden_output = np.random.uniform(
        -2 / hidden_nodes, 2 / hidden_nodes, (hidden_nodes, output_nodes))
    return weights_input_hidden, weights_hidden_output

# Function to calculate correlations between the data and the predictant


def calculate_correlations(data, predictant):
    correlations = data.corr()[predictant].drop(predictant)
    print(f"Correlation coefficients with {predictant}:")
    for col, value in correlations.items():
        print(f"{col}: {value}")
    return correlations[correlations.abs() > 0.5].index.tolist()

# Function to train the network using the backpropagation algorithm


def mlp_backpropagation(x, y, x_validation, y_validation, hidden_nodes, learning_rate, epochs, patience):
    input_nodes = x.shape[1]
    output_nodes = y.shape[1]
    training_loss_history = []
    validation_loss_history = []
    best_validation_loss = np.inf
    best_weights_input_hidden = None
    best_weights_hidden_output = None
    epochs_without_improvement = 0

    # Initialize weights and biases using the initialize_weights function
    weights_input_hidden, weights_hidden_output = initialise_weights(
        input_nodes, hidden_nodes, output_nodes)

    # Main loop
    for epoch in range(epochs):
        # Make a forward pass through the network computing weighted sums, and activations for every node
        hidden_layer_input = np.dot(x, weights_input_hidden)
        hidden_layer_output = sigmoid(hidden_layer_input)

        output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
        output_layer_output = sigmoid(output_layer_input)

        # Backward pass computing, for each node j
        output_error = y - output_layer_output
        output_delta = output_error * sigmoid_derivative(output_layer_output)

        hidden_error = np.dot(output_delta, weights_hidden_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(hidden_layer_output)

        # Update the weights and biases
        weights_hidden_output += np.dot(hidden_layer_output.T,
                                        output_delta) * learning_rate
        weights_input_hidden += np.dot(x.T, hidden_delta) * learning_rate

        # Add the current training error to the training loss history
        training_loss_history.append(np.mean(np.abs(output_error)))

        # Calculate the validation error
        y_validation_predicted = predict(
            x_validation, weights_input_hidden, weights_hidden_output)
        validation_error = y_validation - y_validation_predicted
        current_validation_loss = np.mean(np.abs(validation_error))
        validation_loss_history.append(current_validation_loss)

        # Stop training if the validation loss has not improved for a defined number of epochs
        if current_validation_loss < best_validation_loss:
            best_validation_loss = current_validation_loss
            best_weights_input_hidden = weights_input_hidden.copy()
            best_weights_hidden_output = weights_hidden_output.copy()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        # Print error every 100 epochs
        if epoch % 10000 == 0:
            print(f"Epoch {epoch}: Error {np.mean(np.abs(output_error))}")
        error = np.mean((y - output_layer_output)**2)
        # Add error handling and print statements
        if np.isnan(error).any():
            print(f"Epoch {epoch}: Error contains NaN values")
            print("hidden_layer_output:", hidden_layer_output)
            print("output_layer_output:", output_layer_output)
            break

    return best_weights_input_hidden, best_weights_hidden_output, training_loss_history, validation_loss_history


# Function to make predictions using the trained network
def predict(testing_data, weights_input_hidden, weights_hidden_output):
    # Compute the input for the hidden layer by multiplying the input data with the input-to-hidden layer weights,
    # then apply the sigmoid activation function
    hidden_layer_inp = np.dot(testing_data, weights_input_hidden)
    hidden_layer_out = sigmoid(hidden_layer_inp)

    # Compute the input for the output layer by multiplying the hidden layer output with the hidden-to-output layer weights,
    # then apply the sigmoid activation function to get the predictions
    output_layer_inp = np.dot(hidden_layer_out, weights_hidden_output)
    output_layer_out = sigmoid(output_layer_inp)

    return output_layer_out


# Read the data from the Excel file
data = pd.read_excel('DataSet.xlsx', engine='openpyxl')


# Convert the date format
data['Date'] = data['Date'].astype(str).apply(convert_date)


# Convert all columns to numeric types and replace non-numeric values with null values
for col in data.columns:
    if col != 'Date':
        data[col] = pd.to_numeric(data[col], errors='coerce')


# Remove rows with null values
data_cleaned = data.dropna()


# Set the Date column as the index for plotting
data_cleaned.set_index('Date', inplace=True)

# Plot the original data
# plot_data(data_cleaned, "Plot of daily environmental factors at a site in San Diego, USA; 1/1/1987 - 31/12/1990 (Rows containing missing data removed)")


# Remove outliers from the dataset
data_cleaned = remove_outliers(data_cleaned, data_cleaned.columns)

# Plot the data with outliers removed
# plot_data(data_cleaned, "Plot of daily environmental factors at a site in San Diego, USA; 1/1/1987 - 31/12/1990 (Outliers removed))")


# remove the seasonal trend
data_cleaned = calculate_rolling_average(data_cleaned, 15)

# Plot the data with seasonal trend removed
# plot_data(data_cleaned, "Plot of daily environmental factors at a site in San Diego, USA; 1/1/1987 - 31/12/1990 (Seasonality Removed)")


# Apply min-max scaling to the entire dataset
data_cleaned = data_cleaned.apply(min_max_scaling)

# Plot the normalized data
# plot_data(data_cleaned,
#           "Plot of daily environmental factors at a site in San Diego, USA; 1/1/1987 - 31/12/1990 (Min max scaling applied)")


# Remove any rows with invalid values
data_cleaned = data_cleaned.dropna()


# Calculate correlations for each predictor with PanE (the predictant)
# strong_correlations = calculate_correlations(data_cleaned, 'PanE')
# print("Predictors with correlation coefficient greater than 0.5:")
# print(strong_correlations)


# Split the data into calibration, testing, and validation sets with a ratio of 60:20:20
calibration_ratio, testing_ratio = 0.6, 0.2
calibration_data, testing_data, validation_data = split_data(
    data_cleaned, calibration_ratio, testing_ratio)

# Separate input features and target variables for each split
input_calibration, target_calibration = calibration_data.drop(
    'PanE', axis=1).values, calibration_data[['PanE']].values
input_testing, target_testing = testing_data.drop(
    'PanE', axis=1).values, testing_data[['PanE']].values
input_validation, target_validation = validation_data.drop(
    'PanE', axis=1).values, validation_data[['PanE']].values


# Initialize the network parameters
hidden_nodes = 5
learning_rate = 0.01
epochs = 100000
patience = 500

# Train the network using the calibration dataset
weights_input_hidden, weights_hidden_output, training_loss_history, validation_loss_history = mlp_backpropagation(
    input_calibration, target_calibration, input_validation, target_validation, hidden_nodes, learning_rate, epochs, patience
)


# Evaluate the network using the testing dataset
predicted_values = predict(
    input_testing, weights_input_hidden, weights_hidden_output)

# Calculate the Mean Squared Error (MSE)
mse = np.mean((target_testing - predicted_values)**2)
print(f"Mean Squared Error (MSE) on testing data: {mse}")

# Calculate the Mean Absolute Error (MAE)
mae = np.mean(np.abs(target_testing - predicted_values))
print(f"Mean Absolute Error (MAE) on testing data: {mae}")

# Calculate the R-squared (R2) value
sst = np.sum((target_testing - target_testing.mean())**2)
ssr = np.sum((predicted_values - target_testing.mean())**2)
r2 = ssr / sst
print(f"R-squared (R2) on testing data: {r2}")

# Creating a dataframe with the actual and predicted values using dates as index
results = pd.DataFrame({'Actual': target_testing.flatten(
), 'Predicted': predicted_values.flatten()}, index=testing_data.index)

results.sort_index(inplace=True)

plt.figure(figsize=(20, 5))
plt.scatter(results.index, results['Actual'], label='Actual', marker='o')
plt.scatter(results.index, results['Predicted'], label='Predicted', marker='x')
# Connect corresponding pairs with lines
for index, row in results.iterrows():
    plt.plot([index, index], [row['Actual'], row['Predicted']],
             color='gray', linestyle='--', alpha=0.5)
plt.xlabel('Date')
plt.ylabel('Values')
plt.title('Actual vs Predicted Values')
plt.legend()
plt.show()
