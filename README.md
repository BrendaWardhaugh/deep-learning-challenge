# deep-learning-challenge
Through the use of machine learning and neural networks, use the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup

From Alphabet Soup’s business team, you have received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as:

EIN and NAME—Identification columns
APPLICATION_TYPE—Alphabet Soup application type
AFFILIATION—Affiliated sector of industry
CLASSIFICATION—Government organization classification
USE_CASE—Use case for funding
ORGANIZATION—Organization type
STATUS—Active status
INCOME_AMT—Income classification
SPECIAL_CONSIDERATIONS—Special considerations for application
ASK_AMT—Funding amount requested
IS_SUCCESSFUL—Was the money used effectively

## Step 1: Preprocess the Data
Using your knowledge of Pandas and scikit-learn’s StandardScaler(), you’ll need to preprocess the dataset. This step prepares you for Step 2, where you'll compile, train, and evaluate the neural network model.

Start by uploading the starter file to Google Colab, then using the information we provided in the Challenge files, follow the instructions to complete the preprocessing steps.

Read in the charity_data.csv to a Pandas DataFrame, and be sure to identify the following in your dataset:
* What variable(s) are the target(s) for your model?
* What variable(s) are the feature(s) for your model?
* Drop the EIN and NAME columns.

Determine the number of unique values for each column.

For columns that have more than 10 unique values, determine the number of data points for each unique value.

Use the number of data points for each unique value to pick a cutoff point to bin "rare" categorical variables together in a new value, Other, and then check if the binning was successful.

Use pd.get_dummies() to encode categorical variables.

Split the preprocessed data into a features array, X, and a target array, y. Use these arrays and the train_test_split function to split the data into training and testing datasets.

Scale the training and testing features datasets by creating a StandardScaler instance, fitting it to the training data, then using the transform function.

## Step 2: Compile, Train, and Evaluate the Model
Using your knowledge of TensorFlow, you’ll design a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup-funded organization will be successful based on the features in the dataset. You’ll need to think about how many inputs there are before determining the number of neurons and layers in your model. Once you’ve completed that step, you’ll compile, train, and evaluate your binary classification model to calculate the model’s loss and accuracy.

* Continue using the file in Google Colab in which you performed the preprocessing steps from Step 1.

* Create a neural network model by assigning the number of input features and nodes for each layer using TensorFlow and Keras.

* Create the first hidden layer and choose an appropriate activation function.

* If necessary, add a second hidden layer with an appropriate activation function.

* Create an output layer with an appropriate activation function.

* Check the structure of the model.

* Compile and train the model.

* Create a callback that saves the model's weights every five epochs.

* Evaluate the model using the test data to determine the loss and accuracy.

* Save and export your results to an HDF5 file. Name the file AlphabetSoupCharity.h5.

## Step 3: Optimize the Model
Using your knowledge of TensorFlow, optimize your model to achieve a target predictive accuracy higher than 75%.

Use any or all of the following methods to optimize your model:

*Adjust the input data to ensure that no variables or outliers are causing confusion in the model, such as:
* Dropping more or fewer columns.
* Creating more bins for rare occurrences in columns.
* Increasing or decreasing the number of values for each bin.
* Add more neurons to a hidden layer.
* Add more hidden layers.
* Use different activation functions for the hidden layers.
* Add or reduce the number of epochs to the training regimen.

Import your dependencies and read in the charity_data.csv to a Pandas DataFrame.

Preprocess the dataset as you did in Step 1. Be sure to adjust for any modifications that came out of optimizing the model.

Design a neural network model, and be sure to adjust for modifications that will optimize the model to achieve higher than 75% accuracy.

Save and export your results to an HDF5 file. Name the file AlphabetSoupCharity_Optimization.h5.

## Step 4: Write a Report on the Neural Network Model
#### Overview of the analysis:
The Alphabet Soup foundation wants to create an algorithm to predict whether or not applicants for 
funding will be successful. Using machine learning and neural networks, the features provided in the 
dataset must be used to create a binary classifier that is capable of predicting whether applicants will be 
successful if they receive funding from Alphabet Soup. 
#### Results: 
##### Data Preprocessing
The target variable of my model is IS_SUCCESSFUL and has the value of 1 for yes 
ad 0 for no. The feature variables of my model are Application type, Affiliation,
Classification, use case, organization, status, income amt, and ask amt. Variables 
that were removed from the input data were EIN, NAME and 
SPECIAL_CONSIDERATIONS because they were neither targets nor features.
##### Compiling, Training, and Evaluating the Model
o How many neurons, layers, and activation functions did you select for your 
neural network model, and why?
▪ I selected 3 layers(90 neurons, 50 neurons, 20 neurons) and the ‘relu’ and 
‘sigmoid’ functions and 100 epochs. I selected these parameters because 
after 4 attempts this was the combination that yielded the highest 
accuracy percentage. 
o Were you able to achieve the target model performance?
▪ No, after four attempts I was not able to achieve the target model 
performance of 75%. I was able to achieve 73.01% accuracy through 
adjusting different elements of the model.
o What steps did you take in your attempts to increase model performance?
▪ In an attempt to increases model performance, I attempted dropping 
another column, creating more bins, added more neurons to a hidden 
layer, added more hidden layers, tried different activation functions for 
the hidden layers and increased the number of epochs to the training 
regimen. 
#### Summary: 
The final automatically optimized neural network trained model was able to achieved 73% 
prediction accuracy, using relu and sigmoid activation functions with 3 hidden layers at a 90, 50, 
20 neurons split and 100 training epochs. Further analysis should be conducted to increase the 
predicted accuracy rate to achieve at least 75%
