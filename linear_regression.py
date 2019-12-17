# -*- coding: utf-8 -*-

"""
@author: Carlos Meza
"""
import numpy as np
import sys

## Return matrix of big phi
def get_bigphi(training_file, rows, cols, degree):
    result = []
    t_values = []
    for i in range(rows):
        # Append 1 to each row at the beginning
        result.append(1)
        for k in range(cols):
            # If last column is reached, add value to t values list
            if(k == (cols - 1)):
                t_values.append(training_file[i][k])
            # Summation of degree values for each dimension and add to vector of bigphi
            else:
                for deg in range(degree):
                    # If value of degree is 1, just add itself to matrix
                    if(deg == 0):
                        result.append(training_file[i][k])
                    else:
                        result.append(np.power(training_file[i][k], deg + 1))
    return result, t_values

# Return vector of weight = wml
def get_weights(big_phi, t_values, lamb):
    # Phi transpose 
    phi_t = np.transpose(big_phi)
    # value of identity matrix, lamb * I
    identity_value = len(phi_t)
    identity = np.identity(identity_value)
    lamb_m = np.multiply(identity, lamb)
    # lamb * I + Phi.t * Phi
    temp = np.matmul(phi_t, big_phi)
    first = np.add(lamb_m, temp)
    # Sudo inverse of the last result
    inverse = np.linalg.pinv(first)
    # The inverse result of the inner brackets * Phi Transpose
    second = np.matmul(inverse, phi_t)
    # Last result * T value vector
    weights = np.matmul(second, t_values)
    # Return weight vector
    return weights

# Return matrix of prediction of the test file using weights of training
def get_prediction(test_phi, weights):
    # Do dot product for each row of matrix, each row represents small phi
    test_result = np.dot(np.transpose(weights), test_phi)
    return test_result

# Simple function to print out weights vector
def print_weights(weights):
    length = len(weights)
    for x in range(length):
        print("w%d=%.4f" % (x, weights[x]))

# Simple function to print out test phase
def print_test(prediction, target_value):
    length = len(prediction)
    for x in range(length):
        # Get squared errors by subtracting values and squaring it
        error = prediction[x] - target_value[x]
        error = error ** 2
        # Print out in format for each error
        print("ID=%5d, output=%5.4f, target value = %5.4f, squared error = %.4f" % (x+1, prediction[x], target_value[x], error))

# Linear regression function which takes training and test file with degree and lambda
# Then generates bigphi for both and output values and weights, then uses weights to predict
# Output values for test file
def linear_regression(training, degree, lamb, test):
    # Get length of both files for rows and columns to help with reshaping later
    training_rows = len(training)
    training_cols = len(training[0])
    test_rows = len(test)
    test_cols = len(test[0])
    
    ########## T R A I N I N G   S T A G E ##########
    # Get the bigphi and t values
    big_phi, t_values = get_bigphi(training, training_rows, training_cols, degree)
    # Convert list of both values into np array, and reshape big phi values of col values not including last + 1
    big_phi = np.asarray(big_phi)
    big_phi = np.reshape(big_phi, (training_rows, (degree * (training_cols - 1) + 1)))
    t_values = np.asarray(t_values)
    # Get weights of the values from given input text file
    weights = get_weights(big_phi, t_values, lamb)
    # Print out values of the weights for training portion
    print_weights(weights)
    
    ########## T E S T I N G   S T A G E ##########
    # Get big phi values for test matrix
    test_phi, target_value = get_bigphi(test, test_rows, test_cols, degree)
    # Convert list of values into array and reshape to matrix
    test_phi = np.asarray(test_phi)
    test_phi = np.reshape(test_phi, (test_rows, (degree * (test_cols - 1) + 1)))
    # Transpose matrix of phi to do dot product, each row represents small phi vector
    # So instead of looping through each row and multiplying, just transpose and dot matrix
    test_phi = np.transpose(test_phi)
    target_value = np.asarray(target_value)
    # Get predicted outcome from test file with prediction function
    prediction = get_prediction(test_phi, weights)
    # Print out values for test result
    print_test(prediction, target_value)


######################## M A I N ########################
# Check number of arguments given
if(len(sys.argv) != 5):
    print("Error, not enough arguments given!")
else:
    # When running python, uses training file name, degree
    # lambda and test file name
    training_file = sys.argv[1]
    degree = int(sys.argv[2])
    lamb = int(sys.argv[3])
    test_file = sys.argv[4]
    
    # Load file from text file into array
    with open(training_file) as textFile:
        training_temp = [line.split() for line in textFile]  
    with open(test_file) as textFile:
        test_temp = [line.split() for line in textFile]     
    
    # Convert list items into float
    training = np.array(training_temp, float)
    test = np.array(test_temp, float)
    
    linear_regression(training, degree, lamb, test)
#########################################################