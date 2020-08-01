import numpy
import random
import math
import csv


def convert_csv_file_to_array(file_name):
    array = []
    with open(file_name) as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        for row in reader:
            array.append(row)
    return array


def index_of_max(array):
    temp = array[0]
    num = 0
    for i in range(0, len(array)):
        if temp < array[i]:
            num = i
            temp = array[i]
    return num


def normalisation_factor(array):
    maximum = array[0]
    minimum = array[0]
    for i in range(0, len(array)):
        if maximum < array[i]:
            maximum = array[i]
        if minimum > array[i]:
            minimum = array[i]
    difference = maximum - minimum
    return minimum, difference


def sigmoid(number):
    sol = (1 / (1 + math.exp(-number)))
    return sol


def d_sigmoid(number):
    sol = number * (1 - number)
    return sol


def probability_generator(probability):
    number = random.random()
    if number < probability:
        return 1
    else:
        return 0


def parameters_gathering(ans):
    # get the parameters of the neural network
    lay = int(input("Type the number of hidden layers: "))
    layer_nodes = []
    for i in range(0, lay):
        layer_nodes.append(int(input("type the number of nodes in layer " + str(i + 1) + ": ")))
    layer_nodes.append(ans)
    print("")
    learning_rate = float(input("type the learning rate: "))
    print("")
    return lay, layer_nodes, learning_rate


def initialise_neural_network(inp, ans, lay, layer_nodes):
    # initialising the neural network

    # initialising the layers
    layers = []
    temp = []
    for i in range(0, inp):
        temp.append(0)
    layers.append(numpy.transpose(numpy.matrix(temp)))
    for i in range(0, lay):
        temp = []
        for j in range(0, layer_nodes[i]):
            temp.append(0)
        layers.append(numpy.transpose(numpy.matrix(temp)))
    temp = []
    for i in range(0, ans):
        temp.append(0)
    layers.append(numpy.transpose(numpy.matrix(temp)))

    # initialising the weights
    weights = []
    for i in range(0, len(layers) - 1):
        temp = []
        for j in range(0, len(layers[i + 1])):
            temp1 = []
            for k in range(0, len(layers[i])):
                temp1.append(random.random() * 2 - 1)
            temp.append(temp1)
        weights.append(numpy.matrix(temp))

    # initialising the biases
    bias = []
    for i in range(0, len(layers) - 1):
        temp = []
        for j in range(0, len(layers[i + 1])):
            temp.append(random.random() * 2 - 1)
        bias.append(numpy.transpose(numpy.matrix(temp)))

    return layers, weights, bias


def forward_prop(inputs, weights, bias):
    # Forward propagation
    forward_propagation = [inputs]
    for i in range(0, len(bias)):
        forward_propagation.append(numpy.matmul(weights[i], forward_propagation[i]) + bias[i])
        for j in range(0, len(forward_propagation[i + 1])):
            forward_propagation[i + 1][j][0] = sigmoid(float(forward_propagation[i + 1][j][0]))

    return forward_propagation


def back_prop(forward_propagation, answer, learning_rate, bias, weights):
    # Weighted error
    master_error = [answer - forward_propagation[len(forward_propagation) - 1]]
    for i in range(1, len(bias)):
        j = len(bias) - i
        master_error.append(numpy.matmul(numpy.transpose(weights[j]), master_error[i - 1]))

    # Calculate gradient
    gradients = []
    for i in range(1, len(forward_propagation)):
        temp = []
        m = len(forward_propagation) - i - 1
        for j in range(0, len(forward_propagation[i])):
            temp.append(d_sigmoid(float(forward_propagation[i][j])) * float(master_error[m][j]) * learning_rate)
        gradients.append(numpy.transpose(numpy.matrix(temp)))

    # Calculation of differences
    forward_propagation_transpose = []
    for i in range(0, len(forward_propagation)):
        forward_propagation_transpose.append(numpy.transpose(forward_propagation[i]))

    weights_differences = []
    for i in range(0, len(gradients)):
        weights_differences.append(numpy.matmul(gradients[i], forward_propagation_transpose[i]))

    bias_differences = gradients

    for i in range(0, len(weights)):
        weights[i] = weights[i] + weights_differences[i]
    for i in range(0, len(bias)):
        bias[i] = bias[i] + bias_differences[i]

    return weights, bias
