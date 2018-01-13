import neuro_py
from neuro_py import NeuralNetwork
import numpy as np
import matplotlib.pyplot
import scipy.ndimage.interpolation
import json

input_nodes = 784
hidden_nodes = 200
output_nodes = 10

learning_rate = 0.01

neural_net = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

training_data_file = open('mnist_dataset/mnist_train.csv', 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

epochs = 10

for i in range(epochs):
    print('epoch: ', i)
    for record in training_data_list:
        all_values = record.split(',')

        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        targets = np.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        neural_net.train(inputs, targets)

        inputs_plus_10 = scipy.ndimage.interpolation.rotate(inputs.reshape(28, 28), 10, cval=0.01, order=1, reshape=False).reshape(784)
        neural_net.train(inputs_plus_10, targets)

        inputs_minus_10 = scipy.ndimage.interpolation.rotate(inputs.reshape(28, 28), -10, cval=0.01, order=1, reshape=False).reshape(784)
        neural_net.train(inputs_minus_10, targets)

data = {
    'input_nodes':input_nodes,
    'hidden_nodes':hidden_nodes,
    'output_nodes':output_nodes,
    'learning_rate':learning_rate,
    'wih':neural_net.wih.tolist(),
    'who':neural_net.who.tolist()
    }

json.dump(data, open(neuro_py.DATAFILE, 'w'))

test_data_file = open('mnist_dataset/mnist_test.csv', 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

score = 0
for entry in test_data_list:
    all_values = entry.split(',')
    image_array = np.asfarray(all_values[1:]).reshape(28,28)
    outputs = neural_net.query(np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    if np.argmax(outputs) == int(all_values[0]): score += 1

print('Score:', (score/len(test_data_list))*100)
