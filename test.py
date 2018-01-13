from neuro_py import NeuralNetwork
import numpy as np
import matplotlib.pyplot
import json
import scipy.misc
import sys

data = json.load(open('neural_network.json', 'r'))

neural_net = NeuralNetwork(data['input_nodes'], data['hidden_nodes'], data['output_nodes'], data['learning_rate'])

neural_net.wih = np.array(data['wih'])
neural_net.who = np.array(data['who'])

test_data_file = open('mnist_dataset/mnist_test.csv', 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

score = 0
i = 0
for entry in test_data_list:
    all_values = entry.split(',')
    image_array = np.asfarray(all_values[1:]).reshape(28,28)
    outputs = neural_net.query(np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    if max(outputs) > 1.0:
        print(i)
    if np.argmax(outputs) == int(all_values[0]): score += 1
    i += 1

print('Score: {}', score/len(test_data_list))

all_values = test_data_list[0].split(',')
image_array = np.asfarray(all_values[1:]).reshape(28,28)
outputs = neural_net.query(np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
print(outputs)
print(np.argmax(outputs))
matplotlib.pyplot.imshow(image_array, cmap='Greys', interpolation=None)
