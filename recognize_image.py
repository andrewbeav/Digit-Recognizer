from neuro_py import NeuralNetwork
import numpy as np
import json
import scipy.misc
import sys

data = json.load(open('neural_network.json', 'r'))

neural_net = NeuralNetwork(data['input_nodes'], data['hidden_nodes'], data['output_nodes'], data['learning_rate'])

neural_net.wih = np.array(data['wih'])
neural_net.who = np.array(data['who'])

img_array = scipy.misc.imread(sys.argv[1], flatten=True)
img_data = 255.0 - img_array.reshape(784)
img_data = (img_data / 255.0 * 0.99) + 0.01

print(np.argmax(neural_net.query(img_data)))
