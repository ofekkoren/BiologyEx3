from Buildnet0 import NeuralNetwork
import sys

# The code for running the trained network on new data
# Load Trained Network Weights
def load_weights(file_path):
    weights = []
    with open(file_path, 'r') as file:
        for line in file:
            weight = float(line.strip())
            weights.append(weight)
    return weights

# Data Loading and Preprocessing
def load_data(file_path):
    strings = []
    with open(file_path, 'r') as file:
        for line in file:
            inputs = line.strip()
            strings.append(inputs)
    return strings

# Load and Run Test Data
def run_network(weights, data):
    predictions = []
    for inputs in data:
        network = NeuralNetwork()
        network.weights = weights
        prediction = network.predict([int(digit) for digit in inputs])
        predictions.append(round(prediction))
    return predictions

def runnet0(weights_file, unclassified_data_file):
    # The code for running the trained network on new data
    weights = load_weights(weights_file)  # Change file name accordingly
    new_data = load_data(unclassified_data_file)
    test_predictions = run_network(weights, new_data)
    with open('testnet0_predictions.txt', 'w') as file:
        for prediction in test_predictions:
            file.write(str(prediction) + '\n')


# import itertools
#
# def generate_binary_combinations():
#     combinations = []
#     positions = list(range(16))
#     for combination in itertools.combinations(positions, 16):
#         bits = ['0'] * 16
#         for pos in combination:
#             bits[pos] = '1'
#         combinations.append(''.join(bits))
#
#     with open('test_file.txt', 'w') as file:
#         for s in combinations:
#             file.write(str(s) + '\n')
#

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Please provide two filename arguments for the weights file and the test file.")
        sys.exit(1)

    weights_file = sys.argv[1]
    test_file = sys.argv[2]
    runnet0(weights_file, test_file)
    # runnet0('wnet0.txt', 'testnet1.txt')

    # combinations = generate_binary_combinations()
    # runnet0('wnet0.txt', 'test_file.txt')
