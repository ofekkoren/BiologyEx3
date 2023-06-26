import random
import math
import sys

# Genetic Algorithm Parameters
POPULATION_SIZE = 150
MUTATION_RATE = 0.1
NUM_GENERATIONS = 150
KEEP_BEST_POPULATION_PERCENT = 0.2
NUM_OF_MUTATIONS = 1
NUM_OF_GENERATIONS_IN_LOCAL_MAX = 5
LOCAL_MAX_NOT_IMPROVED_THRESHOLD = 0

# Neural Network Parameters
NUM_INPUTS = 16
NUM_OUTPUTS = 1
NUM_HIDDEN_LAYERS = 2
NEURONS_PER_HIDDEN_LAYER = [6, 8]  # Number of neurons in each hidden layer

def split_data(file_path):
    # Read the lines from the input file
    with open(file_path, 'r') as f:
        lines = f.readlines()
    # Shuffle the lines randomly
    random.shuffle(lines)
    # Calculate the split index for 80-20 division
    split_index = int(len(lines) * 0.8)
    # Split the lines into train and test sets
    train_lines = lines[:split_index]
    test_lines = lines[split_index:]
    # Write train lines to the train file
    with open(TRAIN_FILE, 'w') as f:
        f.writelines(train_lines)
    # Write test lines to the test file
    with open(TEST_FILE, 'w') as f:
        f.writelines(test_lines)


# Data Loading and Preprocessing
def load_data(file_path):
    strings = []
    targets = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            inputs = line[:-2]  # Extract all digits except the last one
            inputs = inputs.replace(" ", "")
            target = int(line[-1])  # Get the last digit as the target
            strings.append(inputs)
            targets.append(target)
    return strings, targets

# Neural Network Definition
class NeuralNetwork:
    def __init__(self):
        self.weights = self.initialize_weights()

    def initialize_weights(self):
        num_weights = (NUM_INPUTS + 1) * NEURONS_PER_HIDDEN_LAYER[0]  # Input-hidden layer weights
        for i in range(NUM_HIDDEN_LAYERS - 1):  # Hidden-hidden layer weights
            num_weights += (NEURONS_PER_HIDDEN_LAYER[i] + 1) * NEURONS_PER_HIDDEN_LAYER[i + 1]
        num_weights += (NEURONS_PER_HIDDEN_LAYER[-1] + 1) * NUM_OUTPUTS  # Hidden-output layer weights

        weights = [random.uniform(-0.5, 0.5) for _ in range(num_weights)]
        # weights = [random.uniform(-1, 1) for _ in range(num_weights)]
        return weights

    def predict(self, inputs):
        start = 0
        end = (NUM_INPUTS + 1) * NEURONS_PER_HIDDEN_LAYER[0]  # Input-hidden layer weights
        hidden_layers = [[] for _ in range(NUM_HIDDEN_LAYERS)]

        for layer in range(NUM_HIDDEN_LAYERS):
            hidden_weights = self.weights[start:end]
            start = end
            if layer < NUM_HIDDEN_LAYERS - 1:
                end += (NEURONS_PER_HIDDEN_LAYER[layer] + 1) * NEURONS_PER_HIDDEN_LAYER[layer + 1]
            else:
                end += (NEURONS_PER_HIDDEN_LAYER[-1] + 1) * NUM_OUTPUTS

            if layer == 0:
                for i in range(NEURONS_PER_HIDDEN_LAYER[layer]):
                    weight_start = i * (NUM_INPUTS + 1)
                    weight_end = weight_start + (NUM_INPUTS + 1)
                    weights = hidden_weights[weight_start:weight_end]

                    weighted_sum = sum([inputs[j] * weights[j] for j in range(NUM_INPUTS)])
                    hidden_layers[layer].append(self.activation_function(weighted_sum))
            else:
                for i in range(NEURONS_PER_HIDDEN_LAYER[layer]):
                    weight_start = i * (NEURONS_PER_HIDDEN_LAYER[layer - 1] + 1)
                    weight_end = weight_start + (NEURONS_PER_HIDDEN_LAYER[layer - 1] + 1)
                    weights = hidden_weights[weight_start:weight_end]

                    weighted_sum = sum(
                        [hidden_layers[layer - 1][j] * weights[j] for j in range(NEURONS_PER_HIDDEN_LAYER[layer - 1])])
                    hidden_layers[layer].append(self.activation_function(weighted_sum))

        output = hidden_layers[-1][0]  # Only one output neuron in this example
        return output

    def get_fitness_value(self):
        return self.fitness

    @staticmethod
    def activation_function(x):
        # Sigmoid activation function
        return 1 / (1 + math.exp(-x))


# Genetic Algorithm Functions
def calculate_fitness(network, data):
    correct_predictions = 0
    for inputs, target in data:
        prediction = network.predict([int(digit) for digit in inputs])
        if round(prediction) == target:
            correct_predictions += 1
    network.fitness = correct_predictions / len(data)


def select_parent(population):
    relative_fitness = [f.get_fitness_value() for f in population]
    # choose a random value from the list based on the given probabilities
    chosen_index = random.choices(range(len(relative_fitness)), weights=relative_fitness)[0]
    return population[chosen_index]


def crossover(parent1, parent2):
    child = NeuralNetwork()
    child.weights = []
    length = len(parent1.weights)
    for i in range(length):
        child.weights.append(0.5 * (parent1.weights[i] + parent2.weights[i]))
    return child


def mutate(network):
    # Random mutation of weights within each layer
    length = len(network.weights)
    for i in range(length):
        if random.uniform(0, 1) < MUTATION_RATE:
            network.weights[i] += random.uniform(-0.5, 0.5)

# The code for running the trained network on new data
# Load Trained Network Weights
def load_weights(file_path):
    weights = []
    with open(file_path, 'r') as file:
        for line in file:
            weight = float(line.strip())
            weights.append(weight)
    return weights


# Load and Run Test Data
def run_network(weights, data):
    predictions = []
    for inputs, _ in data:
        network = NeuralNetwork()
        network.weights = weights
        prediction = network.predict([int(digit) for digit in inputs])
        predictions.append(round(prediction))
    return predictions


def check_if_stuck_in_local_max(best_solutions, NUM_OF_GENERATIONS_IN_LOCAL_MAX, LOCAL_MAX_NOT_IMPROVED_THRESHOLD):
    generation = len(best_solutions)
    if (generation < NUM_OF_GENERATIONS_IN_LOCAL_MAX or best_solutions[generation - 1] - best_solutions[
        generation - 1 - NUM_OF_GENERATIONS_IN_LOCAL_MAX] > LOCAL_MAX_NOT_IMPROVED_THRESHOLD):
        return True
    return False


def buildnet0(train_file, test_file):
    global MUTATION_RATE
    global NUM_OF_MUTATIONS

    train_x, train_y = load_data(train_file)
    test_x, test_y = load_data(test_file)
    # Initialize the population with random neural networks
    population = [NeuralNetwork() for _ in range(POPULATION_SIZE)]
    # Iterate over a fixed number of generations
    generation = 0
    best_solutions_score = []
    data = list(zip(train_x, train_y))
    while generation < NUM_GENERATIONS and (len(best_solutions_score) == 0 or best_solutions_score[generation - 1] < 1):
        # Calculate fitness for each network in the population using the calculate_fitness function
        for network in population:
            # data = list(zip(data_x, data_y))
            calculate_fitness(network, data)
        # Sort the population based on fitness:
        population.sort(key=lambda x: x.fitness, reverse=True)
        # Print best network fitness for each generation
        best_fitness = population[0].fitness
        worst_fitness = population[-1].fitness
        best_solutions_score.append(best_fitness)
        print(f"Generation {generation + 1}: Best Fitness = {best_fitness} Worst Fitness = {worst_fitness}")
        generation += 1
        num_of_best_solutions = KEEP_BEST_POPULATION_PERCENT * POPULATION_SIZE
        # Create a new population by selecting parents, performing crossover, and applying mutation
        # Keep the best network
        new_population = population[:int(num_of_best_solutions)]
        number_of_solutions_from_best = 0.5 * POPULATION_SIZE
        best_population = population[:int(num_of_best_solutions)]
        i = 0
        while i < number_of_solutions_from_best:
            parent1 = select_parent(best_population)
            parent2 = select_parent(best_population)
            child = crossover(parent1, parent2)
            mutate(child)
            new_population.append(child)
            i += 1

        while len(new_population) < POPULATION_SIZE:
            parent1 = select_parent(population)
            parent2 = select_parent(population)
            child = crossover(parent1, parent2)
            mutate(child)
            new_population.append(child)

        population = new_population
        if check_if_stuck_in_local_max(best_solutions_score, NUM_OF_GENERATIONS_IN_LOCAL_MAX,
                                       LOCAL_MAX_NOT_IMPROVED_THRESHOLD):
            MUTATION_RATE = 0.3
            NUM_OF_MUTATIONS = 3
        else:
            MUTATION_RATE = 0.1
            NUM_OF_MUTATIONS = 1

    # Save the best-performing network
    best_network = population[0]
    with open('wnet0.txt', 'w') as file:  # Change file name accordingly
        for weight in best_network.weights:
            file.write(str(weight) + '\n')

    check_prediction('wnet0.txt', test_x, test_y)


def check_prediction(weights_file, test_x, test_y):
    # The code for running the trained network on test data
    weights = load_weights(weights_file)  # Change file name accordingly
    test_data = list(zip(test_x, test_y))
    test_predictions = run_network(weights, test_data)
    # Save Predictions to File
    counter = 0
    i = 0
    for prediction in test_predictions:
        if prediction == test_y[i]:
            counter += 1
        i += 1
    print("prediction rate:" + "{:.2f}".format((counter / len(test_y) * 100)) + "%")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Please provide two filename arguments for train and test.")
        sys.exit(1)

    TRAIN_FILE = sys.argv[1]
    TEST_FILE = sys.argv[2]

    # Split the shuffled data into training and testing files
    split_data('nn0.txt')
    # Genetic Algorithm
    buildnet0(TRAIN_FILE, TEST_FILE)



