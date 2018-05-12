import tensorflow as tf
import txt2panda_onehotrepresentation as fetch_data
import tf_nn_eager as my_nn
import gen_wb

# Dir settings
relative_path = '/home/guiga/Desktop/Guiga/UEFEY/6_semestre_sd/IA/P1/'
model_dir = relative_path + 'models/'

# File to get data from
filename = 'Football_Manager_2015'

### NN Hyperparameters
## Initialization hyperparameters

# Initial weights/biases
stddev = 0.35
mean = 0.0

# Number of neurons in hidden layer
n_hidden = 256

# Learning rate
learning_rate = 0.1
# Momentum
momentum = 0.01

# Number of elements per batch
batch_size = 100
epochs = 30

# Validation
validation = False

## Optimizer
# 1 = SDG
# 2 = SDG with momentum
# 3 = Adagrad?
optimizer = 1



# FOR A GIVEN PARAMETER

# Fetch data
#dimension, total_comments, x_train, y_train, x_test, y_test, x_val, y_val, val_comments
vocab_size, n_comments, x_train, y_train, x_test, y_test, x_val, y_val, val_comments = fetch_data.one_hot_representation_load(filename, MINIMUM_WORD_APPEARANCE = 5, translate=False)

# First "batch" of comparison. First training creates weights/biases, subsequent trainings use the same.
# After 10 runs with hyperparameter variations, change the initial weights (by training a new session without weights) and train the rest.
# That way, we have combined mesures.
#w1, w2, b1, b2 = my_nn.tf_eager('Teste0', 'Football_Manager_2015', 0.1, 0.01, 256, 0, 100, 30, '~/tf/', 1, 0.0, 0.35)
#weights, biases = gen_wb.init_values(w1, w2, b1, b2)
#my_nn.tf_eager('Teste1', 'Football_Manager_2015', 0.1, 0.01, 256, 0, 100, 30, '~/tf/', 1, 0.0, 0.35, weights=weights, biases=biases)


#def nn_val_set(name, vocab_size, learning_rate, momentum, n_hidden, n_comments, batch_size, epochs, optimizer, mean, stddev, x_train, y_train, x_test, y_test, x_val, y_val, weights=None, biases=None):
my_nn.nn_val_set('Teste1', vocab_size, learning_rate, momentum, n_hidden, n_comments, val_comments, batch_size, epochs, optimizer, 0.0, 0.35, x_train, y_train, x_test, y_test, x_val, y_val)



# Plot comparison
#comparison_matrix = [10][epochs]



