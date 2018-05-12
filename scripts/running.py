import tensorflow as tf
import txt2panda_onehotrepresentation as fetch_data
import tf_eager_tutorial as my_nn
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

# Validation
validation = False

## Optimizer
# 1 = SDG
# 2 = SDG with momentum
# 3 = Adagrad?
optimizer = 1

# Fetch data
vocab_size, n_input, (x_train, y_train), (x_teste, y_teste) = fetch_data.one_hot_representation_load(filename, MINIMUM_WORD_APPEARANCE = 5, translate=False)


w1, w2, b1, b2 = my_nn.tf_eager('Teste0', 'Football_Manager_2015', 0.1, 0.01, 256, 0, 100, 30, '~/tf/', 1, 0.0, 0.35)
weights, biases = gen_wb.init_values(w1, w2, b1, b2)
my_nn.tf_eager('Teste1', 'Football_Manager_2015', 0.1, 0.01, 256, 0, 100, 30, '~/tf/', 1, 0.0, 0.35, weights=weights, biases=biases)
