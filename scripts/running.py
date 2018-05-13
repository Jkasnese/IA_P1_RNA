import tensorflow as tf
import txt2panda_onehotrepresentation as fetch_data
import tf_nn_eager as my_nn
import gen_wb
import plot_graph as my_plt

# Dir settings
relative_path = '/home/guiga/Desktop/Guiga/UEFEY/6_semestre_sd/IA/P1/'
exp = relative_path + 'exp/'

# File to get data from
filename = 'Grand_Theft_Auto_V'

# Fetch data
vocab_size, n_comments, x_train, y_train, x_test, y_test, x_val, y_val, val_comments = fetch_data.one_hot_representation_load(filename, MINIMUM_WORD_APPEARANCE = 5, translate=False)

### NN Default Hyperparameters
# Initial weights/biases. Given randonly from below parameters
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

## Optimizer
# 1 = SDG
# 2 = SDG with momentum
# 3 = Adagrad?
optimizer = 1


# # # # TESTING FOR CLOSE INITIAL WEIGHTS/BIASES # # # #
train_losses = []
train_accs = []
test_accs = []
n_runs = 100
plot_run = 14

for i in range (n_runs):
    w1, w2, b1, b2, test_acc, train_loss, train_acc = my_nn.tf_eager(vocab_size, learning_rate, momentum, n_hidden, n_comments, batch_size, epochs, optimizer, mean, stddev, x_train, y_train, x_test, y_test)
    test_accs.append(test_acc)
    if (i % plot_run == 0):
        train_losses.append(train_loss)
        train_accs.append(train_acc)

my_plt.acc_loss("Pesos Iniciais próximos", int(n_runs/plot_run), train_accs, train_losses, test_accs)
my_plt.test_acc("Pesos Iniciais próximos", test_accs)

max_acc = max(test_accs)
min_acc = min(test_accs)

with open (relative_path + exp + 'pesos_iniciais_proximos', 'w+') as pesos_prox:
    pesos_prox.write("Máxima acurácia: " + str(max_acc) + "\nMínima acurácia: " + str(min_acc) + "\nVariação máxima: " + str(max_acc - min_acc))


# # # # TESTING FOR WIDE INITIAL WEIGHTS/BIASES # # # #
train_losses = []
train_accs = []
test_accs = []
n_runs = 100
plot_run = 14

for i in range (n_runs):
    w1, w2, b1, b2, test_acc, train_loss, train_acc = my_nn.tf_eager(vocab_size, learning_rate, momentum, n_hidden, n_comments, batch_size, epochs, optimizer, mean, stddev, x_train, y_train, x_test, y_test)
    test_accs.append(test_acc)
    if (i % plot_run == 0):
        train_losses.append(train_loss)
        train_accs.append(train_acc)
    # Varing parameters to test:
    mean += 0.07
    stddev += 0.0035


my_plt.acc_loss("Pesos Iniciais distantes", int(n_runs/plot_run), train_accs, train_losses, test_accs)
my_plt.test_acc("Pesos Iniciais distantes", test_accs)

max_acc = max(test_accs)
min_acc = min(test_accs)

with open (relative_path + exp + 'pesos_iniciais_distantes', 'w+') as pesos_prox:
    pesos_prox.write("Máxima acurácia: " + str(max_acc) + "\nMínima acurácia: " + str(min_acc) + "\nVariação máxima: " + str(max_acc - min_acc))


"""
#TODO: save 3 best parameters to find a best combination later
# Restore parameters to initial values
mean = 0.0
stddev = 0.35


# # # # TESTING FOR NUMBER OF NEURONS # # # #
train_losses = []
train_accs = []
test_accs = []
n_runs = 7

# Initializing parameter to vary
n_hidden = 64

for i in range (n_runs):
    w1, w2, b1, b2, test_acc, train_loss, train_acc = my_nn.tf_eager(vocab_size, learning_rate, momentum, n_hidden, n_comments, batch_size, epochs, optimizer, mean, stddev, x_train, y_train, x_test, y_test)
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_accs.append(test_acc)

    # Varing parameters to test:
    n_hidden *= 4

my_plt.acc_loss("Acurácia e Erro por Época de treino", n_runs, train_accs, train_losses, test_accs)

# Restore parameters to initial values
n_hidden = 256


# # # # TESTING FOR LEARNING RATE # # # #
train_losses = []
train_accs = []
test_accs = []
n_runs = 7

# Initializing parameter to vary
learning_rate = 0.0001

for i in range (n_runs):
    w1, w2, b1, b2, test_acc, train_loss, train_acc = my_nn.tf_eager(vocab_size, learning_rate, momentum, n_hidden, n_comments, batch_size, epochs, optimizer, mean, stddev, x_train, y_train, x_test, y_test)
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_accs.append(test_acc)

    # Varing parameters to test:
    learning_rate *= 10    

my_plt.acc_loss("Acurácia e Erro por Época de treino", n_runs, train_accs, train_losses, test_accs)

# Restore parameters to initial values
learning_rate = 0.1


# # # # TESTING FOR BATCH_SIZE # # # #
train_losses = []
train_accs = []
test_accs = []
n_runs = 7

# Initializing parameter to vary
batch_size = 1

for i in range (n_runs):
    w1, w2, b1, b2, test_acc, train_loss, train_acc = my_nn.tf_eager(vocab_size, learning_rate, momentum, n_hidden, n_comments, batch_size, epochs, optimizer, mean, stddev, x_train, y_train, x_test, y_test)
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_accs.append(test_acc)

    # Varing parameters to test:
    batch_size += 110

my_plt.acc_loss("Acurácia e Erro por Época de treino", n_runs, train_accs, train_losses, test_accs)

# Restore parameters to initial values
batch_size = 100


# # # # TESTING FOR OPTIMIZERS # # # #
train_losses = []
train_accs = []
test_accs = []
n_runs = 3

# Initializing parameter to vary
# default is 1

for i in range (n_runs):
    w1, w2, b1, b2, test_acc, train_loss, train_acc = my_nn.tf_eager(vocab_size, learning_rate, momentum, n_hidden, n_comments, batch_size, epochs, optimizer, mean, stddev, x_train, y_train, x_test, y_test)
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_accs.append(test_acc)

    # Varing parameters to test:
    optimizer += 1

my_plt.acc_loss("Acurácia e Erro por Época de treino", n_runs, train_accs, train_losses, test_accs)

# Restore parameters to initial values
optimizer = 1


# # # # TESTING FOR MOMENTUM # # # #
train_losses = []
train_accs = []
test_accs = []
n_runs = 7

# Initializing parameter to vary
optimizer = 2
momentum = 0.00001

for i in range (n_runs):
    w1, w2, b1, b2, test_acc, train_loss, train_acc = my_nn.tf_eager(vocab_size, learning_rate, momentum, n_hidden, n_comments, batch_size, epochs, optimizer, mean, stddev, x_train, y_train, x_test, y_test)
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_accs.append(test_acc)

    # Varing parameters to test:
    momentum *= 10    

my_plt.acc_loss("Acurácia e Erro por Época de treino", n_runs, train_accs, train_losses, test_accs)

# Restore parameters to initial values
optimizer = 1



# # # # # # # # # # # # # # # # # # STAGE 2 - VALIDATION # # # # # # # # # # # # # # #


 


# First "batch" of comparison. First training creates weights/biases, subsequent trainings use the same.
# After 10 runs with hyperparameter variations, change the initial weights (by training a new session without weights) and train the rest.
# That way, we have combined mesures.
#w1, w2, b1, b2 = my_nn.tf_eager('Teste0', 'Football_Manager_2015', 0.1, 0.01, 256, 0, 100, 30, '~/tf/', 1, 0.0, 0.35)
#weights, biases = gen_wb.init_values(w1, w2, b1, b2)
#my_nn.tf_eager('Teste1', 'Football_Manager_2015', 0.1, 0.01, 256, 0, 100, 30, '~/tf/', 1, 0.0, 0.35, weights=weights, biases=biases)


#def nn_val_set(vocab_size, learning_rate, momentum, n_hidden, n_comments, batch_size, epochs, optimizer, mean, stddev, x_train, y_train, x_test, y_test, x_val, y_val, weights=None, biases=None):
#my_nn.nn_val_set(vocab_size, learning_rate, momentum, n_hidden, n_comments, val_comments, batch_size, epochs, optimizer, 0.0, 0.35, x_train, y_train, x_test, y_test, x_val, y_val)



# Plot comparison
#comparison_matrix = [10][epochs]
"""




