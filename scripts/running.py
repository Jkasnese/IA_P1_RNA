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
best_hidden = 0

# Learning rate
learning_rate = 0.1
best_learning = 0

# Momentum
momentum = 0.01
best_momentum = 0

# Number of elements per batch
batch_size = 100
epochs = 30
best_batch = 0

## Optimizer
# 1 = SDG
# 2 = SDG with momentum
# 3 = Adagrad?
optimizer = 1


# # # # TESTING FOR CLOSE INITIAL WEIGHTS/BIASES # # # #
train_losses = []
train_accs = []
test_accs = []
n_runs = 20
plot_run = 3

for i in range (n_runs):
    w1, w2, b1, b2, test_acc, train_loss, train_acc = my_nn.tf_eager(vocab_size, learning_rate, momentum, n_hidden, n_comments, batch_size, epochs, optimizer, mean, stddev, x_train, y_train, x_test, y_test)
    test_accs.append(test_acc)
    if (i % plot_run == 0):
        train_losses.append(train_loss)
        train_accs.append(train_acc)
    print(i)

my_plt.acc_loss("Pesos Iniciais próximos", int(n_runs/plot_run), train_accs, train_losses, test_accs)
my_plt.test_acc("Pesos Iniciais próximos", test_accs)

max_acc = max(test_accs)
min_acc = min(test_accs)

with open (exp + 'pesos_iniciais_proximos', 'w+') as pesos_prox:
    pesos_prox.write("Máxima acurácia: " + str(max_acc) + "\nMínima acurácia: " + str(min_acc) + "\nVariação máxima: " + str(max_acc - min_acc))


# # # # TESTING FOR WIDE INITIAL WEIGHTS/BIASES # # # #
train_losses = []
train_accs = []
test_accs = []
n_runs = 20
plot_run = 3

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

with open (exp + 'pesos_iniciais_distantes', 'w+') as pesos_prox:
    pesos_prox.write("Máxima acurácia: " + str(max_acc) + "\nMínima acurácia: " + str(min_acc) + "\nVariação máxima: " + str(max_acc - min_acc))

#TODO: save 3 best parameters to find a best combination later
# Restore parameters to initial values
mean = 0.0
stddev = 0.35


def vary(name, parameter_init, vary, n_runs=20, plot_run=3):
    train_losses = []
    train_accs = []
    test_accs = []
    n_runs = 100
    plot_run = 14
    max_acc = 0
    best_parameter = 0

    # Generating weights and biases
    w1, w2, b1, b2, test_acc, train_loss, train_acc = my_nn.tf_eager(vocab_size, learning_rate, momentum, n_hidden, n_comments, batch_size, epochs, optimizer, mean, stddev, x_train, y_train, x_test, y_test)
    test_accs.append(test_acc)
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    # Saving initial weights and biases
    weights, biases = gen_wb.init_values(w1, w2, b1, b2)

    # Running the rest with same weights and biases
    for i in range (n_runs-1):
        w12, w22, b12, b22, test_acc, train_loss, train_acc = my_nn.tf_eager(vocab_size, learning_rate, momentum, n_hidden, n_comments, batch_size, epochs, optimizer, mean, stddev, x_train, y_train, x_test, y_test, weights=weights, biases=biases)
        test_accs.append(test_acc)
        if (i % plot_run == 0):
            train_losses.append(train_loss)
            train_accs.append(train_acc)

        # Save best parameter value
        if (test_acc > max_acc):
            max_acc = test_acc
            best_parameter = parameter
        print(i)

        # Varing parameters to test:
        parameter += vary

    my_plt.acc_loss(name, int(n_runs/plot_run), train_accs, train_losses, test_accs)
    my_plt.test_acc(name, test_accs)

    min_acc = min(test_accs)

    with open (exp + name, 'w+') as pesos_prox:
        pesos_prox.write("Máxima acurácia: " + str(max_acc) + "\nMínima acurácia: " + str(min_acc) + "\nVariação máxima: " + str(max_acc - min_acc))
        pesos_prox.write("Melhor parametro de " + name + " :" + str(best_parameter))

    return best_parameter


# # # # TESTING FOR NUMBER OF NEURONS # # # #

#def vary(name, parameter_init, operation, vary, n_runs=100, plot_run=14):
best_hidden = vary("Numero_de_Neuronios", 31, 25)

# # # # TESTING FOR LEARNING RATE # # # #
best_learning = vary("Taxa_de_Aprendizagem", 0.001, 0.25)

# # # # TESTING FOR BATCH_SIZE # # # #
best_batch = vary("Tamanho_do_Batch", 1, 10)

# # # # TESTING FOR MOMENTUM # # # #
optimizer = 2
best_momentum = vary("Variacao_do_Momento", 0.001, 0.05)

# Restore parameters to initial values
optimizer = 1

# # # # TESTING FOR OPTIMIZERS # # # #
test_acc_temp = []
train_loss_temp = []
train_acc_temp = []

train_losses = []
train_accs = []
max_acc_1 = 0

train_losses_2 = []
train_accs_2 = []
max_acc_2 = 0

train_losses_3 = []
train_accs_3 = []
max_acc_3 = 0

for i in range(5):
    # Generating weights and biases
    w1, w2, b1, b2, test_acc_temp, train_loss_temp, train_acc_temp = my_nn.tf_eager(vocab_size, learning_rate, momentum, n_hidden, n_comments, batch_size, epochs, optimizer, mean, stddev, x_train, y_train, x_test, y_test)

    # Saving initial weights and biases
    weights, biases = gen_wb.init_values(w1, w2, b1, b2)

    # Save best parameter value
    if (test_acc_temp > max_acc_1):
        max_acc_1 = test_acc_temp
        train_losses = train_loss_temp
        train_accs = train_acc_temp

    # Varing parameters to test:
    optimizer += 1

    w12, w22, b12, b22, test_acc_temp, train_loss_temp, train_acc_temp = my_nn.tf_eager(vocab_size, learning_rate, momentum, n_hidden, n_comments, batch_size, epochs, optimizer, mean, stddev, x_train, y_train, x_test, y_test, weights=weights, biases=biases)

    # Save best parameter value
    if (test_acc_temp > max_acc_2):
        max_acc_2 = test_acc_temp
        train_losses_2 = train_loss_temp
        train_accs_2 = train_acc_temp
    
    # Varing parameters to test:
    optimizer += 1

    w12, w22, b12, b22, test_acc_temp, train_loss_temp, train_acc_temp = my_nn.tf_eager(vocab_size, learning_rate, momentum, n_hidden, n_comments, batch_size, epochs, optimizer, mean, stddev, x_train, y_train, x_test, y_test, weights=weights, biases=biases)

    # Save best parameter value
    if (test_acc_temp > max_acc_2):
        max_acc_2 = test_acc_temp
        train_losses_2 = train_loss_temp
        train_accs_2 = train_acc_temp
    
    # Varing parameters to test:
    optimizer = 1

train_losses_matrix = []
train_accs_matrix = []
test_accs_array = []

train_losses_matrix.append(train_losses)
train_losses_matrix.append(train_losses_2)
train_losses_matrix.append(train_losses_3)

train_accs_matrix.append(train_accs)
train_accs_matrix.append(train_accs_2)
train_accs_matrix.append(train_accs_3)

test_accs_array.append(max_acc)
test_accs_array.append(max_acc_2)
test_accs_array.append(max_acc_3)

my_plt.acc_loss(name, 3, train_accs_matrix, train_losses_matrix, test_accs)

max_acc = max(test_accs_array)
min_acc = min(test_accs_array)

with open (relative_path + exp + 'Otimizadores', 'w+') as pesos_prox:
    pesos_prox.write("Máxima acurácia: " + str(max_acc) + "\nMínima acurácia: " + str(min_acc) + "\nVariação máxima: " + str(max_acc - min_acc))
    pesos_prox.write("Melhor parametro de " + 'otimizadores' + " :" + str(test_accs_array.index(max(test_accs_array)) +1) )


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




