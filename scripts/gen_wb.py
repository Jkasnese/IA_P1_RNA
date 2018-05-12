import tensorflow as tf

def gen_wb (mean, stddev, vocab_size, n_hidden_1):

    # Store layers weight & bias

    # TODO: Vary mean and stddev to vary initial weights.
    # 68.27% of numbers between mean-stddev and mean+stddev
    # 99,7% between mean-2stddev and mean+2stddev

    # TODO: Armazenar pesos iniciais pra constar no relatório e usar em outros treinos.
    weights = {
        'h1': tf.Variable(tf.random_normal([vocab_size, n_hidden_1], mean=mean, stddev=stddev)), 
        'out': tf.Variable(tf.random_normal([n_hidden_1, 2], mean=mean, stddev=stddev))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1], stddev=stddev)),
        'out': tf.Variable(tf.random_normal([2]))
    }

    return weights, biases

def init_values(w1, w2, b1, b2):
    weights = {
        'h1': tf.Variable(initial_value=w1),
        'out': tf.Variable(initial_value=w2)
    }
    biases = {
        'b1': tf.Variable(initial_value=b1),
        'out': tf.Variable(initial_value=b2)
    }

    return weights, biases

