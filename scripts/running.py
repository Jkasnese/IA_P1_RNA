import tensorflow as tf

relative_path = '/home/aluno/Buba/IA_P1_RNA/'
model_dir = relative_path + 'models/'

stddev = 0.35
weights = tf.get_variable("weights", trainable=true, initializer=tf.random_normal_initializer(stddev=stddev))

#(name, learning_rate, momentum, weights, n_hidden, batch_size, model_dir, optimizer, steps, preductions=False):
rna('Teste0', 0.01, 0, weights, 256, 100, model_dir, optimizer, steps, predictions=False)