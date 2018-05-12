
from __future__ import print_function
import tensorflow as tf
import gen_wb
import txt2panda_onehotrepresentation as fetch_data

def tf_eager(name, filename, learning_rate, momentum, n_hidden, n_input, batch_size, epochs, model_dir, optimizer, mean, stddev, weights=None, biases=None):


#def rna (name, learning_rate, momentum, weights, n_hidden, batch_size, model_dir, optimizer, steps, predictions=False):
    """
    learning rate = float
    momentum = float
    weights = tf.Variable
    n_hidden = integer
    batch_size = integer
    model_dir = dir path
    optimizer = integer
    """
    
    #Constructing data
    vocab_size, n_input, (x_train, y_train), (x_test, y_test) = fetch_data.one_hot_representation_load(filename, MINIMUM_WORD_APPEARANCE = 5, translate=False)

    # Parameters
    display_step = 1

    # Train placeholders
    X = tf.placeholder("float", [None, vocab_size], name="X")
    Y = tf.placeholder("float", [None, 2], name="Y")

    if (weights == None):
        weights, biases = gen_wb.gen_wb(mean, stddev, vocab_size, n_hidden)


    # Create model
    def multilayer_perceptron(x, weights, biases):
        # Hidden fully connected layer with 256 neurons
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        # Output fully connected layer with a neuron for each class
        out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
        return out_layer

    # Construct model
    logits = multilayer_perceptron(X, weights, biases)

    # Define loss
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=Y))

    # Define Optimizer
    if (optimizer == 1):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    elif (optimizer == 2):
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum)
    elif (optimizer == 3):
        optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)

    train_op = optimizer.minimize(loss_op)

    # Initializing the variables
    init = tf.global_variables_initializer()

    def next_batch(step, batch_size, matrix):
        return matrix[step*batch_size:batch_size+step*batch_size]


    with tf.Session() as sess:
        sess.run(init)

        initial_weights_1 = weights['h1'].read_value().eval()
        initial_weights_2 = weights['out'].read_value().eval()
        initial_biases_1 = biases['b1'].read_value().eval()
        initial_biases_2 = biases['out'].read_value().eval()

        # Training cycle
        for epoch in range(epochs):
            avg_cost = 0.
            total_batch = int(n_input/batch_size)

            # Loop over all batches
            for step in range(total_batch):

                batch_x = next_batch(step, batch_size, x_train)
                batch_y = next_batch(step, batch_size, y_train)

                # Run optimization op (backprop) and cost op (to get loss value)
                _,c = sess.run([train_op, loss_op], feed_dict={X: batch_x, Y: batch_y})
                # Compute average loss
                avg_cost += c / total_batch

            # Display logs per epoch step
            if epoch % display_step == 0:

                print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))


        print("Optimization Finished!")

        # Test model
        pred = tf.nn.softmax(logits)  # Apply softmax to logits
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))

        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy:", accuracy.eval({X: x_test, Y: y_test}))

        return initial_weights_1, initial_weights_2, initial_biases_1, initial_biases_2
