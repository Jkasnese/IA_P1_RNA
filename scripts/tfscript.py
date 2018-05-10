""" Multilayer Perceptron.
A Multilayer Perceptron (Neural Network) implementation example using
TensorFlow library. This example is using the MNIST database of handwritten
digits (http://yann.lecun.com/exdb/mnist/).
Links:
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

# ------------------------------------------------------------------
#
# THIS EXAMPLE HAS BEEN RENAMED 'neural_network.py', FOR SIMPLICITY.
#
# ------------------------------------------------------------------

def(learning_rate, n_hidden, batch_size, vocab_size, mean, stddev):

    from __future__ import print_function
    import tensorflow as tf

#   Modify to get our data set
#    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)



    # Parameters
    #training_epochs = 15
    #display_step = 1

    # Network Parameters
    n_hidden_1 = n_hidden # 1st layer number of neurons
    n_input = vocab_size

    # tf Graph input
    X = tf.placeholder("float", [None, n_input])
    Y = tf.placeholder("float", [None, 1])

    # Store layers weight & bias

    # TODO: Vary mean and stddev to vary initial weights.
    # 68.27% of numbers between mean-stddev and mean+stddev
    # 99,7% between mean-2stddev and mean+2stddev

    # TODO: Armazenar pesos iniciais pra constar no relatório e usar em outros treinos.
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], mean=mean, stddev=steddev)), 
        'out': tf.Variable(tf.random_normal([n_hidden_1, 1], mean=mean, stddev=stddev))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'out': tf.Variable(tf.random_normal([1]))
    }


    # Create model
    def multilayer_perceptron(x):
        # Hidden fully connected layer with 256 neurons
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        # Output fully connected layer with a neuron for each class
        out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
        return out_layer

    # Construct model
    logits = multilayer_perceptron(X)

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)
    # Initializing the variables
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(mnist.train.num_examples/batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x,
                                                                Y: batch_y})
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
        print("Accuracy:", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels}))

        © 2018 GitHub, Inc.
        Terms
        Privacy
        Security
        Status
        Help

        Contact GitHub
        API
        Training
        Shop
        Blog
        About

    Press h to open a hovercard with more details.

