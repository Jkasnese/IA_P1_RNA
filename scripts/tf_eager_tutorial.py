from __future__ import print_function
import tensorflow as tf
import gen_wb
import txt2panda_onehotrepresentation as fetch_data
import matplotlib.pyplot as plt
import plot_graph as pltg

# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    return out_layer

def next_batch(step, batch_size, matrix):
    return matrix[step*batch_size:batch_size+step*batch_size]

def tf_eager(name, vocab_size, (x_train, y_train), (x_test, y_test), learning_rate, momentum, n_hidden, n_comments, batch_size, epochs, optimizer, mean, stddev, weights=None, biases=None):

    #Constructing data
    vocab_size, n_comments, (x_train, y_train), (x_test, y_test), (x_val, y_val), val_comments = fetch_data.one_hot_representation_load(filename, MINIMUM_WORD_APPEARANCE = 5, translate=False)

    # Parameters
    display_step = 1

    # Train placeholders
    X = tf.placeholder("float", [None, vocab_size], name="X")
    Y = tf.placeholder("float", [None, 2], name="Y")

    if (weights == None):
        weights, biases = gen_wb.gen_wb(mean, stddev, vocab_size, n_hidden)

    # Construct model
    logits = multilayer_perceptron(X, weights, biases)

    # Define loss
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
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

    with tf.Session() as sess:
        sess.run(init)

        initial_weights_1 = weights['h1'].read_value().eval()
        initial_weights_2 = weights['out'].read_value().eval()
        initial_biases_1 = biases['b1'].read_value().eval()
        initial_biases_2 = biases['out'].read_value().eval()

        train_loss_results = []

        # Training cycle
        for epoch in range(epochs):
            avg_cost = 0.
            total_batch = int(n_comments/batch_size)

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
                # Loss
                print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
                train_loss_results.append(avg_cost)


        print("Optimization Finished!")

        # Test model
        pred = tf.nn.softmax(logits)  # Apply softmax to logits
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))

        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy:", accuracy.eval({X: x_test, Y: y_test}))

        # TODO: Return arrays with data from training to compare with others and plot graphs.
        return initial_weights_1, initial_weights_2, initial_biases_1, initial_biases_2, acc


def nn_val_set(name, filename, learning_rate, momentum, n_hidden, n_comments, batch_size, epochs, optimizer, mean, stddev, weights=None, biases=None):
    
    # Parameters
    display_step = 1

    # Train placeholders
    X = tf.placeholder("float", [None, vocab_size], name="X")
    Y = tf.placeholder("float", [None, 2], name="Y")

    if (weights == None):
        weights, biases = gen_wb.gen_wb(mean, stddev, vocab_size, n_hidden)

    # Construct model
    logits = multilayer_perceptron(X, weights, biases)

    # Define train loss
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=logits, labels=Y))

    val_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=logits, labels=Z))
 

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

    with tf.Session() as sess:
        sess.run(init)

        initial_weights_1 = weights['h1'].read_value().eval()
        initial_weights_2 = weights['out'].read_value().eval()
        initial_biases_1 = biases['b1'].read_value().eval()
        initial_biases_2 = biases['out'].read_value().eval()

        train_loss_results = []
        train_accuracy_results = []
        validation_loss_results = []
        validation_accuracy_results = []

        # Training cycle
        for epoch in range(epochs):
            avg_cost = 0.
            total_batch = int(n_comments/batch_size)

            # Loop over all batches
            for step in range(total_batch):

                batch_x = next_batch(step, batch_size, x_train)
                batch_y = next_batch(step, batch_size, y_train)

                # Run optimization op (backprop) and cost op (to get loss value)
                _,c = sess.run([train_op, loss_op], feed_dict={X: batch_x, Y: batch_y})

                ## Compute average loss
                # Training loss
                avg_cost += c / total_batch


            # Model and define correct
            model = tf.nn.softmax(logits)
            correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, "float"))

            ## Train
            # Train loss
            train_loss_results.append(avg_cost)

            # training accuracy on training set
            train_acc = accuracy.eval({X: x_train, Y: y_train})
            train_accuracy_results.append(train_acc)

            ## Validation
            # Validation loss
            val_cost = sess.run([val_loss], feed_dict={X: x_val, Y: y_val})
            avg_val_cost += val_cost / val_comments

            # Compute validation accuracy on training set
            val_acc = accuracy.eval({X: x_val, Y: y_val})
            validation_accuracy_results.append(val_acc)

            # Display logs per epoch step
            if epoch % display_step == 0:
                # Loss
                print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
                print("Accuracy train:" + str(train_acc) + " Accuracy val: " + str(val_acc))

        print("Optimization Finished!")

        # Test model
        pred = tf.nn.softmax(logits)  # Apply softmax to logits
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))

        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy:", accuracy.eval({X: x_test, Y: y_test}))

        ## Ploting graphics
        # TODO: Call pg.plot_graph function
        fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
        fig.suptitle('Training Metrics')

        axes[0].set_ylabel("Loss", fontsize=14)
        axes[0].plot(train_loss_results, 'r')
        axes[0].plot(validation_loss_results, 'b')

        axes[1].set_ylabel("Accuracy", fontsize=14)
        axes[1].set_xlabel("Epoch", fontsize=14)
        axes[1].plot(train_accuracy_results, 'g')
        axes[1].plot(validation_accuracy_results, 'y')

        plt.show()


        # TODO: Return arrays with data from training to compare with others and plot graphs.
        return initial_weights_1, initial_weights_2, initial_biases_1, initial_biases_2, train_loss_results, train_accuracy_results, validation_loss_results, validation_accuracy_results
