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

def tf_eager(vocab_size, learning_rate, momentum, n_hidden, n_comments, batch_size, epochs, optimizer, mean, stddev, x_train, y_train, x_test, y_test, weights=None, biases=None): 

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
        train_accuracy_results = []

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

            ### ACCURACY
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

        # Test model
        pred = tf.nn.softmax(logits)  # Apply softmax to logits
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))

        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        test_accuracy = accuracy.eval({X: x_test, Y: y_test})

        # TODO: Return arrays with data from training to compare with others and plot graphs.
        return initial_weights_1, initial_weights_2, initial_biases_1, initial_biases_2, test_accuracy, train_loss_results, train_accuracy_results

def nn_val_set(vocab_size, learning_rate, momentum, n_hidden, n_comments, val_comments, batch_size, epochs, optimizer, mean, stddev, x_train, y_train, x_test, y_test, x_val, y_val, weights=None, biases=None):
    
    # Parameters
    display_step = 1

    # Train placeholders
    X = tf.placeholder("float", [None, vocab_size], name="X")
    Y = tf.placeholder("int64", [None], name="Y")

    if (weights == None):
        weights, biases = gen_wb.gen_wb(mean, stddev, vocab_size, n_hidden)

    # Construct model
    logits = multilayer_perceptron(X, weights, biases)

    # Define train loss
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
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
        train_accuracy_results = []
        validation_loss_results = []
        validation_accuracy_results = []
        ttest_acc_res = []
        ttest_loss_res = []
    
        prev_val_cost = 9999999
        prev_val_acc = 0
        min_val_cost = 99999
        max_val_acc = 0

        # Training cycle
        for epoch in range(epochs):
            avg_cost = 0.
            avg_val_cost = 0.
            total_batch = int(n_comments/batch_size)

            # Loop over all batches
            for step in range(total_batch):

                batch_x = next_batch(step, batch_size, x_train)
                batch_y = next_batch(step, batch_size, y_train)

                # Run optimization op (backprop) and cost op (to get loss value)
                _,c = sess.run([train_op, loss_op], feed_dict={X: batch_x, Y: batch_y})

                ## Compute average LOSS
                # Training loss
                avg_cost += c / total_batch

            # Validation loss
            val_cost = sess.run([loss_op], feed_dict={X: x_val, Y: y_val})[0]

            # Test loss
            test_loss = sess.run([loss_op], feed_dict={X: x_test, Y: y_test})[0]

            ### ACCURACY
            # Model and define correct
            model = tf.nn.softmax(logits)
            correct = tf.equal(tf.argmax(model, 1), Y)
            accuracy = tf.reduce_mean(tf.cast(correct, "float"))

            ## Train
            # Train loss
            train_loss_results.append(avg_cost)

            # Training accuracy on training set
            train_acc = accuracy.eval({X: x_train, Y: y_train})
            train_accuracy_results.append(train_acc)

            # Compute validation accuracy on training set
            val_acc = accuracy.eval({X: x_val, Y: y_val})
            validation_accuracy_results.append(val_acc) 

            # Compute validation accuracy on training set
            ttest_acc = accuracy.eval({X: x_test, Y: y_test})
            ttest_acc_res.append(ttest_acc) 

            validation_loss_results.append(val_cost)
            ttest_loss_res.append(test_loss)

            if (val_acc > max_val_acc):
                max_val_acc = val_acc
                accepc = epoch

            if (val_cost < min_val_cost):
                min_val_cost = val_cost
                costepc = epoch

            ## Validation
            # Validation loss
     #       print(str(prev_val_acc) + "   |   " + str(val_acc))
       #     print(str(prev_val_cost) + "   |   " + str(val_cost))
            if (val_acc >= prev_val_acc):
                wepc = epoch
                w1 = weights['h1'].read_value().eval()
                w2 = weights['out'].read_value().eval()
                b1 = biases['b1'].read_value().eval()
                b2 = biases['out'].read_value().eval()
               #print("Entrou")
                prev_val_acc = val_acc
                prev_val_cost = val_cost
           # else:
                #print("Nao entrou")

        print ("Max Acc: " + str(max_val_acc) + "Epoca: " + str(accepc))
        print ("Min Cost: " + str(min_val_cost) + "Epoca: " + str(costepc))
        print ("Pesos adquiridos na Epoca: " + str(wepc))


        ## Standard training
        # Calculate accuracy
        test_accuracy = accuracy.eval({X: x_test, Y: y_test})
        #print(test_accuracy)

        ## Validation
        # Construct weights 
        saved_weights, saved_biases = gen_wb.init_values(w1, w2, b1, b2)

        # Construct final model (best accuracy on validation set)
        logits2 = multilayer_perceptron(X, saved_weights, saved_biases)

        loss_op2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits2, labels=Y))

        init2 = tf.global_variables_initializer()
        sess.run(init2)
        val_cost = sess.run([loss_op2], feed_dict={X: x_val, Y: y_val})[0]
        #print(val_cost)

        # Check if weights are really different
        #print(biases['out'].read_value().eval())
        #print(saved_biases['out'].read_value().eval())

        # Test model
        pred = tf.nn.softmax(logits2)  # Apply softmax to logits
        correct_prediction = tf.equal(tf.argmax(pred, 1), Y)

        #print(logits[0].read_value().eval())
        #print(logits2[0].read_value().eval())

        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        test_acc = accuracy.eval({X: x_test, Y: y_test})
        print(str(test_accuracy) + " | " + str(test_acc))

        # TODO: Return arrays with data from training to compare with others and plot graphs.
#        return initial_weights_1, initial_weights_2, initial_biases_1, initial_biases_2, test_accuracy, train_loss_results, train_accuracy_results
        return initial_weights_1, initial_weights_2, initial_biases_1, initial_biases_2, test_acc, train_loss_results, train_accuracy_results, validation_loss_results, validation_accuracy_results, ttest_acc_res, test_accuracy, ttest_loss_res
