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

def rna (learning_rate, momentum, weights, n_hidden, batch_size, model_dir, optimizer, steps):
"""
    learning rate = float
    momentum = float
    weights = tf.Variable
    n_hidden = integer
    batch_size = integer
    model_dir = dir path
    optimizer = integer
    steps = integer
"""
    vocab_size, (x_train, y_train), (x_teste, y_teste) = text2panda_one_hot_representation_load.one_hot_representation_load(filename, MINIMUM_WORD_APPEARANCE = 5, translate=False)

    my_feature_columns = []

    for i in x_train.columns:
        tf.feature_column.categorical_column_with_vocabulary_list(
            key=i,
            vocabulary_list=x_train.columns
        )

    if (optimizer == 1):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    elif (optimizer == 2):
        optimizer = tf.train.MomemntumOptimizer(learning_rate, momentum)
    elif (optimizer == 3):
        optimizer = tf.train.AdagradOptimizer(learning_rate)
        
    tf.Variable(treinable=true)

    # TODO: mudar activation_fn pra configurar outras funções de ativação; 
    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        hidden_units=[n_hidden],
        n_classes=1,
        optimizer=optimizer)

    dataset = tf.data.Dataset.from_tensor_slices((dict(x_train), y_train))
    dataset = dataset.shuffle(buffer_size=9000).repeat().batch(batch_size) 

    classifier.train(
        input_fn=lambda:dataset, steps=steps)

    # Evaluate the model.
    eval_result = classifier.evaluate(
        input_fn=lambda:eval_input_fn(x_test, y_test, batch_size))

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Dentro de session, dps de run:
    save_path = saver.save(sess, "/tmp/model.ckpt")
    print("Model saved in path: %s" % save_path)

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))


    # Train with all dataset:
#    buffer_size=x_train. numero de linhas

   


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

