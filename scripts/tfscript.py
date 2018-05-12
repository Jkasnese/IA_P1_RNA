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
import tensorflow as tf

def rna (name, learning_rate, momentum, weights, n_hidden, batch_size, model_dir, optimizer, steps, predictions=False):
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
    
    #Constructing data
    vocab_size, epoch_size, (x_train, y_train), (x_teste, y_teste) = text2panda_one_hot_representation_load.one_hot_representation_load(filename, MINIMUM_WORD_APPEARANCE = 5, translate=False)

    my_feature_columns = []

    for i in x_train.columns:
        tf.feature_column.categorical_column_with_vocabulary_list(
            key=i,
            vocabulary_list=x_train.columns
        )

    # Constructing tf.Graph
    if (optimizer == 1):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    elif (optimizer == 2):
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
    elif (optimizer == 3):
        optimizer = tf.train.AdagradOptimizer(learning_rate)


        

    # TODO: mudar activation_fn pra configurar outras funções de ativação; 
    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        hidden_units=[n_hidden],
        n_classes=1,
        optimizer=optimizer)

    dataset = tf.data.Dataset.from_tensor_slices((dict(x_train), y_train))
    dataset = dataset.shuffle(buffer_size=9000).repeat().batch(batch_size) 
    train_itr = dataset.make_one_shot_iterator().get_next()


    classifier.train(
        input_fn=lambda:train_itr, steps=steps)

    inputs = (x_test, y_test)
    dataset_test = tf.data.Dataset.from_tensor_slices(inputs)
    dataset_test = dataset.batch(batch_size)
    test_itr = dataset_test.make_one_shot_iterator().get_next()

    # Evaluate the model.
    eval_result = classifier.evaluate(
        input_fn=lambda:test_itr)

    if (predictions == True):
        predictions = classifier.predict(
        input_fn=lambda:eval_input_fn(predict_x,
                                      labels=None,
                                      batch_size=args.batch_size))

    weights = tf.get_variable("weights", trainable=True, initializer=tf.random_normal_initializer(stddev=stddev))


    # Add ops to save and restore all the variables.
    saver = tf.train.Saver(weights)
    fetches = classifier

    with tf.Session() as session:

        # Define trace options
        options = tf.RunOptions()
        options.output_partition_graphs = True
        options.trace_level = tf.RunOptions.SOFTWARE_TRACE

        # Define a container for the returned metadata.
        metadata = tf.RunMetadata()

        for step in range (steps):
            session.run(y, options=options, run_metadata=metadata)

            if (step % epoch_size == 0):
                #TODO: Mudar local
                saver.save(session, name, "~/Buba/", global_step=step)                






    # Dentro de session, dps de run, adicionar seguinte linha:
    save_path = saver.save(sess, "/tmp/model.ckpt")
    print("Model saved in path: %s" % save_path)

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))


    # Train with all dataset:
#    buffer_size=x_train. numero de linhas
