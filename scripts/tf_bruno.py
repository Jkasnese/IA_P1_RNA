import tensorflow as tf

import json

import numpy as np

#

#trata o json que eh a entrada

#


with open('../item3treino') as data_file:

    data = json.load(data_file)


with open('../item3teste') as data_teste:

    teste = json.load(data_teste)



def get_batch(i,batch_size, dado):

    batches = []

    results = []

    texts = dado[i*batch_size:i*batch_size+batch_size]


    for j in range(len(texts)):

        batches.append(texts[j]["entrada"])

        results.append(texts[j]["saida"]) 

    

     

    return np.array(batches),np.array(results)



# Parameters

learning_rate = 0.01

training_epochs = 30

batch_size = 150

display_step = 1


# Network Parameters

n_hidden_1 = 100 # 1st layer number of features

n_hidden_2 = 100 # 2nd layer number of features

n_input = len(data[0]["entrada"]) # Words in vocab

n_classes = 2 # Categories: graphics, sci.space and baseball


input_tensor = tf.placeholder(tf.float32,[None, n_input],name="input")

output_tensor = tf.placeholder(tf.float32,[None, n_classes],name="output") 



def multilayer_perceptron(input_tensor, weights, biases):

    layer_1_multiplication = tf.matmul(input_tensor, weights['h1'])

    layer_1_addition = tf.add(layer_1_multiplication, biases['b1'])

    layer_1 = tf.nn.relu(layer_1_addition)

    

    # Hidden layer with RELU activation

    layer_2_multiplication = tf.matmul(layer_1, weights['h2'])

    layer_2_addition = tf.add(layer_2_multiplication, biases['b2'])

    layer_2 = tf.nn.relu(layer_2_addition)

    

    # Output layer 

    out_layer_multiplication = tf.matmul(layer_2, weights['out'])

    out_layer_addition = out_layer_multiplication + biases['out']

    

    return out_layer_addition



# Store layers weight & bias

weights = {

    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),

    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),

    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))

}

biases = {

    'b1': tf.Variable(tf.random_normal([n_hidden_1])),

    'b2': tf.Variable(tf.random_normal([n_hidden_2])),

    'out': tf.Variable(tf.random_normal([n_classes]))

}


# Construct model

prediction = multilayer_perceptron(input_tensor, weights, biases)


# Define loss and optimizer

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=output_tensor))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)


# Initializing the variables

init = tf.global_variables_initializer()


# Launch the graph

with tf.Session() as sess:

    sess.run(init)


    # Training cycle

    for epoch in range(training_epochs):

        avg_cost = 0.

        total_batch = int(len(data)/batch_size)

        # Loop over all batches

       

        for i in range(total_batch):

            batch_x,batch_y = get_batch(i,batch_size, data)

            

            # Run optimization op (backprop) and cost op (to get loss value)

            c,_ = sess.run([loss,optimizer], feed_dict={input_tensor: batch_x,output_tensor:batch_y})

            # Compute average loss

            avg_cost += c / total_batch

        # Display logs per epoch step

        if epoch % display_step == 0:

            print("Epoch:", '%04d' % (epoch+1), "loss=", \

                "{:.9f}".format(avg_cost))

    print("Optimization Finished!")


    # Test model

    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(output_tensor, 1))

    # Calculate accuracy

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    total_test_data = len(teste)

    batch_x_test,batch_y_test = get_batch(0,total_test_data, teste)

    print("Teste Accuracy:", accuracy.eval({input_tensor: batch_x_test, output_tensor: batch_y_test}))



    # Test model

    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(output_tensor, 1))

    # Calculate accuracy

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    total_test_data = len(data)

    batch_x_test,batch_y_test = get_batch(0,total_test_data, data)

    print("Train Accuracy:", accuracy.eval({input_tensor: batch_x_test, output_tensor: batch_y_test}))
