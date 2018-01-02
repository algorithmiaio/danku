# A simple linear neural network implementation in Tensorflow
# Trained on the sample datasets
import tensorflow as tf
from . import SampleSwirlDataset, SampleCircleDataset,\
SampleHalfDividedDataset, SampleAcrossCornerDataset

class NeuralNetwork():
    def __init__(self, il_nn, ol_nn, hl_nn, lr=0.1, ns=100, bs=10, ds=10):
        # Making Python type-safe!
        if not isinstance(lr, int) and not isinstance(lr, float):
            raise Exception("Learning rate must be an integer or float.")
        if not isinstance(ns, int):
            raise Exception("Number of steps must be an integer.")
        if not isinstance(bs, int):
            raise Exception("Batch size must be an integer.")
        if not isinstance(il_nn. int):
            raise Exception("Input layer number of neurons must be an integer.")
        if not isinstance(ol_nn, int):
            raise Exception("Output layer number of neurons must be an integer.\
            ")
        if not isinstance(hl_nn, list):
            e_msg = "Hidden layer number of neurons must be a a list of\
                integers."
            raise Exception(e_msg)
            for i in hl_nn:
                if not isinstance(i, int):
                    raise Exception(e_msg)
        self.learning_rate = lr
        self.number_steps = ns
        self.batch_size = bs
        self.display_step = ds
        self.input_layer_number_neurons = il_nn
        self.output_layer_number_neurons = ol_nn
        self.hidden_layer_number_neurons = hl_nn
        self.train_data = []
        self.test_data = []
        self.weights = None

    def init_network(self):
        x_vector = tf.placeholder("float", [None, self.input_layer_number_neurons])
        y_vector = tf.placeholder("float", [None, self.output_layer_number_neurons])

        # Initialize weight variables
        weights = {}
        for i in range(len(self.hidden_layer_number_neurons)):
            if i == 0:
                weights["h" + str(i+1)] = tf.Variable(tf.random_normal([\
                    self.input_layer_number_neurons,\
                    self.hidden_layer_number_neurons[i]])),
            else:
                weights["h" + str(i+1)] = tf.Variable(tf.random_normal([\
                    self.hidden_layer_number_neurons[i-1],\
                    self.hidden_layer_number_neurons[i]])),

        if len(self.hidden_layer_number_neurons) == 0:
            weights["out"] = tf.Variable(tf.random_normal([\
                self.input_layer_number_neurons,\
                self.output_layer_number_neurons]))
        else:
            weights["out"] = tf.Variable(tf.random_normal([\
                self.hidden_layer_number_neurons[-1],
                self.output_layer_number_neurons]))

        # Initialize layers
        layers = {}
        for i in range(len(self.hidden_layer_number_neurons)):
            if i == 0:
                layers["l" + str(i+1)] = tf.matmul(\
                    x_vector, weights["h" + str(i+1)]))
            else:
                layers["l" + str(i+1)] = tf.matmul(\
                    weights["h" + str(i-1)], weights["h" + str(i)]))

        if len(self.hidden_layer_number_neurons) == 0:
            layers["out"] = tf.matmul(
                x_vector, weights['out'])
        else:
            layers["out"] = tf.matmul(\
                weights["h" + str(len(self.hidden_layer_number_neurons))],\
                weights['out'])

        # Construct model
        logits = neural_net(x_vector)
        prediction = tf.nn.relu(logits)

        # Loss and optimizer
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=y_vector))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss_op)

        # Model evaluation
        compare_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_vector, 1))
        accuracy = tf.reduce_mean(tf.cast(compare_pred, tf.float32))

        # Initialize tf variables
        init = tf.global_variables_initializer()

    def train(self):
        with tf.Session() as sess:
            sess.run(init)
            for step in range(1, self.number_steps+1):
                x_train_vector = map(lambda x: x[:self.prediction_size],\
                    self.train_data)
                y_train_vector = map(lambda x: x[self.prediction_size:],\
                    self.train_data)
                # Backpropogation
                sess.run(train_op,
                    feed_dict={X: x_train_vector, Y: y_train_vector})
                if step % self.display_step == 0 or step == 1:
                    # Calculate loss and accuracy
                    loss, acc = sess.run([loss_op, accuracy],\
                        feed_dict={X: x_train_vector, Y: y_train_vector})
                    print("Step " + str(step) + ", Loss= " + \
                          "{:.4f}".format(loss) + ", Training Accuracy= " + \
                          "{:.3f}".format(acc))

            print("Training Finished!")

            x_test_vector = map(lambda x: x[:self.prediction_size],\
                self.test_data)
            y_test_vector = map(lambda x: x[self.prediction_size:],\
                self.test_data)
            # Get accuracy with test dataset
            print("Testing Accuracy:", \
                sess.run(accuracy,\
                    feed_dict={X: x_test_vector, Y: y_test_vector}))

    def load_dataset(self, train_data, dps, ps):
        self.data_point_size = dps
        self.prediction_size = ps
        # Validate dataset dimensions
        for data_point in dataset:
            assert(len(data_point) == dps)
        # Load dataset
        self.train_data = train_data

    def get_weights(self):
        # TODO: Get the weights of the trained model
        # TODO: Weights should be in the format defined in the danku contract
