# A simple linear neural network implementation in Tensorflow
# Trained on the sample datasets
import tensorflow as tf

class NeuralNetwork():
    def __init__(self, lr, ns, bs, il_nn, ol_nn, hl_nn):
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
        self.input_layer_number_neurons = il_nn
        self.output_layer_number_neurons = ol_nn
        self.hidden_layer_number_neurons = hl_nn
        self.train_data = []
        self.test_data = []
        self.weights = None
    def load_dataset(self, dataset, dps, ps):
        self.data_point_size = dps
        self.prediction_size = ps
        # TODO: Validate dataset dimensions
        # TODO: Load dataset
    def train(self):
        # TODO: Train on the given dataset
    def get_weights(self):
        # TODO: Get the weights of the trained model
        # TODO: Weights should be in the format defined in the danku contract
