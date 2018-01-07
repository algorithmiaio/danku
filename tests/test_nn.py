from dutils.neural_network import NeuralNetwork
from dutils.dataset import SampleCircleDataset, SampleSwirlDataset,\
    SampleHalfDividedDataset, SampleAcrossCornerDataset

def test_create_2_layer_nn():
    il_nn = 2
    hl_nn = []
    ol_nn = 1
    nn = NeuralNetwork(il_nn, hl_nn, ol_nn)
    nn.init_network()
    assert(not isinstance(nn.tf_weights, type(None)))
    assert(not isinstance(nn.tf_init, type(None)))
    assert(not isinstance(nn.tf_layers, type(None)))

def test_2lnn_load_dataset():
    il_nn = 2
    hl_nn = []
    ol_nn = 1
    nn = NeuralNetwork(il_nn, hl_nn, ol_nn)
    scd = SampleCircleDataset()
    scd.init_random_training_indexes()
    scd.danku_init()
    nn.load_dataset(scd)
    nn.init_network()
    assert(not isinstance(nn.tf_weights, type(None)))
    assert(not isinstance(nn.tf_init, type(None)))
    assert(not isinstance(nn.tf_layers, type(None)))

def test_train_2lnn():
    il_nn = 2
    hl_nn = []
    ol_nn = 2
    nn = NeuralNetwork(il_nn, hl_nn, ol_nn)
    scd = SampleHalfDividedDataset(training_percentage=0.8, max_num_data_groups=50)
    scd.init_random_training_indexes()
    scd.danku_init()
    nn.load_dataset(scd)
    nn.init_network()
    nn.train()
    assert(not isinstance(nn.tf_weights, type(None)))
    assert(not isinstance(nn.tf_init, type(None)))
    assert(not isinstance(nn.tf_layers, type(None)))

def test_create_3_layer_nn():
    il_nn = 2
    hl_nn = [4]
    ol_nn = 1
    nn = NeuralNetwork(il_nn, hl_nn, ol_nn)
    nn.init_network()
    assert(not isinstance(nn.tf_weights, type(None)))
    assert(not isinstance(nn.tf_init, type(None)))
    assert(not isinstance(nn.tf_layers, type(None)))

# TODO: Get multi-layer NN's working

def test_create_5_layer_nn():
    il_nn = 2
    hl_nn = [4,5,6]
    ol_nn = 1
    nn = NeuralNetwork(il_nn, hl_nn, ol_nn)
    nn.init_network()
    assert(not isinstance(nn.tf_weights, type(None)))
    assert(not isinstance(nn.tf_init, type(None)))
    assert(not isinstance(nn.tf_layers, type(None)))
