from dutils.neural_network import NeuralNetwork
from dutils.dataset import SampleCircleDataset, SampleSwirlDataset,\
    SampleHalfDividedDataset, SampleAcrossCornerDataset

def test_create_2_layer_nn():
    il_nn = 2
    hl_nn = []
    ol_nn = 2
    nn = NeuralNetwork(il_nn, hl_nn, ol_nn)
    nn.init_network()
    assert(not isinstance(nn.tf_weights, type(None)))
    assert(not isinstance(nn.tf_init, type(None)))
    assert(not isinstance(nn.tf_layers, type(None)))

def test_2lnn_load_dataset():
    il_nn = 2
    hl_nn = []
    ol_nn = 2
    nn = NeuralNetwork(il_nn, hl_nn, ol_nn)
    scd = SampleHalfDividedDataset()
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
    scd = SampleHalfDividedDataset(training_percentage=0.8)
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
    ol_nn = 2
    nn = NeuralNetwork(il_nn, hl_nn, ol_nn)
    nn.init_network()
    assert(not isinstance(nn.tf_weights, type(None)))
    assert(not isinstance(nn.tf_init, type(None)))
    assert(not isinstance(nn.tf_layers, type(None)))

def test_3lnn_load_dataset():
    il_nn = 2
    hl_nn = [4]
    ol_nn = 2
    nn = NeuralNetwork(il_nn, hl_nn, ol_nn)
    scd = SampleHalfDividedDataset()
    scd.init_random_training_indexes()
    scd.danku_init()
    nn.load_dataset(scd)
    nn.init_network()
    assert(not isinstance(nn.tf_weights, type(None)))
    assert(not isinstance(nn.tf_init, type(None)))
    assert(not isinstance(nn.tf_layers, type(None)))

def test_train_3lnn():
    il_nn = 2
    hl_nn = [4]
    ol_nn = 2
    nn = NeuralNetwork(il_nn, hl_nn, ol_nn)
    scd = SampleHalfDividedDataset(training_percentage=0.8)
    scd.init_random_training_indexes()
    scd.danku_init()
    nn.load_dataset(scd)
    nn.init_network()
    nn.train()
    assert(not isinstance(nn.tf_weights, type(None)))
    assert(not isinstance(nn.tf_init, type(None)))
    assert(not isinstance(nn.tf_layers, type(None)))

def test_create_5_layer_nn():
    il_nn = 2
    hl_nn = [4,5,6]
    ol_nn = 2
    nn = NeuralNetwork(il_nn, hl_nn, ol_nn)
    nn.init_network()
    assert(not isinstance(nn.tf_weights, type(None)))
    assert(not isinstance(nn.tf_init, type(None)))
    assert(not isinstance(nn.tf_layers, type(None)))

def test_5lnn_load_dataset():
    il_nn = 2
    hl_nn = [4,5,6]
    ol_nn = 2
    nn = NeuralNetwork(il_nn, hl_nn, ol_nn)
    scd = SampleHalfDividedDataset()
    scd.init_random_training_indexes()
    scd.danku_init()
    nn.load_dataset(scd)
    nn.init_network()
    assert(not isinstance(nn.tf_weights, type(None)))
    assert(not isinstance(nn.tf_init, type(None)))
    assert(not isinstance(nn.tf_layers, type(None)))

def test_train_5lnn():
    il_nn = 2
    hl_nn = [4,5,6]
    ol_nn = 2
    nn = NeuralNetwork(il_nn, hl_nn, ol_nn)
    scd = SampleHalfDividedDataset(training_percentage=0.8)
    scd.init_random_training_indexes()
    scd.danku_init()
    nn.load_dataset(scd)
    nn.init_network()
    nn.train()
    assert(not isinstance(nn.tf_weights, type(None)))
    assert(not isinstance(nn.tf_init, type(None)))
    assert(not isinstance(nn.tf_layers, type(None)))

def test_packing_weights():
    il_nn = 2
    hl_nn = [2,3]
    ol_nn = 2
    nn = NeuralNetwork(il_nn, hl_nn, ol_nn)
    weights = [[[1.9911426, -1.2063], [1.9911426, -1.2063]],\
        [[0.23531701, 2.9445081], [0.23531701, 2.9445081],\
        [0.23531701, 2.9445081]], [[0.065952107, 0.82053429, -0.21171442],\
        [0.065952107, 0.82053429, -0.21171442]]]

    expected_packed_weights = [1.9911426, -1.2063, 1.9911426, -1.2063,\
        0.23531701, 2.9445081, 0.23531701, 2.9445081, 0.23531701, 2.9445081,\
        0.065952107, 0.82053429, -0.21171442, 0.065952107, 0.82053429,\
        -0.21171442]

    packed_weights = nn.pack_weights(weights)

    for v in range(len(packed_weights)):
        assert(packed_weights[v] == expected_packed_weights[v])

def test_unpacking_weights():
    il_nn = 2
    hl_nn = [2,3]
    ol_nn = 2
    nn = NeuralNetwork(il_nn, hl_nn, ol_nn)
    packed_weights = [1.9911426, -1.2063, 1.9911426, -1.2063, 0.23531701,\
    2.9445081, 0.23531701, 2.9445081, 0.23531701, 2.9445081, 0.065952107,\
    0.82053429, -0.21171442, 0.065952107, 0.82053429, -0.21171442]

    expected_unpacked_weights = [[[1.9911426, -1.2063], [1.9911426, -1.2063]],\
        [[0.23531701, 2.9445081], [0.23531701, 2.9445081],\
        [0.23531701, 2.9445081]], [[0.065952107, 0.82053429, -0.21171442],\
        [0.065952107, 0.82053429, -0.21171442]]]

    unpacked_weights = nn.unpack_weights(packed_weights, il_nn, hl_nn, ol_nn)

    # g for group, d for data point and v for value
    for g in range(len(unpacked_weights)):
        for d in range(len(unpacked_weights[g])):
            for v in range(len(unpacked_weights[g][d])):
                assert(unpacked_weights[g][d][v] ==\
                    expected_unpacked_weights[g][d][v])
