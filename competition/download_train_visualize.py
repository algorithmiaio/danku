import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from dutils.dataset import Dataset
from dutils.neural_network import NeuralNetwork
from web3 import Web3, HTTPProvider, IPCProvider
from matplotlib import pyplot as plt
import numpy as np

w_scale = 1000 # Scale up weights by 1000x
b_scale = 1000 # Scale up biases by 1000x

def scale_packed_data(data, scale):
    # Scale data and convert it to an integer
    return list(map(lambda x: int(x*scale), data))

print("Connecting to geth...\n")
web3 = Web3(HTTPProvider('http://localhost:8545'))
print("Connected to the geth node!\n")

abi = [{"constant":True,"inputs":[],"name":"init1_block_height","outputs":[{"name":"","type":"uint256"}],"payable":False,"stateMutability":"view","type":"function"},{"constant":False,"inputs":[],"name":"init2","outputs":[],"payable":False,"stateMutability":"nonpayable","type":"function"},{"constant":False,"inputs":[{"name":"submission_index","type":"uint256"}],"name":"evaluate_model","outputs":[],"payable":False,"stateMutability":"nonpayable","type":"function"},{"constant":True,"inputs":[{"name":"submission_index","type":"uint256"},{"name":"data","type":"int256[3][]"}],"name":"model_accuracy","outputs":[{"name":"","type":"int256"}],"payable":False,"stateMutability":"view","type":"function"},{"constant":True,"inputs":[],"name":"get_training_index","outputs":[{"name":"","type":"uint256[16]"}],"payable":False,"stateMutability":"view","type":"function"},{"constant":True,"inputs":[],"name":"evaluation_stage_block_size","outputs":[{"name":"","type":"uint256"}],"payable":False,"stateMutability":"view","type":"function"},{"constant":True,"inputs":[{"name":"","type":"uint256"},{"name":"","type":"uint256"}],"name":"test_data","outputs":[{"name":"","type":"int256"}],"payable":False,"stateMutability":"view","type":"function"},{"constant":True,"inputs":[],"name":"get_testing_index","outputs":[{"name":"","type":"uint256[4]"}],"payable":False,"stateMutability":"view","type":"function"},{"constant":False,"inputs":[{"name":"_test_data_groups","type":"int256[]"},{"name":"_test_data_group_nonces","type":"int256"}],"name":"reveal_test_data","outputs":[],"payable":False,"stateMutability":"nonpayable","type":"function"},{"constant":True,"inputs":[{"name":"paymentAddress","type":"address"},{"name":"num_neurons_input_layer","type":"uint256"},{"name":"num_neurons_output_layer","type":"uint256"},{"name":"num_neurons_hidden_layer","type":"uint256[]"},{"name":"weights","type":"int256[]"},{"name":"biases","type":"int256[]"}],"name":"get_submission_id","outputs":[{"name":"","type":"uint256"}],"payable":False,"stateMutability":"view","type":"function"},{"constant":True,"inputs":[],"name":"best_submission_index","outputs":[{"name":"","type":"uint256"}],"payable":False,"stateMutability":"view","type":"function"},{"constant":True,"inputs":[],"name":"use_test_data","outputs":[{"name":"","type":"bool"}],"payable":False,"stateMutability":"view","type":"function"},{"constant":False,"inputs":[{"name":"_train_data_groups","type":"int256[]"},{"name":"_train_data_group_nonces","type":"int256"}],"name":"init3","outputs":[],"payable":False,"stateMutability":"nonpayable","type":"function"},{"constant":True,"inputs":[{"name":"l_nn","type":"uint256[]"},{"name":"input_layer","type":"int256[]"},{"name":"hidden_layers","type":"int256[]"},{"name":"output_layer","type":"int256[]"},{"name":"weights","type":"int256[]"},{"name":"biases","type":"int256[]"}],"name":"forward_pass2","outputs":[{"name":"","type":"int256[]"}],"payable":False,"stateMutability":"pure","type":"function"},{"constant":False,"inputs":[{"name":"_hashed_data_groups","type":"bytes32[20]"},{"name":"accuracy_criteria","type":"int256"},{"name":"organizer_refund_address","type":"address"}],"name":"init1","outputs":[],"payable":False,"stateMutability":"nonpayable","type":"function"},{"constant":True,"inputs":[],"name":"organizer","outputs":[{"name":"","type":"address"}],"payable":False,"stateMutability":"view","type":"function"},{"constant":True,"inputs":[],"name":"init_level","outputs":[{"name":"","type":"uint256"}],"payable":False,"stateMutability":"view","type":"function"},{"constant":True,"inputs":[{"name":"","type":"uint256"}],"name":"testing_partition","outputs":[{"name":"","type":"uint256"}],"payable":False,"stateMutability":"view","type":"function"},{"constant":True,"inputs":[],"name":"get_train_data_length","outputs":[{"name":"","type":"uint256"}],"payable":False,"stateMutability":"view","type":"function"},{"constant":True,"inputs":[],"name":"best_submission_accuracy","outputs":[{"name":"","type":"int256"}],"payable":False,"stateMutability":"view","type":"function"},{"constant":False,"inputs":[],"name":"finalize_contract","outputs":[],"payable":False,"stateMutability":"nonpayable","type":"function"},{"constant":True,"inputs":[],"name":"contract_terminated","outputs":[{"name":"","type":"bool"}],"payable":False,"stateMutability":"view","type":"function"},{"constant":True,"inputs":[],"name":"init3_block_height","outputs":[{"name":"","type":"uint256"}],"payable":False,"stateMutability":"view","type":"function"},{"constant":True,"inputs":[],"name":"get_submission_queue_length","outputs":[{"name":"","type":"uint256"}],"payable":False,"stateMutability":"view","type":"function"},{"constant":False,"inputs":[{"name":"payment_address","type":"address"},{"name":"num_neurons_input_layer","type":"uint256"},{"name":"num_neurons_output_layer","type":"uint256"},{"name":"num_neurons_hidden_layer","type":"uint256[]"},{"name":"weights","type":"int256[]"},{"name":"biases","type":"int256[]"}],"name":"submit_model","outputs":[],"payable":False,"stateMutability":"nonpayable","type":"function"},{"constant":True,"inputs":[{"name":"","type":"uint256"}],"name":"training_partition","outputs":[{"name":"","type":"uint256"}],"payable":False,"stateMutability":"view","type":"function"},{"constant":True,"inputs":[],"name":"reveal_test_data_groups_block_size","outputs":[{"name":"","type":"uint256"}],"payable":False,"stateMutability":"view","type":"function"},{"constant":False,"inputs":[],"name":"cancel_contract","outputs":[],"payable":False,"stateMutability":"nonpayable","type":"function"},{"constant":True,"inputs":[],"name":"model_accuracy_criteria","outputs":[{"name":"","type":"int256"}],"payable":False,"stateMutability":"view","type":"function"},{"constant":True,"inputs":[{"name":"","type":"uint256"},{"name":"","type":"uint256"}],"name":"train_data","outputs":[{"name":"","type":"int256"}],"payable":False,"stateMutability":"view","type":"function"},{"constant":True,"inputs":[],"name":"get_test_data_length","outputs":[{"name":"","type":"uint256"}],"payable":False,"stateMutability":"view","type":"function"},{"constant":True,"inputs":[],"name":"submission_stage_block_size","outputs":[{"name":"","type":"uint256"}],"payable":False,"stateMutability":"view","type":"function"},{"inputs":[],"payable":False,"stateMutability":"nonpayable","type":"constructor"},{"payable":True,"stateMutability":"payable","type":"fallback"}]

contract_tx = "0x9A0991fc223dFFE420e08f15b88a593a3b8D44B8"

# Get contract instance
danku = web3.eth.contract(abi, contract_tx)

print("Downloading training data from the contract...\n")
# Get training data
contract_train_data_length = danku.call().get_train_data_length()
contract_train_data = []
for i in range(contract_train_data_length):
    for j in range(3):
        contract_train_data.append(danku.call().train_data(i,j))
print("Downloading testing data from the contract...\n")
# Get testing data
contract_test_data_length = danku.call().get_test_data_length()
contract_test_data = []
for i in range(contract_test_data_length):
    for j in range(3):
        contract_test_data.append(danku.call().test_data(i,j))
ds = Dataset()
ds.dps = 3
contract_train_data = ds.unpack_data(contract_train_data)
contract_test_data = ds.unpack_data(contract_test_data)
print("Download finished!\n")
print("Contract training data:\n" + str(contract_train_data) + "\n")
print("Contract testing data:\n" + str(contract_test_data) + "\n")

# Train a neural network with the data
il_nn = 2 # 2 input neurons
hl_nn = [2,3] # 2 hidden layers with 2 and 5 neurons respectively
ol_nn = 2 # 2 output neurons for binary classification

# Train a neural network with contract data
print("Training a neural network with the following:\n\
    configuration: " +str(il_nn) + " x " + str(hl_nn) + " x " + str(ol_nn) + "\n\
    total iteration: 100000\n\
    learning rate: 0.001\n")
nn = NeuralNetwork(il_nn, hl_nn, ol_nn, 0.001, 100000, 5, 10000)
nn.load_train_data(nn.binary_2_one_hot(contract_train_data))
nn.init_network()
nn.train()
trained_weights = nn.weights
trained_biases = nn.bias
packed_trained_weights = nn.pack_weights(trained_weights)

packed_trained_biases = nn.pack_biases(trained_biases)

int_packed_trained_weights = scale_packed_data(packed_trained_weights,\
    w_scale)

int_packed_trained_biases = scale_packed_data(packed_trained_biases,\
    b_scale)
print("Neural network trained!\n")

# Print submission data
print("Weights: " + str(int_packed_trained_weights) + "\n")
print("Biases: " + str(int_packed_trained_biases) + "\n")
print("Num_neurons_input_layer: " + str(il_nn) + "\n")
print("Num_neurons_output_layer: " + str(ol_nn) + "\n")
print("Num_neurons_hidden_layer: " + str(hl_nn) + "\n")

# Visualize the training data
print("Visualizing training data...\n")
scatter_x = np.array(list(map(lambda x: x[1:2][0], contract_train_data)))
scatter_y = np.array(list(map(lambda x: x[:1][0], contract_train_data)))
group = np.array(list(map(lambda x: x[2:3][0], contract_train_data)))
cdict = {0: "blue", 1: "red"}

names = []
names.append("Democrat")
names.append("Republican")

fig, ax = plt.subplots()
for g in np.unique(group):
    ix = np.where(group == g)
    ax.scatter(scatter_x[ix], scatter_y[ix], c = cdict[g], label = names[g], s = 4)
ax.legend()
plt.title("Training data points")
plt.show()
