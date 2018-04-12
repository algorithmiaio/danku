import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from dutils.dataset import Dataset
from dutils.neural_network import NeuralNetwork
from web3 import Web3, HTTPProvider, IPCProvider
from matplotlib import pyplot as plt
import numpy as np
import json
import geopandas

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

# Check if data dir exists, if not, createit
if not os.path.exists("data"):
    os.makedirs("data")

def get_datapoint_pred(model_index, x, y):
    # 0 for blue, and 1 for red
    # Returns [x,y,class]
    return(x,y,int(danku.call().model_accuracy(model_index, [[x, y, 0]]) == 0))

def convert_coord(coord):
    coord = str(coord)
    if(len(coord.split(".")[1]) < 6):
        rVal = coord+"0"*(6-len(coord.split(".")[1]))
    elif(len(coord.split(".")[1]) > 6):
        rVal = coord.split(".")[0] + coord.split(".")[1][:6]
    else:
        rVal = coord
    return int(rVal.replace(".",""))

def revert_coord(coord):
    return float(coord) / 1000000

def get_prediction_map(submission_id=None):
    # Generate prediction map for best model
    pred_obj = {}
    # If there's no submission ID, get the best submission instead
    if isinstance(submission_id, type(None)):
        pred_obj["submission_id"] = danku.call().best_submission_index() # Get best submission ID
    else:
        pred_obj["submission_id"] = submission_id
    pred_obj["x_lower"] = 22500000
    pred_obj["x_upper"] = 52000000
    pred_obj["y_lower"] = -126000000
    pred_obj["y_upper"] = -64000000
    pred_obj["iter_step"] = 100000
    pred_obj["map_data"] = []
    total_iter = int((abs(pred_obj["x_upper"]-pred_obj["x_lower"])/pred_obj["iter_step"]) *\
        (abs(pred_obj["y_upper"]-pred_obj["y_lower"])/pred_obj["iter_step"]))
    print_perc = 0.05 # Show 1% progress at a time
    count = 0

    cache_file_name = "map_predictions.json"
    if not os.path.exists("data/" + cache_file_name):
        with open("data/" + cache_file_name, "w") as fh:
            json.dump({"predictions":[]}, fh)

    # Before generating a prediction map, check if a cache exists
    with open("data/" + cache_file_name, "r") as fh:
        all_predictions = json.load(fh)

    attributes = ["submission_id", "x_lower", "x_upper", "y_lower", "y_upper", "iter_step"]
    found = False
    for pred in all_predictions["predictions"]:
        for attr in attributes:
            if pred[attr] != pred_obj[attr]:
                found = False
                break
            else:
                found = True
        if found == True:
            print("Using cached prediction map for submission:" + str(pred_obj["submission_id"]))
            with open("data/" + cache_file_name) as fh:
                pred_obj = pred
            break

    if len(pred_obj["map_data"]) == 0:
        print("Generating prediction map...")
        for x in reversed(range(pred_obj["x_lower"], pred_obj["x_upper"], pred_obj["iter_step"])):
            for y in range(pred_obj["y_lower"], pred_obj["y_upper"], pred_obj["iter_step"]):
                pred_obj["map_data"].append(get_datapoint_pred(pred_obj["submission_id"], x, y))
                count += 1
                if (count) % int(total_iter * print_perc) == 0:
                    print("Progress: " + str(int(count*100) // total_iter + ((count*100) % total_iter > 0)) + "/100%")

        print("Generated prediction map...")
        print("Caching prediction map...")
        all_predictions["predictions"].append(pred_obj)
        with open("data/" + cache_file_name, "w") as fh:
            json.dump(all_predictions, fh)
        print("Cached!")

    # Visualize the training data
    print("Visualizing model on U.S map...\n")
    scatter_x = np.array(list(map(lambda x: x[1:2][0], pred_obj["map_data"])))
    scatter_y = np.array(list(map(lambda x: x[:1][0], pred_obj["map_data"])))
    group = np.array(list(map(lambda x: x[2:3][0], pred_obj["map_data"])))
    cdict = {0: "blue", 1: "red"}

    names = []
    names.append("Democrat")
    names.append("Republican")

    print("Visualizing prediction data...")

    valid_states = ["AL", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA", "ID", "IL",\
        "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE",\
        "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD",\
        "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]

    fig, ax = plt.subplots(figsize=(9,6))

    for g in np.unique(group):
        ix = np.where(group == g)
        ax.scatter(scatter_x[ix], scatter_y[ix], c = cdict[g], label = names[g], s = 1)

    ax.legend()
    plt.title("Prediction map for submission: " + str(pred_obj["submission_id"]))
    print("Processing state coords...")
    # State shape data from: https://www.census.gov/geo/maps-data/data/cbf/cbf_state.html
    df = geopandas.read_file("data/cb_2017_us_state_20m.shp")
    df = df.loc[df["STUSPS"].isin(valid_states)]
    df.geometry = df.geometry.scale(xfact=1000000, yfact=1000000, zfact=1.0, origin=(0, 0))
    print("Visualizing state coords...")
    df.plot(ax=ax, color="white", edgecolor="black", alpha=0.3, linewidth=0.5)
    plt.show()

# Get the winning submission prediction map
get_prediction_map()
