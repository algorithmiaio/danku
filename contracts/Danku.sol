pragma solidity ^0.4.19;
// Danku contract version 0.0.1
// Data points are x, y, and z

contract Danku {
  function Danku() public {
    // Neural Network Structure:
    //
    // (assertd) input layer x number of neurons
    // (optional) hidden layers x number of neurons
    // (assertd) output layer x number of neurons
  }
  struct Submission {
      address payment_address;
      // Define the number of neurons each layer has.
      uint num_neurons_input_layer;
      uint num_neurons_output_layer;
      // There can be multiple hidden layers.
      uint[] num_neurons_hidden_layer;
      // Weights indexes are the following:
      // weights[l_i x l_n_i x pl_n_i]
      // Also number of layers in weights is layers.length-1
      int256[] weights;
  }
  struct NeuralLayer {
    int256[] neurons;
    int256[] errors;
    string layer_type;
  }

  address public organizer;
  // Keep track of the best model
  uint best_submission_index;
  // Keep track of best model accuracy
  int256 best_submission_accuracy = 0;
  // The model accuracy criteria
  int256 model_accuracy_criteria;
  // Use test data if provided
  bool use_test_data = false;
  // Each partition is 5% of the total dataset size
  uint constant partition_size = 5;
  // Data points are made up of x and y coordinates and the prediction
  uint constant datapoint_size = 3;
  uint constant prediction_size = 1;
  // Max number of data groups
  // Change this to your data group size
  uint8 constant max_num_data_groups = 100;
  // Training partition size
  uint8 constant training_data_group_size = 70;
  // Testing partition size
  uint8 constant testing_data_group_size = max_num_data_groups - training_data_group_size;
  // Dataset is divided into data groups.
  // Every data group includes a nonce.
  // Look at sha_data_group() for more detail about hashing a data group
  bytes32[max_num_data_groups/partition_size] hashed_data_groups;
  // Nonces are revelead together with data groups
  uint[max_num_data_groups/partition_size] data_group_nonces;
  // + 1 for prediction
  // A data group has 3 data points in total
  int256[datapoint_size][] train_data;
  int256[datapoint_size][] test_data;
  bytes32 partition_seed;
  // Deadline for submitting solutions in terms of block size
  uint public submission_stage_block_size;
  // Deadline for revealing the testing dataset
  uint public reveal_test_data_groups_block_size;
  // Deadline for evaluating the submissions
  uint public evaluation_stage_block_size;
  uint init1_block_height;
  uint init3_block_height;
  uint init_level = 0;
  // Training partition size is 14 (70%)
  // Testing partition size is 6 (30%)
  uint[training_data_group_size/partition_size] public training_partition;
  uint[testing_data_group_size/partition_size] public testing_partition;
  uint256 train_dg_revealed = 0;
  uint256 test_dg_revealed = 0;
  Submission[] submission_queue;
  bool public contract_terminated = false;

  // Takes in array of hashed data points of the entire dataset,
  // submission and evaluation times
  function init1(bytes32[max_num_data_groups/partition_size] _hashed_data_groups, int accuracy_criteria, uint submission_t, uint evaluation_t, uint test_reveal_t) external {
    // Make sure contract is not terminated
    assert(contract_terminated == false);
    // Make sure it's called in order
    assert(init_level == 0);
    organizer = msg.sender;
    init_level = 1;
    init1_block_height = block.number;

    // Make sure there are in total 20 hashed data groups
    assert(_hashed_data_groups.length == max_num_data_groups/partition_size);
    hashed_data_groups = _hashed_data_groups;
    // Make sure submission, evaluation and test reaveal times are set to
    // at least 1 block
    assert(submission_t > 0);
    assert(evaluation_t > 0);
    assert(test_reveal_t > 0);
    // Accuracy criteria example: 85.9% => 8,590
    // 100 % => 10,000
    assert(accuracy_criteria > 0);
    submission_stage_block_size = submission_t;
    evaluation_stage_block_size = evaluation_t;
    reveal_test_data_groups_block_size = test_reveal_t;
    model_accuracy_criteria = accuracy_criteria;
  }

  function init2(uint[] index_array) external {
    // Make sure contract is not terminated
    assert(contract_terminated == false);
    // Only allow calling it once, in order
    assert(init_level == 1);
    // Make sure it's being called within 5 blocks on init1()
    // to minimize organizer influence on random index selection
    if (block.number <= init1_block_height+5 && block.number > init1_block_height) {
      // TODO: Also make sure it's being called 1 block after init1()
      // Randomly select indexes
      randomly_select_index(index_array);
      init_level = 2;
    } else {
      // Cancel the contract if init2() hasn't been called within 5
      // blocks of init1()
      cancel_contract();
    }
  }

  function init3(int256[] _train_data_groups, int256 _train_data_group_nonces) external {
    // Pass a single data group at a time
    // Make sure contract is not terminated
    assert(contract_terminated == false);
    // Only allow calling once, in order
    assert(init_level == 2);
    // Verify data group and nonce lengths
    assert((_train_data_groups.length/partition_size)/datapoint_size == 1);
    // Verify data group hashes
    // Order of revealed training data group must be the same with training partitions
    // Otherwise hash verification will fail
    assert(sha_data_group(_train_data_groups, _train_data_group_nonces) ==
      hashed_data_groups[training_partition[train_dg_revealed]]);
    train_dg_revealed += 1;
    // Assign training data after verifying the corresponding hash
    unpack_data_groups(_train_data_groups, true);
    if (train_dg_revealed == (training_data_group_size/partition_size)) {
      init_level = 3;
      init3_block_height = block.number;
    }
  }

  function submit_model(
    // Public function for users to submit a solution
    // Returns the submission index
    address paymentAddress,
    uint num_neurons_input_layer,
    uint num_neurons_output_layer,
    uint[] num_neurons_hidden_layer,
    int[] weights) public returns (uint) {
      // Make sure contract is not terminated
      assert(contract_terminated == false);
      // Make sure it's not the initialization stage anymore
      assert(init_level == 3);
      // Make sure it's still within the submission stage
      assert(block.number < init3_block_height + submission_stage_block_size);
      // Make sure that num of neurons in the input & output layer matches
      // the problem description
      assert(num_neurons_input_layer == datapoint_size - prediction_size);
      assert(num_neurons_output_layer == prediction_size);
      // Make sure that the number of weights match network structure
      assert(valid_weights(weights, num_neurons_input_layer, num_neurons_output_layer, num_neurons_hidden_layer));
      // Add solution to submission queue
      submission_queue.push(Submission(
        paymentAddress,
        num_neurons_input_layer,
        num_neurons_output_layer,
        num_neurons_hidden_layer,
        weights));

    return submission_queue.length-1;
  }

    function reveal_test_data(int256[] _test_data_groups, int256[] _test_data_group_nonces) external {
    // Make sure contract is not terminated
    assert(contract_terminated == false);
    // Make sure it's not the initialization stage anymore
    assert(init_level == 3);
    // Make sure it's revealed after the submission stage
    assert(block.number >= init3_block_height + submission_stage_block_size);
    // Make sure it's revealed within the reveal stage
    assert(block.number < init3_block_height + submission_stage_block_size + reveal_test_data_groups_block_size);
    // Verify data group and nonce lengths
    assert(_test_data_groups.length == max_num_data_groups / partition_size - training_partition.length);
    assert(_test_data_group_nonces.length == max_num_data_groups / partition_size - training_partition.length);
    // Verify data group hashes
    for (uint i = 0; i < _test_data_groups.length; i++) {
      assert(sha_data_group(_test_data_groups, _test_data_group_nonces[i]) == hashed_data_groups[testing_partition[i]]);
    }
    // Assign testing data
    unpack_data_groups(_test_data_groups, false);
    // Use test data for evaluation
    use_test_data = true;
  }

  function evaluate_model(uint submission_index) public {
    // Make sure contract is not terminated
    assert(contract_terminated == false);
    // Make sure it's not the initialization stage anymore
    assert(init_level == 3);
    // Make sure it's evaluated after the reveal stage
    assert(block.number >= init3_block_height + submission_stage_block_size + reveal_test_data_groups_block_size);
    // Make sure it's evaluated within the evaluation stage
    assert(block.number < init3_block_height + submission_stage_block_size + reveal_test_data_groups_block_size + evaluation_stage_block_size);
    // Evaluates a submitted model & keeps track of the best model
    int256 submission_accuracy = 0;
    if (use_test_data == true) {
      submission_accuracy = model_accuracy(submission_index, test_data);
    } else {
      submission_accuracy = model_accuracy(submission_index, train_data);
    }

    // Keep track of the most accurate model
    if (submission_accuracy > best_submission_accuracy) {
      best_submission_index = submission_index;
      best_submission_accuracy = submission_accuracy;
    }
  }

  function cancel_contract() public {
    // Make sure contract is not already terminated
    assert(contract_terminated == false);
    // Contract can only be cancelled if initialization has failed.
    assert(init_level < 3);
    // Refund remaining balance to organizer
    organizer.transfer(this.balance);
    // Terminate contract
    contract_terminated = true;
  }

  function finalize_contract() public {
    // Make sure contract is not terminated
    assert(contract_terminated == false);
    // Make sure it's not the initialization stage anymore
    assert(init_level == 3);
    // Make sure the contract is finalized after the evaluation stage
    assert(block.number >= init3_block_height + submission_stage_block_size + reveal_test_data_groups_block_size + evaluation_stage_block_size);
    // Get the best submission to compare it against the criteria
    Submission memory best_submission = submission_queue[best_submission_index];
    // If best submission passes criteria, payout to the submitter
    if (best_submission_accuracy >= model_accuracy_criteria) {
      best_submission.payment_address.transfer(this.balance);
    // If the best submission fails the criteria, refund the balance back to the organizer
    } else {
      organizer.transfer(this.balance);
    }
    contract_terminated = true;
  }

  function model_accuracy(uint submission_index, int256[datapoint_size][] data) public constant returns (int256){
    // Make sure contract is not terminated
    assert(contract_terminated == false);
    // Make sure it's not the initialization stage anymore
    assert(init_level == 3);
    // Leave function public for offline error calculation
    // Get's the sum error for the model
    Submission memory sub = submission_queue[submission_index];
    int256 true_prediction = 0;
    int256 false_prediction = 0;
    int256 accuracy = 0;
    for (uint i = 0; i < data.length; i++) {
      int[] memory prediction;
      int[] memory ground_truth;
      // Get ground truth
      for (uint j = 0; j < data[i].length; j++) {
        // Only get prediction values
        if (j > datapoint_size - prediction_size - 1) {
          ground_truth[ground_truth.length] = data[i][j];
        }
      }
      // Get prediction
      prediction = forward_pass(data[i], sub);
      // Get error for the output layer
      for (uint k = 0; k < ground_truth.length; k++) {
        if (ground_truth[k] - prediction[k] == 0) {
          true_prediction += 1;
        } else {
          false_prediction += 1;
        }
      }
      // We multipl by 10000 to get up to 2 decimal point precision while
      // calculating the accuracy
      accuracy = (true_prediction * 10000) / (true_prediction + false_prediction);
    }
    return accuracy;
  }

  function round_up_division(int256 dividend, int256 divisor) private pure returns(int256) {
    // A special trick since solidity normall rounds it down
    return (dividend + divisor -1) / divisor;
  }

  function not_in_train_partition(uint[training_data_group_size/partition_size] partition, uint number) private pure returns (bool) {
    for (uint i = 0; i < partition.length; i++) {
      if (number == partition[i]) {
        return false;
      }
    }
    return true;
  }

  function randomly_select_index(uint[] array) private {
    uint t_index = 0;
    uint array_length = array.length;
    uint block_i = 0;
    // Randomly select training indexes
    while(t_index < training_partition.length) {
      uint random_index = uint(sha256(block.blockhash(block.number-block_i))) % array_length;
      training_partition[t_index] = array[random_index];
      array[random_index] = array[array_length-1];
      array_length--;
      block_i++;
      t_index++;
    }
    t_index = 0;
    while(t_index < testing_partition.length) {
      testing_partition[t_index] = array[array_length-1];
      array_length--;
      t_index++;
    }
  }

  function valid_weights(int[] weights, uint num_neurons_input_layer, uint num_neurons_output_layer, uint[] num_neurons_hidden_layer) private pure returns (bool) {
    // make sure the number of weights match the network structure
    // get number of weights based on network structure
    uint ns_total = 0;
    uint wa_total = 0;
    uint number_of_layers = 2 + num_neurons_hidden_layer.length;

    if (number_of_layers == 2) {
      ns_total = num_neurons_input_layer * num_neurons_output_layer;
    } else {
      for(uint i = 0; i < num_neurons_hidden_layer.length; i++) {
        // Get weights between first hidden layer and input layer
        if (i==0){
          ns_total += num_neurons_input_layer * num_neurons_hidden_layer[i];
        // Get weights between hidden layers
        } else {
          ns_total += num_neurons_hidden_layer[i-1] * num_neurons_hidden_layer[i];
        }
      }
      // Get weights between last hidden layer and output layer
      ns_total += num_neurons_hidden_layer[num_neurons_hidden_layer.length-1] * num_neurons_output_layer;
    }
    // get number of weights in the weights array
    wa_total = weights.length;

    return ns_total == wa_total;
  }

    function unpack_data_groups(int256[] _data_groups, bool is_train_data) private {
    int256[datapoint_size][] memory merged_data_group = new int256[datapoint_size][](_data_groups.length/datapoint_size);

    for (uint i = 0; i < _data_groups.length/datapoint_size; i++) {
      for (uint j = 0; j < datapoint_size; j++) {
        merged_data_group[i][j] = _data_groups[i*datapoint_size + j];
      }
    }
    if (is_train_data == true) {
      // Assign training data
      for (uint k = 0; k < merged_data_group.length; k++) {
        train_data.push(merged_data_group[k]);
      }
    } else {
      for (uint l = 0; l < merged_data_group.length; l++) {
        test_data.push(merged_data_group[l]);
      }
      // Assign testing data
      test_data = merged_data_group;
    }
  }

    function sha_data_group(int256[] data_group, int256 data_group_nonce) private pure returns (bytes32) {
      // Extract the relevant data points for the given data group index
      // We concat all data groups and add the nounce to the end of the array
      // and get the sha256 for the array
      uint index_tracker = 0;
      uint256 total_size = datapoint_size * partition_size;
      /* uint256 start_index = data_group_index * total_size;
      uint256 iter_limit = start_index + total_size; */
      int256[] memory all_data_points = new int256[](total_size+1);

      for (uint256 i = 0; i < total_size; i++) {
        all_data_points[index_tracker] = data_group[i];
        index_tracker += 1;
      }
      // Add nonce to the whole array
      all_data_points[index_tracker] = data_group_nonce;
      // Return sha256 on all data points + nonce
      return sha256(all_data_points);
    }

  function relu_activation(int256 x) private pure returns (int256) {
    if (x < 0) {
      return 0;
    } else {
      return x;
    }
  }

  function get_layers(Submission sub) private pure returns (NeuralLayer[]) {
    uint256[] memory nn_hl = sub.num_neurons_hidden_layer;
    NeuralLayer[] memory layers;
    uint256 layer_index = 0;

    NeuralLayer memory input_layer;
    input_layer.layer_type = "input_layer";
    layers[layer_index] = input_layer;
    layer_index += 1;
    for (uint i = 0; i < nn_hl.length; i++) {
      NeuralLayer memory hidden_layer;
      hidden_layer.layer_type = "hidden_layer";
      layers[layer_index] = hidden_layer;
      layer_index += 1;
    }
    NeuralLayer memory output_layer;
    input_layer.layer_type = "output_layer";
    layers[layer_index] = output_layer;
    layer_index += 1;
    return layers;
  }

  function forward_pass(int[datapoint_size] data_point, Submission sub) private pure returns (int256[]) {
    NeuralLayer[] memory layers = get_layers(sub);
    // load inputs from input layer
    for (uint input_i = 0; input_i < sub.num_neurons_input_layer; input_i++) {
      layers[0].neurons[input_i] = data_point[input_i];
    }
    // evaluate the neurons in following layers
    // skip input layer by starting at 1
    uint weight_index;
    for (uint layer_i = 1; layer_i < layers.length-1; layer_i++) {
      NeuralLayer memory current_layer = layers[layer_i];
      NeuralLayer memory previous_layer = layers[layer_i-1];
      for (uint layer_neuron_i = 0; layer_neuron_i < current_layer.neurons.length; layer_neuron_i++) {
        int total = 0;
        for (uint prev_layer_neuron_i = 0; prev_layer_neuron_i < previous_layer.neurons.length; prev_layer_neuron_i++) {
          weight_index = layer_i * (layers.length-1) + layer_neuron_i * current_layer.neurons.length + prev_layer_neuron_i * previous_layer.neurons.length;
          total += previous_layer.neurons[prev_layer_neuron_i] * sub.weights[weight_index];
        }
        layers[layer_i].neurons[layer_neuron_i] = relu_activation(total);
      }
    }
    // Return the output layer neurons
    return layers[layers.length-1].neurons;
  }

  // Fallback function for sending ether to this contract
  function () public payable {}
}
