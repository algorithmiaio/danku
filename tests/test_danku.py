from dutils.dataset import SampleCircleDataset

debug = True

def debug_print(message):
    if debug:
        print(message)
    else:
        pass

def test_python_solidity_hashing_compatability():
    # Make sure Python and solidity hashes the data groups in the same manner
    assert(False)

def test_dataset_partitioning():
    assert(False)

def test_train_single_layer_neural_network():
    assert(False)

def test_train_multi_layer_neural_network():
    assert(False)

def test_danku_init(web3, chain):
    _hashed_data_groups = []
    accuracy_criteria = 9059 # 90.59%
    submission_t = 5 # 1 minute for submission
    evaluation_t = 5 # 1 minute for evaluation
    test_reveal_t = 5 # 1 minute for revealing testing dataset

    danku, _ = chain.provider.get_or_deploy_contract('Danku')

    offer_account = web3.eth.accounts[1]
    solver_account = web3.eth.accounts[2]

    # Fund contract
    web3.eth.sendTransaction({
		'from': offer_account,
		'to': danku.address,
		'value': web3.toWei(1, "ether")
	})

    # Check that offerer was deducted
    bal = web3.eth.getBalance(offer_account)
    # Deduct reward amount (1 ETH) and gas cost (21040 wei)
    assert bal == 999998999999999999978960

    scd = SampleCircleDataset()
    scd.generate_nonce()
    scd.sha_all_data_groups()
    # Initialization step 1
    debug_print("Hashed data groups: " + str(scd.hashed_data_group))
    debug_print("Hashed Hex data groups: " +
        str(list(map(lambda x: "0x" + x.hex(), scd.hashed_data_group))))

    init1_tx = danku.transact().init1(scd.hashed_data_group, accuracy_criteria,\
        submission_t, evaluation_t, test_reveal_t)
    chain.wait.for_receipt(init1_tx)

    # Initialization step 2
    # Get data group indexes
    dgi = []
    for i in range(scd.num_data_groups):
        dgi.append(i)

    debug_print("Data group indexes: " + str(dgi))

    init2_tx = danku.transact().init2(dgi)
    chain.wait.for_receipt(init2_tx)

    # Can only access one element of a public array at a time
    training_partition = list(map(lambda x: danku.call().training_partition(x),\
        range(scd.num_train_data_groups)))
    testing_partition = list(map(lambda x: danku.call().testing_partition(x),\
        range(scd.num_test_data_groups)))
    # get partitions
    debug_print("Training partition: " + str(training_partition))
    debug_print("Testing partition: " + str(testing_partition))

    scd.partition_dataset(training_partition, testing_partition)
    # Initialization step 3
    # Time to reveal the training dataset
    training_nonces = []
    training_data = []
    for i in training_partition:
        training_nonces.append(scd.nonce[i])
    # Pack data into a 1-dimension array
    train_data = scd.pack_data(scd.train_data)
    debug_print("Train data: " + str(train_data))
    debug_print("Train nonce: " + str(scd.train_nonce))
    init3_tx = danku.transact().init3(train_data, scd.train_nonce)
    chain.wait.for_receipt(init3_tx)
    assert(False)

def test_danku_model_submission():
    assert(False)

def test_danku_reveal_test_data():
    assert(False)

def test_danku_dont_reveal_test_data():
    assert(False)

def test_danku_evaluate_model():
    assert(False)

def test_danku_cancel_contract():
    assert(False)

def test_danku_finalize_contract():
    assert(False)

def test_danku_model_accuracy():
    assert(False)
