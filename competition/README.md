# First DanKu Comptetition

## Some Info

This is the first ever publicly launched DanKu contract/competition.

The contract address is: [`0x9A0991fc223dFFE420e08f15b88a593a3b8D44B8`](https://etherscan.io/address/0x9A0991fc223dFFE420e08f15b88a593a3b8D44B8)

The contract was initialized on `block 5121944` (on Feb 19th, 2018), with 400 training data samples.

The contract has a submission period of `241920 blocks`, which corresponds to 6 weeks.

The contract has a test data reveal period of `17280 blocks`, which corresponds to 3 days.

The contract has a evaluation period of `40320 blocks`, which corresponds to 7 days.

The total size of the dataset is made up of 500 samples. For this contract, an 80% 20% split was made for training and testing data.

You can find the DanKu contract [here](#).

## How to participate

You first need a few things installed & running. Obviously you need some experience training ML models, but for the purpose of this competition, some sample code is provided for training a simple neural network. You can modify, or create your own model from scratch.

Since we're going to use the `populus` framework instead of `truffle`, this will make it easier to use most ML frameworks. This is because `populus` is Python based.

```
python -m venv venv;
source venv/bin/activate;
pip install populus==2.2.0
```

You also need `geth` up and running. It helps you interact with the contract, like downloading the test dataset and submitting your solution. Please refer to the official Ethereum guide for installing `geth` [here](https://github.com/ethereum/go-ethereum/wiki/Installing-Geth).

After you get `geth` running locally on the default port number `8545` you can connect to it and download the training dataset with the following piece of code:

```python
download_training_dataset_from_contract()
```

You can use the following command to visualize the dataset that you've just downloaded.

```python
visualize_dataset()
```

You can run the following sample code to train a neural network on the test data:

```python
train_nn()
```

Before submitted your solution/model, you need to scale the weights and biases by 1000, and convert them into integers:

```python
pack_data()
```

Now we can submit our model to the DanKu contract. If you already have your account setup with `populus`, you can do the following to submit your solution:

```python
pack_data()
```

Another alternative way to submit your solution is to use the mist browser. First install the Mist browser by following the instructions [here](https://github.com/ethereum/mist).

## End to end example

If you want to see how a DanKu contract functions end to end, you can run one of the tests in debug mode. This will spit out all of the important info in the console to help better understand how DanKu contracts function.

```python
example_unit_test_with_debug_true()
```

## Questions and more

If you have questions, or find a bugs anywhere, please create an issue [here](https://github.com/algorithmiaio/danku/issues). After installing the mist browser, follow the intructions below:

insert_screen_shots_or_gif_of_mist_browser_steps_here
