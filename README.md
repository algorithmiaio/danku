# Danku

Machine Learning algorithms are being developed and improved at an incredible rate, but are not necessarily getting more accessible to the broader community. That’s why today Algorithmia is announcing DanKu, a new blockchain-based protocol for evaluating and purchasing ML models on a public blockchain such as Ethereum. DanKu enables anyone to get access to high quality, objectively measured machine learning models. At Algorithmia, we believe that widespread access to algorithms and deployment solutions is going to be a fundamental building block of a balanced future for AI, and DanKu is a step towards that vision.

The DanKu protocol utilizes blockchain technology via smart contracts. The contract allows anyone to post a data set, an evaluation function, and a monetary reward for anyone who can provide the best trained machine learning model for the data. Participants train deep neural networks to model the data, and submit their trained networks to the blockchain. The blockchain executes these neural network models to evaluate submissions, and ensure that payment goes to the best model.

The contract allows for the creation of a decentralized and trustless marketplace for exchanging ML models. This gives ML practitioners an opportunity to monetize their skills directly. It also allows any participant or organization to solicit machine learning models from all over the world. This will incentivize the creation of better machine learning models, and make AI more accessible to companies and software agents. Anyone with a dataset, including software agents can create DanKu contracts.

[Whitepaper](https://algorithmia.com/research/ml-models-on-blockchain)

## 0. Requirements

To run the DanKu unittests, we first need to setup our local development environment. The following commands have been tested for OSX and Linux.

### 0.1. Installing the Solidity Compiler

### For OSX

Install a these homebrew packages:

```
brew install pkg-config libffi autoconf automake libtool openssl
```

Install the solidity compiler (solc):

```
brew update
brew upgrade
brew tap ethereum/ethereum
brew install solidity
brew link solidity
```

### For Linux

```
sudo add-apt-repository ppa:ethereum/ethereum
sudo apt-get update
sudo apt-get install solc libssl-dev
```

### 0.2. Initialize your Virtual Environment

Install [virtualenv](https://virtualenv.pypa.io/en/stable/) if you don't have it yet. (Comes installed with Python3.6)

Setup a virtual environment with Python 3:

```
cd danku;
python3.6 -m venv venv;
source venv/bin/activate;

```

### 0.3. Install the Populus Framework

Install `populus` and other requirements while in virtualenv:

```
pip install -r requirements.txt
```

Yay! You should be able to develop Ethereum contracts in Python 3 now!

## 1. Populus

All contracts are developed and tested using the Populus framework.

To compile all contracts, run the following:

```
populus compile
```

To run all the tests, use the following:

```
python -m pytest --disable-pytest-warnings tests/*
```

## 2. Danku Contracts

The DanKu contract can be found in the `contracts` directory.

For more information about DanKu contracts, please read the [white paper](#).
