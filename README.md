# Danku

## 0. Requirements

To run and test Ethereum contracts, we first need to setup our local development environment. The following commands have been tested for OSX and Linux.

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

Install `populus` while in virtualenv:

```
pip install populus==2.2.0
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

TODO
