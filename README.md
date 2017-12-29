# Danku

## 0. Requirements

Since we're dealing with Ethereum contracts, you'll need a few things installed locally. We've mainly developed in a mac environment, so some of these things may not work as expected on linux. (for now)

### 0.1. Installing the solidity compiler

### For OSX

Install a few homebrew packages:

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

### 0.2. Initializing your Virtual Environment

Install [virtualenv](https://virtualenv.pypa.io/en/stable/) if you don't have it yet.

Setup a virtual environment with Python 3:

```
cd danku;
virtualenv --python=python3 venv;
source venv/bin/activate;
```

### 0.3. Install the Populus Framework

Install `populus` while in virtualenv:

```
pip install populus==2.2.0
```

Yay! You should be able to develop Ethereum contracts in Python 3 now!

## 1. Populus

All contracts are testing using the Populus framework.

To compile all contracts, run the following:

```
populus compile
```

To run all the tests, use the following:

```
python -m pytest --disable-pytest-warnings tests/*
```
