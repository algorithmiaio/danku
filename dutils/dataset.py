# Simple datasets designed for danku contracts
# Data format is (X, Y, C)
# Where X and Y are the coordinates, and C is the class
import random
from secrets import choice, randbelow
from random import shuffle
from hashlib import sha256

class Dataset(object):
    def __init__(self, max_num_data_groups=100, training_percentage=0.7,\
        partition_size=5):
        self.partition_size = partition_size
        self.max_num_data_groups = max_num_data_groups
        self.training_percentage = training_percentage
        self.training_data_group_size = int(training_percentage *\
            self.max_num_data_groups)
        self.testing_data_group_size = self.max_num_data_groups -\
            self.training_data_group_size
        self.num_data_groups = int(self.max_num_data_groups /\
            self.partition_size)
        self.num_train_data_groups = int(self.training_data_group_size/\
            self.partition_size)
        self.num_test_data_groups = int(self.testing_data_group_size/\
            self.partition_size)
        self.dps = None
        self.data = []
        self.train_data = []
        self.test_data = []
        self.train_nonce = []
        self.test_nonce = []
        self.hashed_data_group = []
        self.nonce = []
        self.training_partition = []
        self.testing_partition = []
        # Make sure total dataset size is a multiplicative of partition size
        assert(self.max_num_data_groups % self.partition_size == 0)

    def generate_nonce(self):
        l = self.num_data_groups * [None]
        self.nonce = list(map(lambda x: randbelow(2**32), l))

    def sha_data_group(self, data_group, nonce):
        # TODO: Also check if sha3_256() keccak version works
        serialized_dg = b""
        for data_point in data_group:
            for number in data_point:
                serialized_dg += number.to_bytes(32, signed=True, byteorder="big")
        serialized_dg += nonce.to_bytes(32, signed=True, byteorder="big")
        return sha256(serialized_dg).digest()

    def sha_all_data_groups(self):
        assert(len(self.data)/self.partition_size == len(self.nonce))
        for i in range(self.num_data_groups):
            start = i * self.partition_size
            end = start + self.partition_size
            dg_hash = self.sha_data_group(self.data[start:end], self.nonce[i])
            self.hashed_data_group.append(dg_hash)

    def partition_dataset(self, training_partition, testing_partition):
        # Partition the dataset
        for t_index in training_partition:
            for i in range(self.partition_size):
                self.train_data.append(self.data[self.partition_size*t_index + i])
        for t_index in testing_partition:
            for i in range(self.partition_size):
                self.test_data.append(self.data[self.partition_size*t_index + i])
        # Partition the nonces
        for t_index in training_partition:
            self.train_nonce.append(self.nonce[t_index])
        for t_index in testing_partition:
            self.test_nonce.append(self.nonce[t_index])

    def danku_init(self, training_partition=None, testing_partition=None):
        # Initialize all of the danku stuff with partition info
        if isinstance(training_partition, type(None)) or \
            isinstance(testing_partition, type(None)):
            training_partition = self.training_partition
            testing_partition = self.testing_partition

        self.generate_nonce()
        self.partition_dataset(training_partition, testing_partition)
        self.sha_all_data_groups()

    def shuffle(self):
        # Shuffle the dataset
        shuffle(self.data)

    def init_random_training_indexes(self):
        # For testing purposes
        indexes = range(self.num_data_groups)
        train_index = []
        test_index = []
        max_train_index = int(self.num_data_groups*self.training_percentage)
        while len(train_index) < max_train_index:
            random_index = choice(indexes)
            if random_index not in train_index:
                train_index.append(random_index)
        for index in indexes:
            if index not in train_index:
                test_index.append(index)
        self.training_partition = train_index
        self.testing_partition = test_index
    def pack_data(self, data):
        packed_data = []
        for item in data:
            for point in item:
                packed_data.append(point)
        return packed_data
    def unpack_data(self, data):
        unpacked_data = []
        total_iter = range(int(len(data) / self.dps))
        for i in total_iter:
            start = i * self.dps
            end = start + self.dps
            unpacked_data.append(data[start:end])
        return unpacked_data

class SampleCircleDataset(Dataset):
    '''
                               (+)
    0 0 0 0 - 1 1 1 - 0 0 0 0 | 6
    0 0 - - - 1 1 1 - - - 0 0 | 5
    0 - - - 1 1 - 1 1 - - - 0 | 4
    0 - - 1 1 - 0 - 1 1 - - 0 | 3
    - - 1 1 - 0 0 0 - 1 1 - - | 2
    - 1 1 - 0 0 0 0 0 - 1 1 1 | 1
    1 1 - 0 0 0 0 0 0 0 - 1 1 | 0
    - 1 1 - 0 0 0 0 0 - 1 1 - | 1
    - - 1 1 - 0 0 0 - 1 1 - - | 2
    0 - - 1 1 - 0 - 1 1 - - 0 | 3
    0 - - - 1 1 - 1 1 - - - 0 | 4
    0 0 - - - 1 1 1 - - - 0 0 | 5
    0 0 0 - - 1 1 1 - - 0 0 0 | 6
    _________________________  (-)
(-) 6 5 4 3 2 1 0 1 2 3 4 5 6 (+)
    '''
    def __init__(self):
        data = [(0,6,1),(-1,5,1),(0,5,1),(1,5,1),(-2,4,1),(-1,4,1),(1,4,1),
        (2,4,1),(-3,3,1),(-2,3,1),(2,3,1),(3,3,1),(-4,2,1),(-3,2,1),(3,2,1),
        (4,2,1),(-6,0,1),(-5,0,1),(5,0,1),(6,0,1),(-5,-1,1),(-4,-1,1),(4,-1,1),
        (5,-1,1),(-4,-2,1),(-3,-2,1),(3,-2,1),(4,-2,1),(-3,-3,1),(-2,-3,1),
        (2,-3,1),(3,-3,1),(-2,-4,1),(-1,-4,1),(1,-4,1),(2,-4,1),(-1,-5,1),
        (0,-5,1),(1,-5,1),(0,-6,1),(0,3,0),(-1,2,0),(0,2,0),(1,2,0),(-2,1,0),
        (-1,1,0),(0,1,0),(1,1,0),(2,1,0),(-3,0,0),(-2,0,0),(-1,0,0),(0,0,0),
        (1,0,0),(2,0,0),(3,0,0),(-2,-1,0),(-1,-1,0),(0,-1,0),(1,-1,0),(2,-1,0),
        (-1,-2,0),(0,-2,0),(1,-2,0),(0,-3,0),(-1,6,1),(1,6,1),(-1,-6,1),
        (1,-6,1),(6,1,1),(-6,6,0),(-5,6,0),(-4,6,0),(-3,6,0),(-6,5,0),(-5,5,0),
        (-6,4,0),(-6,3,0),(3,6,0),(4,6,0),(5,6,0),(6,6,0),(5,5,0),(6,5,0),
        (6,4,0),(6,3,0),(-6,-3,0),(-6,-4,0),(-6,-5,0),(-6,-6,0),(-5,-5,0),
        (-5,-6,0),(-4,-6,0),(4,-6,0),(5,-6,0),(5,-6,0),(6,-6,0),(6,-5,0),
        (6,-4,0),(6,-3,0)]
        max_num_data_groups = len(data)
        super().__init__(max_num_data_groups=max_num_data_groups)
        self.data = data
        self.shuffle()
        self.dps = 3

class SampleSwirlDataset(Dataset):
    '''
                               (+)
    - 0 0 0 0 0 0 0 0 0 - - - | 6
    - 0 - - - - - - - - - - - | 5
    - 0 - 1 1 1 1 1 1 1 1 1 - | 4
    - 0 - 1 - 0 0 0 0 0 - 1 - | 3
    - 0 - 1 - 0 1 1 - 0 - 1 - | 2
    - 0 - 1 - 0 - 1 - 0 - 1 - | 1
    - 0 - 1 - 0 - 1 - 0 - 1 - | 0
    - 0 - 1 - - - 1 - 0 - 1 - | 1
    - 0 - 1 1 1 1 1 - 0 - 1 - | 2
    - 0 - - - - - - - 0 - 1 - | 3
    - 0 0 0 0 0 0 0 0 0 - 1 - | 4
    - - - - - - - - - - - 1 - | 5
    - - - - - - 1 1 1 1 1 1 - | 6
    _________________________  (-)
(-) 6 5 4 3 2 1 0 1 2 3 4 5 6 (+)
    '''
    def __init__(self):
        data = [(-5,6,0),(-4,6,0),(-3,6,0),(-2,6,0),(-1,6,0),(0,6,0),
        (1,6,0),(2,6,0),(3,6,0),(-5,5,0),(-5,4,0),(-5,3,0),(-5,2,0),(-5,1,0),
        (-5,0,0),(-5,-1,0),(-5,-2,0),(-5,-3,0),(-5,-4,0),(-4,-4,0),(-3,-4,0),
        (-2,-4,0),(-1,-4,0),(0,-4,0),(1,-4,0),(2,-4,0),(3,-4,0),(3,-3,0),
        (3,-2,0),(3,-1,0),(3,0,0),(3,1,0),(3,2,0),(3,3,0),(2,3,0),(1,3,0),
        (0,3,0),(-1,3,0),(-1,2,0),(-1,1,0),(-1,0,0),(0,2,1),(1,2,1),(1,1,1),
        (1,0,1),(1,-1,1),(1,-2,1),(0,-2,1),(-1,-2,1),(-2,-2,1),(-3,-2,1),
        (-3,-1,1),(-3,0,1),(-3,1,1),(-3,2,1),(-3,3,1),(-3,4,1),(-2,4,1),
        (-1,4,1),(0,4,1),(1,4,1),(2,4,1),(3,4,1),(4,4,1),(5,4,1),(5,3,1),
        (5,2,1),(5,1,1),(5,0,1),(5,-1,1),(5,-2,1),(5,-3,1),(5,-4,1),(5,-5,1),
        (5,-6,1),(4,-6,1),(3,-6,1),(2,-6,1),(1,-6,1),(0,-6,1)]
        max_num_data_groups = len(data)
        super().__init__(max_num_data_groups=max_num_data_groups)
        self.data = data
        self.shuffle()
        self.dps = 3

class SampleHalfDividedDataset(Dataset):
    '''
                               (+)
    1 - - 1 - 1 - - 1 - - - - | 6
    - 1 - - - - - 1 - - - - - | 5
    1 - 1 - 1 - 1 - - - - - 0 | 4
    - 1 - - 1 - - - - - - 0 - | 3
    1 - 1 1 - - - - - 0 - - 0 | 2
    - - - - - - - - - 0 - - - | 1
    1 1 - - - - - - - - 0 - - | 0
    1 - - - - - - - 0 0 - - 0 | 1
    - - - - - - - - - - 0 - - | 2
    - - - - - - - - 0 0 - - 0 | 3
    - - - - - - 0 0 - - 0 - - | 4
    - - - - - 0 - - - 0 - 0 - | 5
    - - - - - - 0 - 0 - - - 0 | 6
    _________________________  (-)
(-) 6 5 4 3 2 1 0 1 2 3 4 5 6 (+)
    '''
    def __init__(self):
        data = [(-3,6,1),(-1,6,1),(-5,5,1),(1,5,1),(-6,4,1),(-4,4,1),
        (-2,4,1),(0,4,1),(-5,3,1),(-2,3,1),(-6,2,1),(-4,2,1),(-3,2,1),(-5,0,1),
        (6,4,0),(5,3,0),(3,2,0),(6,2,0),(3,1,0),(4,0,0),(2,-1,0),(3,-1,0),
        (6,-1,0),(4,-2,0),(2,-3,0),(3,-3,0),(6,-3,0),(0,-4,0),(1,-4,0),(4,-4,0),
        (-1,-5,0),(3,-5,0),(5,-5,0),(0,-6,0),(6,-6,0),(-6,6,1),(-6,0,1),
        (-6,-1,1),(2,6,1),(2,-6,0)]
        max_num_data_groups = len(data)
        super().__init__(max_num_data_groups=max_num_data_groups)
        self.data = data
        self.shuffle()
        self.dps = 3

class SampleAcrossCornerDataset(Dataset):
    '''
                               (+)
    0 0 0 - - - - - - - 1 1 1 | 6
    0 0 0 0 - - - - - 1 1 1 1 | 5
    0 0 0 0 - - - - - 1 1 1 1 | 4
    - 0 0 0 0 - - - 1 1 1 1 - | 3
    - - - 0 0 - - - 1 1 - - - | 2
    - - - - - 0 - 1 - - - - - | 1
    - - - - - - - - - - - - - | 0
    - - - - - 1 - 0 - - - - - | 1
    - - - 1 1 - - - 0 0 - - - | 2
    - 1 1 1 1 - - - 0 0 0 0 - | 3
    1 1 1 1 - - - - - 0 0 0 0 | 4
    1 1 1 1 - - - - - 0 0 0 0 | 5
    1 1 1 - - - - - - - 0 0 0 | 6
    _________________________  (-)
(-) 6 5 4 3 2 1 0 1 2 3 4 5 6 (+)
    '''
    def __init__(self):
        data = [(-6,6,0),(-5,6,0),(-4,6,0),(-6,5,0),(-5,5,0),(-4,5,0),
        (-3,5,0),(-6,4,0),(-5,4,0),(-4,4,0),(-3,4,0),(-5,3,0),(-4,3,0),(-3,3,0),
        (-2,2,0),(-1,1,0),(1,-1,0),(2,-2,0),(3,-3,0),(4,-3,0),(5,-3,0),(3,-4,0),
        (4,-4,0),(5,-4,0),(6,-4,0),(3,-5,0),(4,-5,0),(5,-5,0),(6,-5,0),(4,-6,0),
        (5,-6,0),(6,-6,0),(-2,3,0),(-3,2,0),(3,-2,0),(2,-3,0),(2,3,1),(3,2,1),
        (-3-2,1),(-2,-3,1)]
        max_num_data_groups = len(data)
        super().__init__(max_num_data_groups=max_num_data_groups)
        self.data = data
        self.shuffle()
        self.dps = 3
