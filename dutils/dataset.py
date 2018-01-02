# Simple datasets designed for danku contracts
# Data format is (X, Y, C)
# Where X and Y are the coordinates, and C is the class
import random
from hashlib import sha256

class Dataset(object):
    def __init__(self, max_num_data_groups=100, traiaing_percentage=0.8,\
        partition_size=5):
        self.partition_size = partition_size
        self.max_num_data_groups = max_num_data_groups
        self.training_data_group_size = int(traiaing_percentage *\
            self.max_num_data_groups)
        self.num_data_groups = self.max_num_data_groups /\
            self.partition_size
        self.testing_data_group_size = self.max_num_data_groups -\
            self.training_data_group_size
        self.dimensions = (13, 13)
        self.data = []
        self.train_data = []
        self.test_data = []
        self.sha_data_group = []
        self.nonce = []
        # Make sure total dataset size is a multiplicative of partition size
        assert(self.max_num_data_groups % self.partition_size == 0)

    def generate_nonce(self):
        l = self.num_data_groups * [None]
        self.nonce = list(map(lambda x: random.randint(0, 2**32), l))

    def sha_data_group(self, data_group, nonce):
        # TODO: Also check if sha3_256() keccak version works
        serialized_dg = b""
        for data_point in data_group:
            serialized_dg += data_point.to_bytes(32, byteorder="big")
        serialized_dg += nonce.to_bytes(32, byteorder="big")
        return sha256(serialized_dg).digest()

    def sha_all_data_groups(self, data_groups, nonces):
        assert(len(data_groups) == len(nonces))
        for i in range(len(nonces)):
            self.sha_data_group.append(data_groups[i], nonces[i])

    def partition_dataset(self, training_partition, testing_partition):
        for t_index in training_partition:
            self.train_data.append(self.data[t_index])
        for t_index in testing_partition:
            self.test_data.append(self.data[t_index])

    def danku_init(self, training_partition, testing_partition):
        # Initialize all of the danku stuff with partition info
        self.partition_dataset(training_partition, testing_partition)
        self.generate_nonce()
        self.sha_all_data_groups()

class SampleCircleDataset(Dataset):
    '''
                               (+)
    - - - - - 1 1 1 - - - - - | 6
    - - - - - 1 1 1 - - - - - | 5
    - - - - 1 1 - 1 1 - - - - | 4
    - - - 1 1 - 0 - 1 1 - - - | 3
    - - 1 1 - 0 0 0 - 1 1 - - | 2
    - 1 1 - 0 0 0 0 0 - 1 1 1 | 1
    1 1 - 0 0 0 0 0 0 0 - 1 1 | 0
    - 1 1 - 0 0 0 0 0 - 1 1 - | 1
    - - 1 1 - 0 0 0 - 1 1 - - | 2
    - - - 1 1 - 0 - 1 1 - - - | 3
    - - - - 1 1 - 1 1 - - - - | 4
    - - - - - 1 1 1 - - - - - | 5
    - - - - - 1 1 1 - - - - - | 6
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
        (1,-6,1),(6,1,1)]
        max_num_data_groups = len(data)
        super().__init__(max_num_data_groups=max_num_data_groups)
        self.data = data

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
