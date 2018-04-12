# Simple datasets designed for danku contracts
# Data format is (X, Y, C)
# Where X and Y are the coordinates, and C is the class
import random
from secrets import choice, randbelow
from random import shuffle
from hashlib import sha256
import dutils.debug as dbg
import pandas as pd
import numpy as np

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

    def sha_data_group(self, data_group, nonce, i):
        # TODO: Also check if sha3_256() keccak version works
        serialized_dg = b""
        l = []
        for data_point in data_group:
            for number in data_point:
                l.append(number)
                serialized_dg += number.to_bytes(32, signed=True, byteorder="big")
        serialized_dg += nonce.to_bytes(32, signed=True, byteorder="big")
        l.append(nonce)
        dbg.dprint("(" + str(i) + ") Hashed data group: " + str(l))
        return sha256(serialized_dg).digest()

    def sha_all_data_groups(self):
        assert(len(self.data)/self.partition_size == len(self.nonce))
        for i in range(self.num_data_groups):
            start = i * self.partition_size
            end = start + self.partition_size
            dg_hash = self.sha_data_group(self.data[start:end], self.nonce[i], i)
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
            unpacked_data.append(tuple(data[start:end]))
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
    def __init__(self, training_percentage=0.7,\
        partition_size=5):
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
        super().__init__(max_num_data_groups=max_num_data_groups, training_percentage=training_percentage,\
        partition_size=partition_size)
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
    def __init__(self, training_percentage=0.7,\
        partition_size=5):
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
        super().__init__(max_num_data_groups=max_num_data_groups, training_percentage=training_percentage,\
        partition_size=partition_size)
        self.data = data
        self.shuffle()
        self.dps = 3

class SampleHalfDividedDataset(Dataset):
    '''
                               (+)
    1 1 1 1 1 1 1 - 1 - - - - | 6
    - 1 - 1 - - - 1 - - - - - | 5
    1 - 1 - 1 - 1 - - - - - 0 | 4
    - 1 - - 1 - - - - - - 0 - | 3
    1 - 1 1 - - - - - 0 - - 0 | 2
    - 1 - - - - - - - 0 - - - | 1
    1 1 - 1 - - - - - - 0 - - | 0
    1 - - - - - - - 0 0 - - 0 | 1
    1 1 - - - - - - - - 0 - - | 2
    1 - - - - - - - 0 0 - - 0 | 3
    - - - - - - 0 0 - - 0 - - | 4
    - - - - - 0 - - - 0 - 0 - | 5
    - - - - - - 0 - 0 - - - 0 | 6
    _________________________  (-)
(-) 6 5 4 3 2 1 0 1 2 3 4 5 6 (+)
    '''
    def __init__(self, training_percentage=0.7,\
        partition_size=5):
        data = [(-3,6,1),(-1,6,1),(-5,5,1),(1,5,1),(-6,4,1),(-4,4,1),
        (-2,4,1),(0,4,1),(-5,3,1),(-2,3,1),(-6,2,1),(-4,2,1),(-3,2,1),(-5,0,1),
        (6,4,0),(5,3,0),(3,2,0),(6,2,0),(3,1,0),(4,0,0),(2,-1,0),(3,-1,0),
        (6,-1,0),(4,-2,0),(2,-3,0),(3,-3,0),(6,-3,0),(0,-4,0),(1,-4,0),(4,-4,0),
        (-1,-5,0),(3,-5,0),(5,-5,0),(0,-6,0),(6,-6,0),(-6,6,1),(-6,0,1),
        (-6,-1,1),(2,6,1),(2,-6,0),(-5,6,1),(-4,6,1),(-2,6,1),(0,6,1),(-5,1,1),
        (-1,-2,1),(-3,5,1),(-3,0,1),(-5,-2,1),(-6,-3,1)]
        max_num_data_groups = len(data)
        super().__init__(max_num_data_groups=max_num_data_groups, training_percentage=training_percentage,\
        partition_size=partition_size)
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
    - - - - - - - - - - - - - | 1
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
    def __init__(self, training_percentage=0.7,\
        partition_size=5):
        data = [(-6,6,0),(-5,6,0),(-4,6,0),(-6,5,0),(-5,5,0),(-4,5,0),
        (-3,5,0),(-6,4,0),(-5,4,0),(-4,4,0),(-3,4,0),(-5,3,0),(-4,3,0),(-3,3,0),
        (-2,2,0),(1,-1,0),(2,-2,0),(3,-3,0),(4,-3,0),(5,-3,0),(3,-4,0),(4,-4,0),
        (5,-4,0),(6,-4,0),(3,-5,0),(4,-5,0),(5,-5,0),(6,-5,0),(4,-6,0),(5,-6,0),
        (6,-6,0),(-2,3,0),(-3,2,0),(3,-2,0),(2,-3,0),(-6,-6,1),(-5,-6,1),
        (-4,-6,1),(-6,-5,1),(-5,-5,1),(-4,-5,1),(-3,-5,1),(-6,-4,1),(-5,-4,1),
        (-4,-4,1),(-3,-4,1),(-5,-3,1),(-4,-3,1),(-3,-3,1),(-2,-3,1),(-3,-2,1),
        (-2,-2,1),(-1,-1,1),(2,2,1),(3,2,1),(2,3,1),(3,3,1),(4,3,1),(5,3,1),
        (3,4,1),(4,4,1),(5,4,1),(6,4,1),(3,5,1),(4,5,1),(5,5,1),(6,5,1),(4,6,1),
        (5,6,1),(6,5,1)]
        max_num_data_groups = len(data)
        super().__init__(max_num_data_groups=max_num_data_groups, training_percentage=training_percentage,\
        partition_size=partition_size)
        self.data = data
        self.shuffle()
        self.dps = 3

class DemoDataset(Dataset):
    def __init__(self, training_percentage=0.7,\
        partition_size=5):
        # Voting data
        # https://data.world/data4democracy/election-transparency/workspace/file?filename=election_results_with_demographics.csv

        # Zipcode data
        # https://www.gaslampmedia.com/download-zip-code-latitude-longitude-city-state-county-csv/

        # Loading the files

        cols_to_use2 = ["fips_fixed", "clinton", "trump", "state", "jurisdiction"]
        df2 = pd.read_csv("data/election_results_with_demographics.csv", usecols=cols_to_use2)

        cols_to_use4 = ["zip_code", "latitude", "longitude", "city", "state", "county"]
        df4 = pd.read_csv("data/zip_codes_states.csv", usecols=cols_to_use4, dtype=str)

        # Rename column
        df2 = df2.rename(columns={"fips_fixed": "fips"})

        # Remove duplicate state-county pairs
        c_maxes = df4.groupby(["state", "county"]).zip_code.transform(max)
        df4 = df4.loc[df4.zip_code == c_maxes]
        df4 = df4.reset_index(drop=True)

        # Lower case state and county names
        df2["state"] = df2["state"].str.lower()
        df2["jurisdiction"] = df2["jurisdiction"].str.lower()

        df4["state"] = df4["state"].str.lower()
        df4["county"] = df4["county"].str.lower()

        # Merge table
        dfN = pd.merge(df2, df4,  how="left", left_on=["state","jurisdiction"], right_on = ["state","county"])

        # Drop null value rows
        dfN = dfN[pd.notnull(dfN["latitude"])]
        dfN = dfN[pd.notnull(dfN["longitude"])]
        dfN = dfN.reset_index(drop=True)


        # Remove ° from latitude & longitude
        dfN["latitude"].replace(regex=True,inplace=True,to_replace="°",value="")
        dfN["longitude"].replace(regex=True,inplace=True,to_replace="°",value="")

        # Class 0 for Democrat and class 1 for Republican
        def pick_part(row):
            if row["clinton"] > row["trump"]:
                return 0
            else:
                return 1

        # Assign party for majority vote
        dfN["party"] = dfN.apply(lambda row: pick_part(row), axis=1)

        # Remove the voting percentages & fips code
        del dfN["clinton"]
        del dfN["trump"]
        del dfN["fips"]
        del dfN["state"]
        del dfN["jurisdiction"]
        del dfN["zip_code"]
        del dfN["city"]
        del dfN["county"]

        # Properly convert and format latitude & longitude
        dfN["latitude"].replace(regex=True,inplace=True,to_replace=r"\+",value="")
        dfN["longitude"].replace(regex=True,inplace=True,to_replace=r"\+",value="")

        dfN["latitude"].replace(regex=True,inplace=True,to_replace=r"\–",value="-")
        dfN["longitude"].replace(regex=True,inplace=True,to_replace=r"\–",value="-")

        # If lan-long isn't length 10, extend it up to length 10
        # dfN["latitude"] = dfN["latitude"].apply(lambda x: ("0"*(3-len(str(x).split(".")[0]))+str(x)) if(len(str(x).split(".")[0]) < 3) else str(x))
        dfN["latitude"] = dfN["latitude"].apply(lambda x: (str(x)+"0"*(6-len(str(x).split(".")[1]))) if(len(str(x).split(".")[1]) < 6) else str(x))

        # dfN["longitude"] = dfN["longitude"].apply(lambda x: ("0"*(3-len(str(x).split(".")[0]))+str(x)) if(len(str(x).split(".")[0]) < 3) else str(x))
        dfN["longitude"] = dfN["longitude"].apply(lambda x: (str(x)+"0"*(6-len(str(x).split(".")[1]))) if(len(str(x).split(".")[1]) < 6) else str(x))

        dfN["latitude"].replace(regex=True,inplace=True,to_replace=r"\.",value="")
        dfN["longitude"].replace(regex=True,inplace=True,to_replace=r"\.",value="")

        # Normally there's 2442 data points
        # Get the first 2440 values (so it's divisible by 5)
        max_values = 500
        super().__init__(max_num_data_groups=max_values, training_percentage=training_percentage,\
        partition_size=partition_size)
        self.data = [tuple(map(lambda y: int(y), x)) for x in dfN.values]

        # Some visualization code if you want to see how your data looks
        # Don't forget to import matplotlib
        #
        # scatter_x = np.array(list(map(lambda x: x[1:2][0], self.data)))
        # scatter_y = np.array(list(map(lambda x: x[:1][0], self.data)))
        # group = np.array(list(map(lambda x: x[2:3][0], self.data)))
        # cdict = {0: "blue", 1: "red"}
        #
        # fig, ax = plt.subplots()
        # for g in np.unique(group):
        #     ix = np.where(group == g)
        #     ax.scatter(scatter_x[ix], scatter_y[ix], c = cdict[g], label = g, s = 4)
        # ax.legend()
        # plt.show()

        random.shuffle(self.data)
        self.data = self.data[:max_values]
        self.shuffle()
        self.dps = 3
