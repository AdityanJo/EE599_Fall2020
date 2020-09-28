## copyright, Keith Chugg
##  EE599, 2020

#################################################
## this is a template to illustrate hd5 files
##
## also can be used as template for HW1 problem
##################################################

import h5py
import numpy as np
import statistics
import math
import matplotlib.pyplot as plt


def runsTest(l, l_median):
    runs, n1, n2 = 0, 0, 0

    # Checking for start of new run
    for i in range(len(l)):

        # no. of runs
        if (l[i] >= l_median and l[i - 1] < l_median) or \
                (l[i] < l_median and l[i - 1] >= l_median):
            runs += 1

            # no. of positive values
        if (l[i]) >= l_median:
            n1 += 1

            # no. of negative values
        else:
            n2 += 1

    runs_exp = ((2 * n1 * n2) / (n1 + n2)) + 1
    stan_dev = math.sqrt((2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / \
                         (((n1 + n2) ** 2) * (n1 + n2 - 1)))

    z = (runs - runs_exp) / stan_dev

    return z

DEBUG = False
DATA_FNAME = 'adityan_jothi_hw1_1.hd5'

if DEBUG:
    num_sequences = 3
    sequence_length = 4
else:
    num_sequences = 25
    sequence_length = 20

### Enter your data here...
### Be sure to generate the data by hand.  DO NOT:
###     copy-n-paste
###     use a random number generator
###
if DEBUG:
    x_list = [
        [ 0, 1, 1, 0],
        [ 1, 1, 0, 0],
        [ 0, 0, 0, 1]
    ]
else:
    x_list=[
        [1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0], #alternates
        [1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0], #2 0s between 1s
        [1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0], #3 0s between 1s
        [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0], #4 0s between 1s
        [1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0], #5 0s between 1s
        [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1], #rev alternates
        [0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1], # 2 1s between 0s
        [0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1], # 3 1s betweem 0s
        [0,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1,1,1], # 4 1s between 0s
        [0,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,0,1], # 5 1s between 0s
        [0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1], # 00s 11s
        [0,0,0,1,1,1,0,0,0,1,1,1,0,0,0,1,1,1,0,0], # 000s 111s
        [0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0], # 0000s 1111s
        [0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,1,1,1,1,1], # 00000s 11111s
        [1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0], # 11s 00s
        [1,1,1,0,0,0,1,1,1,0,0,0,1,1,1,0,0,0,1,1], # 111s 000s
        [1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1], # 1111s 0000s
        [1,1,1,1,1,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0], # 11111s 00000s
        [1,0,1,0,1,1,0,1,1,0,1,0,1,1,0,1,1,0,1,0], # 1010 1101
        [1,1,0,1,1,0,1,0,1,1,0,1,1,0,1,0,1,1,0,1], # 1101 1010
        [1,0,1,0,1,0,1,1,1,0,1,0,1,0,1,1,1,0,1,0], # 1010 1011
        [1,0,1,1,1,0,1,0,1,0,1,1,1,0,1,0,1,0,1,1], # 1011 1010
        [0,0,1,1,1,0,1,0,0,0,1,1,1,0,1,0,0,0,1,1], # 0011 1010
        [1,0,1,0,0,0,1,1,1,0,1,0,0,0,1,1,1,0,1,0], # 1010 0011
        [1,1,0,1,0,0,0,0,1,1,0,1,0,0,0,0,1,1,0,1] # 1101 0000
     ]

    # x_list=np.random.normal(size=(25,20))
    # x_list=1/(1+np.exp(-x_list))
    # x_list=[[0 if el<0.5 else 1 for el in seq] for seq in x_list]

l_median= statistics.median(x_list)
Z = abs(runsTest(x_list, l_median))

print('Z-statistic= ', Z)
# convert list to a numpy array...
human_binary = np.asarray(x_list)

### do some error trapping:

assert human_binary.shape[0] == num_sequences, 'Error: the number of sequences was entered incorrectly'
assert human_binary.shape[1] == sequence_length, 'Error: the length of the seqeunces is incorrect'

# the with statement opens the file, does the business, and close it up for us...
with h5py.File(DATA_FNAME, 'w') as hf:
    hf.create_dataset('human_binary', data = human_binary)
    #hf.create_dataset('dummy',data=np.array([1,2,3,4,5]))
    ## note you can write several data arrays into one hd5 file, just give each a different name.

###################
# Let's read it back from the file and then check to make sure it is as we wrote...
with h5py.File(DATA_FNAME, 'r') as hf:
    hb_copy = hf['human_binary'][:]
    #dummy_copy = hf['dummy'][:]
    #print(dummy_copy)
data=hb_copy
assert all([all([ el in [0,1] and np.issubdtype(el, int) for el in seq ]) for seq in data])

### this will throw and error if they are not the same...
np.testing.assert_array_equal(human_binary, hb_copy)
#np.testing.assert_array_equal(np.array([1,2,3,4,5]),dummy_copy)
#np.testing.assert_array_equal(np.array([1,2,3,4,5,6]),dummy_copy)
