import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import huffmancodec

def min_join_entropy(val1,val2):
    size = len(val1)
    if size == 0:
        size = 1
    joint_prob = {}

    for i in range(size):
        pair = (val1[i], val2[i])
        if pair in joint_prob:
            joint_prob[pair] += 1
        else:
            joint_prob[pair] = 1

    joint_entropy = 0
    for count in joint_prob.values():
        prob = count / size
        joint_entropy += - prob * np.log2(prob)

    return joint_entropy

def frequency(val,alfabeto):
    freq = np.zeros(alfabeto, dtype=np.uint16)
    for i in np.nditer(val):
        freq[i] += 1
    return freq

def binning(values,window):

    # For every window of values, find the most frequent value and replace all the values in the window with it
    # example: window number is 3, window of values is [0,1,2]
    # values are [1,0,4,2,1]
    # most frequent value is 1
    # replace all the [0,1,2] to 1
    # result is [1,1,4,1,1]
    # if window number is 40, window of values is [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14, ... , 39]

    freq = frequency(values,65536)

    new_values = np.zeros(values.size, dtype=np.uint16)

    for i in range(0,values.size,window):
        window_values = values[i:i+window]
        window_freq = freq[window_values]
        most_freq = np.argmax(window_freq)
        new_values[i:i+window] = most_freq
    return new_values

def plot_values(title,values):
    plt.xlabel(title)
    plt.ylabel("Count")
    freq = frequency(values,65536)
    freq = freq[freq>0]
    plt.bar(np.arange(freq.size), freq, color='red')
    plt.show()
