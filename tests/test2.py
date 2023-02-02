import math
import itertools

l = [1, 2, 1, 3]
for e in itertools.combinations(l, 3):
    print(e)

s = "63133,6406,83237,1,95471,170.0"
s2 = s.split(",", maxsplit=2)
print(math.floor(2.7))

def bin_search(l, target):
    left = 0
    right = len(l)-1
    while left <= right:
        mid = left + math.floor((right-left)/2)
        if l[mid] < target:
            left = mid+1
        elif l[mid] > target:
            right = mid-1
        else:
            return mid-1
    return right

l = [1, 2, 5, 6, 7, 8, 9]
res = bin_search(l, 2)

import pandas as pd
# sample_df = pd.read_csv("C:\\data\\taobao_data\\sample_10.csv")
sample_df = pd.read_csv("C:\\data\\avazu-ctr-prediction\\train.csv", iterator=True)
sample_df = sample_df.read(5)
print(sample_df)
