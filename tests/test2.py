import math
import itertools

# avazu
# fm: 0.7062
# afm: 0.6086
# afm 加权: 0.7295
# deepfm: 0.7480

dic1 = {'a':1, 'b':2}
dic2 = {'b':22, 'd':4}
# print(**dic1)
# print(**dic2)
print({**dic1, **dic2})
s1 = {1, 2, 3}
s2 = {2, 3, 8, 9}
s2 |= s1
print(s2)
l = [1, 2, 1, 3]
d = {(e, 0) for e in l}
print(d)
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

# import pandas as pd
# sample_df = pd.read_csv("C:\\data\\Ali-CCP\\sample_skeleton_train.csv")
# with open("C:\\data\\Ali-CCP\\sample_skeleton_train.csv", 'rt', encoding='ascii') as f:
#     for item in f:
#         if item.endswith("\n"):
#             item = item[:-1]
#         feature_list = item.split(",")[-1]
#         feature_list = feature_list.split(chr(1))
#         l = []
#         for e in feature_list:
#             tmp = e.split(chr(2))
#             tmp1 = tmp[1].split(chr(3))
#             if tmp[0] in ["508", "509", "702", "853"]:
#                 print((tmp[0], tmp1[0], tmp1[1]))
#             l.append((tmp[0], tmp1[0], tmp1[1]))
#         print(l)

with open("C:\\data\\Ali-CCP\\common_features_train.csv", 'rt', encoding='ascii') as f:
    d = {}
    i = 0
    max_feat = 0
    for item in f:
        i += 1
        if i % 100000 == 0:
            print(f"{i / 100000} * 10w")
        if item.endswith("\n"):
            item = item[:-1]
        feature_list = item.split(",")[-1]
        feature_list = feature_list.split(chr(1))
        # l = []
        for e in feature_list:
            tmp = e.split(chr(2))
            tmp1 = tmp[1].split(chr(3))
            if tmp[0] not in d:
                d[tmp[0]] = set()
            d[tmp[0]].add(tmp1[0])
            if int(tmp1[0]) > max_feat:
                max_feat = int(tmp1[0])
            # l.append((tmp[0], tmp1[0], tmp1[1]))
        # print(l)
    s = set()
    pv = 0
    for feat_field, feat in d.items():
        pv += len(feat)
        print(f"{feat_field} len: {len(feat)}")
        s |= feat
    print(f"uv all feat len: {len(s)}")
    print(f"pv all feat len: {pv}")
    print(f"max feat len: {max_feat}")
# sample_df = pd.read_csv("C:\\data\\avazu-ctr-prediction\\train.csv", iterator=True)
# sample_df = sample_df.read(5)
# print(sample_df)
