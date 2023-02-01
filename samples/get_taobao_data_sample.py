import math
base_dir = "C:\\data\\taobao_data"


def filter_behavior_log():
    i = 0
    with open(base_dir+"\\behavior_log.csv", 'r') as fr:
        with open(base_dir+"\\cvr_behavior_log.csv", 'w') as fw:
            for item in fr:
                i += 1
                if i % 10000000 == 0:
                    print(f"read {i/10000000} kw items")
                item_list = item.split(',')
                if "pv" not in item_list[2]:
                    fw.write(item)


def get_behavior_list():
    cvr_dict = {}  # 用户收藏序列 {user_id: [(time_stamp, cate, brand)]}
    with open(base_dir+"\\cvr_behavior_log.csv", 'r') as fr:
        i = 0
        for item in fr:
            i += 1
            if item.endswith("\n"):
                item = item[:-1]
            item = item.split(',')
            if i != 1 and len(item) == 5:
                if item[2] == "fav":
                    if item[0] not in cvr_dict:
                        cvr_dict[item[0]] = []
                    cvr_dict[item[0]].append((item[1], item[3], item[4]))
    print("read data completed!")
    with open(base_dir+"\\fav_behavior_feat_log.csv", 'w') as fw:
        fw.write("user,ts_cate_brand_list\n")
        for user, beh_list in cvr_dict.items():
            fav_list = sorted(beh_list, key=lambda x: x[0])
            fav_list = [":".join(e) for e in fav_list]
            seq = "|".join(fav_list)
            res = user+","+seq+"\n"
            fw.write(res)


def bin_search(l, target):
    left = 0
    right = len(l)-1
    while left <= right:
        mid = left + math.floor((right-left)/2)
        if l[mid][0] < target:
            left = mid+1
        elif l[mid][0] > target:
            right = mid-1
        else:
            return mid-1
    return right


def get_sample_with_feat():
    print("read ad_feature.csv")
    ad_feature_dict = {}
    with open(base_dir + "\\ad_feature.csv", 'r') as fr:
        i = 0
        for item in fr:
            i += 1
            if item.endswith("\n"):
                item = item[:-1]
            item = item.split(",", maxsplit=1)
            if i != 1 and len(item) == 2:
                ad_feature_dict[item[0]] = item[1]

    print("read user_profile.csv")
    user_profile_dict = {}
    with open(base_dir + "\\user_profile.csv", 'r') as fr:
        i = 0
        for item in fr:
            i += 1
            if item.endswith("\n"):
                item = item[:-1]
            item = item.split(",", maxsplit=1)
            if i != 1 and len(item) == 2:
                user_profile_dict[item[0]] = item[1]

    print("read fav_behavior_feat_log.csv")
    fav_seq_dict = {}
    with open(base_dir + "\\fav_behavior_feat_log.csv", 'r') as fr:
        i = 0
        for item in fr:
            i += 1
            if item.endswith("\n"):
                item = item[:-1]
            item = item.split(",", maxsplit=1)
            if i != 1 and len(item) == 2:
                fav_seq_dict[item[0]] = item[1]

    print("read raw_sample.csv")
    with open(base_dir + "\\sample.csv", 'w') as fw:
        fw.write("user_id,time_stamp,adgroup_id,pid,nonclk,clk,"
                 "tag_category_list,tag_brand_list,"
                 "cms_segid,cms_group_id,final_gender_code,age_level,pvalue_level,shopping_level,occupation,new_user_class_level,"
                 "cate_id,campaign_id,customer,brand,price\n")
        with open(base_dir + "\\raw_sample.csv", 'r') as fr:
            i = 0
            for item in fr:
                i += 1
                if item.endswith("\n"):
                    item = item[:-1]
                res = item
                item = item.split(",", maxsplit=3)
                if len(item) == 4 and i != 1:
                    user = item[0]
                    time_stamp = item[1]
                    adgroup_id = item[2]
                    raw_other = item[3]
                    user_profile = user_profile_dict.get(user, ",,,,,,,")
                    ad_feature = ad_feature_dict.get(adgroup_id, ",,,,")
                    fav_seq = fav_seq_dict.get(user, "")
                    cate = []
                    brand = []
                    if fav_seq != "":
                        fav_seq = fav_seq.split("|")
                        fav_seq = [e.split(":") for e in fav_seq]
                        index = bin_search(fav_seq, time_stamp)
                        j = 0
                        while index >= 0 and j < 5:
                            cate.append(fav_seq[index][1])
                            brand.append(fav_seq[index][2])
                            index -= 1
                            j += 1
                    cate = "|".join(cate)
                    brand = "|".join(brand)
                    res += f",{cate},{brand},{user_profile},{ad_feature}\n"
                    fw.write(res)


if __name__ == '__main__':
    # filter_behavior_log()
    # get_behavior_list()
    get_sample_with_feat()
